"""Banking chat with explicit service checks, bounded history and audio deduplication."""
import hashlib
import logging
import re
import uuid
from io import BytesIO
from typing import Any
import streamlit as st
from src.config import get_settings
from src.sumopod_services import ServiceError, get_embedding, generate_grounded_response, transcribe_audio, require_asr_model
from src.vector_db import collection_health, search_documents

logger = logging.getLogger(__name__)
NO_CONTEXT = "I could not find supporting information in the banking knowledge base. Please rephrase your question or contact the bank for help."


class UserMessageError(Exception):
    """A fixed, safe message intended for the interface."""


def _initialize_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("audio_epoch", 0)
    st.session_state.setdefault("processed_audio_hashes", [])
    st.session_state.setdefault("voice_draft", None)
    st.session_state.setdefault("chat_ui", {})
    ui = st.session_state.chat_ui
    for key, value in (("epoch", 0), ("pending_action", None), ("active_request", None), ("edit_turn_id", None), ("notice", None)):
        ui.setdefault(key, value)
    turn_id = None
    for message in st.session_state.chat_history:
        if message.get("role") == "user":
            turn_id = message.setdefault("turn_id", uuid.uuid4().hex)
        else:
            message.setdefault("turn_id", turn_id or uuid.uuid4().hex)
        message.setdefault("id", f"{message.get('role')}:{message['turn_id']}")
        message.setdefault("status", "success")
        message.setdefault("attempt", 1)
    st.session_state.pop("voice_notice", None)
    if not ui["pending_action"] and not ui["active_request"]:
        for message in st.session_state.chat_history:
            if message.get("status") == "pending":
                message["status"] = "error"
                if message.get("role") == "assistant":
                    message.update(content="This request was interrupted. Retry it when you are ready.", sources=[])
        draft = st.session_state.voice_draft
        if draft and draft["status"] == "pending":
            draft["status"] = "untranscribed"


def _chat_is_busy() -> bool:
    ui = st.session_state.get("chat_ui", {})
    return bool(ui.get("pending_action") or ui.get("active_request"))


def _successful_history(history):
    rejected = {"pending", "error", "interrupted", "cancelled"}
    failed = {m.get("turn_id") for m in history if m.get("turn_id") and m.get("status") in rejected}
    indexes = set()
    for index, message in enumerate(history):
        if message.get("role") == "assistant" and message.get("status") in rejected:
            indexes.add(index)
            if index and history[index - 1].get("role") == "user":
                indexes.add(index - 1)
    return [m for index, m in enumerate(history) if index not in indexes
        and m.get("turn_id") not in failed and m.get("status") not in rejected]


def _bounded_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    limit = get_settings().max_history_messages
    if not limit:
        return []
    messages = []
    for item in reversed(_successful_history(history)):
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))[:4000]})
            if len(messages) == limit:
                break
    return list(reversed(messages))


def _retrieval_query(query: str, history: list[dict[str, Any]]) -> str:
    """Resolve short referring follow-ups using one prior user turn, without another model call."""
    tokens = set(re.findall(r"\w+", query.lower()))
    referring = {"it", "that", "those", "them", "earlier", "previous", "tadi", "sebelumnya", "tersebut", "itu", "nya"}
    if len(query.split()) <= 18 and tokens & referring:
        previous = next((item["content"] for item in reversed(_successful_history(history)) if item.get("role") == "user"), "")
        if previous:
            return f"Previous question: {previous[:2000]}\nFollow-up question: {query}"
    return query


def _show_sources(documents: list[dict[str, Any]]) -> None:
    if not documents:
        st.caption("No supporting documents retrieved.")
        return
    with st.expander(f"Sources ({len(documents)})"):
        for i, doc in enumerate(documents, 1):
            text = str(doc.get("text", "")).strip()
            title = doc.get("title")
            first_line, _, body = text.partition("\n")
            if not title and first_line.startswith("[KATEGORI:") and body.strip():
                title = first_line.removeprefix("[KATEGORI:").removesuffix("]").strip()
                text = body.strip()
            title = title or "Bank policy"
            st.markdown(f"**[{i}] {str(title)[:100]}**")
            st.write(text)
            st.caption(f"Source: {doc.get('source') or 'Bank policy'}")
            with st.popover("Technical source details"):
                st.json({key: doc.get(key) for key in ("id", "kb_version", "score", "intent_ids")})
                st.caption("Similarity is a retrieval signal, not a guarantee of answer correctness.")


def _refresh_health() -> dict[str, Any]:
    try:
        health = collection_health()
    except Exception as exc:
        logger.warning("Collection health check failed (%s)", type(exc).__name__)
        health = {"status": "error"}
    st.session_state.database_health = health
    return health


def _health_is_ready(health: dict[str, Any]) -> bool:
    return health.get("status") == "ready" and health.get("points_count", 0) > 0


def _render_health() -> None:
    st.button("Refresh database status", key="refresh_database", on_click=_refresh_health, disabled=_chat_is_busy())
    health = st.session_state.get("database_health")
    if health is None:
        st.caption("Policy availability will be checked when you ask a question.")
    elif _health_is_ready(health):
        st.success(f"Knowledge base ready · {health['points_count']} documents")
        if not health.get("labelled", False):
            st.warning("This collection has no verified intent labels. Retrieval relevance may be unscorable.")
    else:
        messages = {
            "missing": "The configured knowledge base is missing. Ingest the bank policy first.",
            "empty": "The knowledge base is empty. Ingest the bank policy first.",
            "incompatible": "The knowledge base does not match the configured embedding model. Re-ingest it with matching settings.",
        }
        if health.get("reason") == "embedding_provider":
            st.error("Re-ingest the bank policy to build a knowledge base for the current embedding provider.")
        else:
            st.error(messages.get(health.get("status"), "The knowledge base is unavailable. Check its configuration and refresh the status."))


def _process_query(query: str, history: list[dict[str, Any]] | None = None, progress=None) -> dict[str, Any]:
    history = _successful_history(history or [])
    report = progress or (lambda stage: None)
    report("Checking the policy knowledge base…")
    if not _health_is_ready(_refresh_health()):
        raise UserMessageError("The knowledge base is not ready. Check its status and ingest the policy before asking a question.")
    retrieval_query = _retrieval_query(query, history)
    settings = get_settings()
    report("Finding supporting policy sources…")
    documents = search_documents(query_vector=get_embedding(retrieval_query), top_k=settings.retrieval_top_k, score_threshold=settings.retrieval_score_threshold)
    documents = [doc for doc in documents if str(doc.get("text", "")).strip()]
    if not documents:
        return {"content": NO_CONTEXT, "sources": [], "retrieval_query": retrieval_query}
    report("Writing an answer from the sources…")
    answer = generate_grounded_response(prompt=query, documents=documents, history=_bounded_history(history))
    if not isinstance(answer, str) or not answer.strip():
        raise UserMessageError("No answer was returned. Retry the request or edit your question.")
    return {"content": answer, "sources": documents, "retrieval_query": retrieval_query}


def _clear_chat() -> None:
    _initialize_session_state()
    st.session_state.chat_history = []
    st.session_state.processed_audio_hashes = []
    st.session_state.audio_epoch += 1
    st.session_state.voice_draft = None
    ui = st.session_state.chat_ui
    ui.update(epoch=ui["epoch"] + 1, pending_action=None, active_request=None, edit_turn_id=None, notice=None)
    st.session_state.pop("voice_notice", None)


def _asr_configuration_notice() -> str | None:
    """Check voice configuration locally without creating a client or making a request."""
    if not get_settings().asr_model:
        return "Voice input and ASR experiments are unavailable. Set ASR_MODEL to a transcription model supported by your SumoPod account. Text input remains available."
    try:
        require_asr_model()
    except ValueError:
        return "Configure the SumoPod API_KEY before using voice input or ASR experiments."
    return None


def _render_audio_input():
    key = f"audio_recorder_{st.session_state.audio_epoch}"
    disabled = _asr_configuration_notice() is not None or _chat_is_busy()
    if hasattr(st, "audio_input"):
        return st.audio_input("Record your question", key=key, disabled=disabled, on_change=_recording_changed, args=(key,))
    return st.file_uploader("Upload a question", type=["wav", "mp3", "m4a", "ogg", "webm"], key=key, disabled=disabled, on_change=_recording_changed, args=(key,))


def cancel_pending_chat() -> None:
    """Cancel queued work on navigation without discarding local drafts."""
    _initialize_session_state()
    ui = st.session_state.chat_ui
    ui.update(epoch=ui["epoch"] + 1, pending_action=None, active_request=None)
    for message in st.session_state.chat_history:
        if message.get("status") == "pending":
            message["status"] = "error"
            if message.get("role") == "assistant":
                message.update(content="This request was interrupted. Retry it when you are ready.", sources=[])
    draft = st.session_state.voice_draft
    if draft and draft["status"] == "pending":
        draft["status"] = "untranscribed"


def _turn_pair(turn_id):
    history = st.session_state.chat_history
    for index, message in enumerate(history):
        if message.get("role") == "user" and message.get("turn_id") == turn_id:
            return index, message, history[index + 1]
    raise ValueError("This conversation turn is unavailable.")


def _latest_failed_turn():
    history = st.session_state.chat_history
    if history and history[-1].get("role") == "assistant" and history[-1].get("status") == "error":
        return history[-1]["turn_id"]
    return None


def _submit_query(query: str, origin="text") -> None:
    """Queue locally; the rendering pass performs any provider request."""
    _initialize_session_state()
    if _chat_is_busy() or not isinstance(query, str) or not query.strip():
        return
    query = query.strip()
    ui = st.session_state.chat_ui
    if len(query) > get_settings().max_context_chars:
        ui["notice"] = "Your question is too long. Shorten it and try again."
        return
    turn_id = uuid.uuid4().hex
    st.session_state.chat_history.extend([
        dict(id=f"user:{turn_id}", turn_id=turn_id, role="user", content=query,
             status="pending", attempt=1, origin=origin),
        dict(id=f"assistant:{turn_id}", turn_id=turn_id, role="assistant", content="",
             status="pending", attempt=1, sources=[]),
    ])
    ui.update(pending_action=dict(id=uuid.uuid4().hex, kind="chat", turn_id=turn_id, epoch=ui["epoch"]),
              edit_turn_id=None, notice=None)
    st.session_state.pop("voice_notice", None)


def _on_text_submit():
    _submit_query(st.session_state.get("chat_text_input", ""))


def _prefill_example(query):
    if not _chat_is_busy():
        st.session_state["chat_text_input"] = query


def _queue_retry(turn_id, edited_text=None):
    if _chat_is_busy() or turn_id != _latest_failed_turn():
        return
    _, user, assistant = _turn_pair(turn_id)
    query = user["content"] if edited_text is None else edited_text.strip()
    if not query or len(query) > get_settings().max_context_chars:
        st.session_state.chat_ui["notice"] = "Enter a shorter, non-empty question before sending."
        return
    attempt = assistant.get("attempt", 1) + 1
    user.update(content=query, status="pending", attempt=attempt)
    assistant.update(content="", sources=[], status="pending", attempt=attempt)
    ui = st.session_state.chat_ui
    ui.update(pending_action=dict(id=uuid.uuid4().hex, kind="chat", turn_id=turn_id, epoch=ui["epoch"]),
              edit_turn_id=None, notice=None)


def _begin_edit(turn_id):
    if not _chat_is_busy() and turn_id == _latest_failed_turn():
        _, user, _ = _turn_pair(turn_id)
        st.session_state.chat_ui.update(edit_turn_id=turn_id, edit_text=user["content"])


def _save_edit(key):
    st.session_state.chat_ui["edit_text"] = st.session_state.get(key, "")


def _send_edit(turn_id, key):
    _queue_retry(turn_id, st.session_state.get(key, ""))


def _cancel_edit():
    st.session_state.chat_ui["edit_turn_id"] = None


def _queue_transcription(draft_id):
    draft = st.session_state.voice_draft
    if not draft or draft["id"] != draft_id or not draft.get("bytes") or _chat_is_busy() or _asr_configuration_notice():
        return
    ui = st.session_state.chat_ui
    draft.update(status="pending", error=None)
    ui["pending_action"] = dict(id=uuid.uuid4().hex, kind="asr", draft_id=draft_id, epoch=ui["epoch"])


def _capture_recording(audio, explicit=False):
    if audio is None or _asr_configuration_notice():
        return
    data = audio.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    audio_key = f"{st.session_state.audio_epoch}:{digest}"
    seen = st.session_state.processed_audio_hashes
    if audio_key in seen or (not explicit and digest in seen):
        return
    st.session_state.processed_audio_hashes = (st.session_state.processed_audio_hashes + [audio_key])[-100:]
    draft = dict(id=uuid.uuid4().hex, audio_key=audio_key, bytes=data, revision=0,
                 filename=getattr(audio, "name", "recording.wav"), text="", status="untranscribed")
    st.session_state.voice_draft = draft
    if not data or len(data) > get_settings().max_audio_bytes:
        draft.update(status="error", error="Choose a non-empty recording within the audio upload limit.", bytes=b"")
        return
    _queue_transcription(draft["id"])


def _recording_changed(key):
    _capture_recording(st.session_state.get(key), explicit=True)


def _save_voice_text(draft_id, key):
    draft = st.session_state.voice_draft
    if draft and draft["id"] == draft_id:
        draft["text"] = st.session_state.get(key, "")


def _discard_voice():
    if not _chat_is_busy():
        st.session_state.voice_draft = None
        st.session_state.audio_epoch += 1


def _confirm_voice(draft_id, key):
    draft = st.session_state.voice_draft
    if not draft or draft["id"] != draft_id or _chat_is_busy():
        return
    text = st.session_state.get(key, draft.get("text", "")).strip()
    if text:
        _submit_query(text, origin="voice")
        if st.session_state.chat_ui.get("pending_action"):
            st.session_state.voice_draft = None
            st.session_state.audio_epoch += 1


def _render_voice():
    draft = st.session_state.voice_draft
    notice = _asr_configuration_notice()
    if notice:
        st.caption("Voice input unavailable. See Setup in the sidebar.")
        return st.empty()
    with st.expander("Voice input", expanded=bool(draft)):
        st.caption("Recording creates a transcript. Check it before sending your question.")
        audio = _render_audio_input()
        # Retained recorder values and injected recorder seams are deduplicated.
        _capture_recording(audio)
        draft = st.session_state.voice_draft
        slot = st.empty()
        if not draft:
            st.caption("Your transcript will appear here before it is sent.")
            return slot
        if draft.get("error"):
            st.error(draft["error"])
        key = f"voice_transcript_{draft['id']}_{draft.get('revision', 0)}"
        if draft.get("text") or draft["status"] == "ready":
            if key not in st.session_state:
                st.session_state[key] = draft.get("text", "")
            st.text_area("Check or edit the transcript", key=key,
                max_chars=get_settings().max_context_chars, disabled=_chat_is_busy(),
                on_change=_save_voice_text, args=(draft["id"], key))
            st.button("Send question", key="send_voice", type="primary",
                disabled=_chat_is_busy() or not st.session_state.get(key, "").strip(),
                on_click=_confirm_voice, args=(draft["id"], key))
        st.button("Transcribe recording" if draft["status"] == "untranscribed" else "Retry voice transcription",
            key="retry_voice", disabled=_chat_is_busy() or bool(notice) or not draft.get("bytes"),
            on_click=_queue_transcription, args=(draft["id"],))
        st.button("Discard recording", key="discard_voice", disabled=_chat_is_busy(), on_click=_discard_voice)
        return slot


def _render_history():
    slots = {}
    latest_failed = _latest_failed_turn()
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("status") == "pending":
                slots[message["turn_id"]] = st.empty()
                continue
            st.markdown(message["content"])
            if message["role"] != "assistant":
                continue
            if message.get("status") == "success":
                _show_sources(message.get("sources", []))
            elif message.get("status") == "error":
                st.caption("Request not completed.")
                if message["turn_id"] == latest_failed:
                    left, right = st.columns(2)
                    left.button("Retry answer", key=f"retry_answer_{latest_failed}",
                        disabled=_chat_is_busy(), on_click=_queue_retry, args=(latest_failed,))
                    right.button("Edit question", key=f"edit_answer_{latest_failed}",
                        disabled=_chat_is_busy(), on_click=_begin_edit, args=(latest_failed,))
    return slots


def _render_edit():
    ui = st.session_state.chat_ui
    if not (turn_id := ui.get("edit_turn_id")):
        return
    key = f"edit_question_{turn_id}"
    if key not in st.session_state:
        st.session_state[key] = ui.get("edit_text", "")
    st.text_area("Edit your question", key=key, max_chars=get_settings().max_context_chars,
        on_change=_save_edit, args=(key,))
    left, right = st.columns(2)
    left.button("Send edited question", key="send_edited_question", type="primary",
        on_click=_send_edit, args=(turn_id, key), disabled=_chat_is_busy() or not st.session_state[key].strip())
    right.button("Cancel edit", key="cancel_chat_edit", on_click=_cancel_edit)


def _safe_request_error(exc):
    if isinstance(exc, UserMessageError):
        return str(exc)
    if isinstance(exc, ServiceError):
        if exc.status_code in (401, 403):
            return "The answer service could not authenticate this request. Ask the app administrator to check the API key."
        if exc.status_code == 429:
            return "The answer service is busy or its request allowance has been reached. Try again shortly."
        if "Timeout" in exc.error_type:
            return "The answer took too long. Try again."
    return "The request could not be completed. Try again or edit your question."


def _run_pending_action(slots, voice_slot):
    ui = st.session_state.chat_ui
    action = ui.get("pending_action")
    if not action or ui.get("active_request"):
        return
    ui["pending_action"] = None
    if action["epoch"] != ui["epoch"]:
        return
    ui["active_request"] = action["id"]
    try:
        if action["kind"] == "chat":
            index, user, assistant = _turn_pair(action["turn_id"])
            history = _successful_history(st.session_state.chat_history[:index])
            with slots[action["turn_id"]].container():
                with st.status("Checking the policy knowledge base…", expanded=False) as status:
                    try:
                        result = _process_query(user["content"], history,
                            progress=lambda stage: status.update(label=stage))
                        if action["epoch"] == ui["epoch"]:
                            user["status"] = "success"
                            assistant.update(result, status="success")
                        status.update(label="Answer ready", state="complete")
                    except Exception as exc:
                        logger.warning("Chat request failed (%s)", type(exc).__name__)
                        if action["epoch"] == ui["epoch"]:
                            user["status"] = "error"
                            assistant.update(content=_safe_request_error(exc), sources=[], status="error")
                        status.update(label="Request not completed", state="error")
        else:
            draft = st.session_state.voice_draft
            if draft and draft["id"] == action["draft_id"]:
                with voice_slot.container(), st.status("Transcribing your recording…", expanded=False) as status:
                    try:
                        require_asr_model()
                        buffer = BytesIO(draft["bytes"])
                        buffer.name = draft["filename"]
                        text = transcribe_audio(buffer).strip()
                        if not text:
                            raise UserMessageError("No speech was recognized. Record again or retry transcription.")
                        draft.update(text=text, transcript=text, status="ready", error=None,
                            revision=draft.get("revision", 0) + 1)
                        status.update(label="Transcript ready to check", state="complete")
                    except Exception as exc:
                        logger.warning("Audio transcription failed (%s)", type(exc).__name__)
                        draft.update(status="error", error=str(exc) if isinstance(exc, UserMessageError)
                            else "Transcription could not be completed. Retry it or type your question.")
                        status.update(label="Transcription not completed", state="error")
    finally:
        ui["active_request"] = None
    # The action was consumed before I/O. This render re-enables controls without replaying it.
    st.rerun()


def _render_chat_content(health_panel):
    st.header("Live Banking Assistant")
    st.caption("Ask about balance checks, cards, transfers and other guidance in the policy documents.")
    if not st.session_state.chat_history:
        st.write("Choose an example, then edit or send it.")
        examples = ("Bagaimana cara mengecek saldo?", "Apa yang harus dilakukan jika kartu hilang?")
        for index, (column, example) in enumerate(zip(st.columns(2), examples)):
            column.button(example, key=f"chat_example_{index}", on_click=_prefill_example,
                args=(example,), width="stretch", disabled=_chat_is_busy())
    slots = _render_history()
    _render_edit()
    voice_slot = _render_voice()
    st.chat_input("Ask a policy question…", key="chat_text_input", max_chars=get_settings().max_context_chars,
        on_submit=_on_text_submit, disabled=_chat_is_busy() or bool(st.session_state.chat_ui.get("edit_turn_id")))
    with health_panel:
        _render_health()
    if notice := st.session_state.chat_ui.get("notice"):
        st.warning(notice)
    _run_pending_action(slots, voice_slot)


def render_live_demo_tab() -> None:
    _initialize_session_state()
    with st.sidebar:
        health_panel = st.container()
        st.button("Clear chat history", key="clear_chat", on_click=_clear_chat,
            disabled=not st.session_state.chat_history and st.session_state.voice_draft is None)
    with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(width=760):
            _render_chat_content(health_panel)
