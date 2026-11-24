"""Explicit research journeys with reproducible runs and persistent, inspectable results."""
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import pandas as pd
import streamlit as st
from src.config import get_settings
from src.metrics import calculate_wer, calculate_raw_wer, calculate_cer, cosine_similarity_score
from src.minds14_loader import MINDS14Loader, get_available_languages
from src.sumopod_services import get_embedding, generate_grounded_response, transcribe_audio, require_asr_model
from src.vector_db import search_documents
from src.tabs.batch_evaluation import render_batch_evaluation_tab
from src.tabs.live_demo import NO_CONTEXT, UserMessageError, _show_sources, _refresh_health, _health_is_ready, _render_health, _asr_configuration_notice
from src.tabs.ui_state import _spreadsheet_safe_csv, model_configuration, navigation_choice, remember_widget, saved_widget_value

logger = logging.getLogger(__name__)


def _model_configuration() -> dict[str, Any]:
    return model_configuration(get_settings(), st.session_state.get("database_health", {}))


def _safe_error(message: str, exc: Exception) -> None:
    logger.warning("Research action failed (%s)", type(exc).__name__)
    st.error(message)


def _get_dataset_loader(language: str) -> MINDS14Loader:
    key = f"minds14_loader_{language}"
    if key not in st.session_state:
        st.session_state[key] = MINDS14Loader(language=language)
    return st.session_state[key]


def _run_text_pipeline(text: str, collection_name: str | None = None) -> dict[str, Any]:
    embedding = get_embedding(text)
    settings = get_settings()
    docs = search_documents(query_vector=embedding, top_k=settings.retrieval_top_k,
                            score_threshold=settings.retrieval_score_threshold, collection_name=collection_name)
    docs = [doc for doc in docs if str(doc.get("text", "")).strip()]
    response = generate_grounded_response(prompt=text, documents=docs, history=[]) if docs else NO_CONTEXT
    return {"transcript": text, "embedding": embedding, "retrieved_docs": docs, "llm_response": response}


def _run_voice_pipeline(audio_buffer, collection_name: str | None = None) -> dict[str, Any]:
    require_asr_model()
    transcript = transcribe_audio(audio_buffer).strip()
    if not transcript:
        raise UserMessageError("No speech was recognized. Choose another sample or retry the experiment.")
    return _run_text_pipeline(transcript, collection_name)


def _sample_controls():
    languages = get_available_languages()
    language_key = "research_language_select"
    saved_locale = saved_widget_value(language_key, "en-US", allowed=languages)
    language = st.selectbox(
        "Dataset locale", languages, index=languages.index(saved_locale), key=language_key,
        on_change=remember_widget, args=(language_key,),
    )
    loader = st.session_state.get(f"minds14_loader_{language}")
    if loader is None and st.button("Load dataset", key="load_single_dataset", type="primary"):
        try:
            with st.spinner("Loading dataset…"):
                loader = _get_dataset_loader(language)
        except Exception as exc:
            _safe_error("The dataset could not be loaded. Check dataset availability and the configured revision, then retry.", exc)
    if loader is None:
        st.write("Load the dataset to preview an audio sample. An uncached dataset may need to download first.")
        return None, None, None, language
    try:
        size = len(loader)
        if size == 0:
            st.warning("This dataset contains no samples.")
            return None, None, None, language
        sample_key = f"research_sample_index_{language}"
        saved_index = saved_widget_value(sample_key, 0, allowed=range(size))
        index = st.slider(
            "Sample index", 0, size - 1, saved_index, key=sample_key,
            on_change=remember_widget, args=(sample_key,),
        ) if size > 1 else 0
        sample = loader.get_sample(index)
        st.write(f"Sample `{sample['id']}` · Intent: {sample['intent_name']}")
        st.write(sample["transcription"])
        if sample.get("english_transcription") and sample["english_transcription"] != sample["transcription"]:
            st.caption(f"English translation (display only): {sample['english_transcription']}")
        st.audio(sample["audio_path"], format="audio/wav")
        return loader, sample, index, language
    except Exception as exc:
        _safe_error("This sample could not be decoded. Choose another sample or dataset locale.", exc)
        return None, None, None, language


def _render_last_experiment(marker: dict[str, Any] | None) -> None:
    attempt = st.session_state.get("last_single_attempt", {})
    result = st.session_state.get("last_experiment")
    if result is None:
        if attempt.get("error"):
            st.error(attempt["error"])
        return
    st.subheader("Experiment results")
    if attempt.get("error"):
        st.warning(f"Latest attempt failed for {attempt['locale']} · `{attempt['sample_id']}`. {attempt['error']} Showing the last successful result below.")
    if marker != result["marker"]:
        st.info("Showing a previous experiment. Its sample, language or model configuration differs from the current selection.")
    finished = datetime.fromisoformat(result["created_utc"]).astimezone(timezone.utc)
    st.write(f"{result['marker']['locale']} · Sample `{result['marker']['sample_id']}` · {finished:%d %b %Y, %H:%M UTC}")
    cer = "Not measured" if result.get("cer") is None else f"{result['cer']:.2%}"
    st.write(f"**Word error rate (WER)** {result['wer']:.2%} · **Character error rate (CER)** {cer}")
    st.markdown("#### Transcripts")
    labels = ("Reference transcript", "ASR transcript")
    branches = (result["text_result"], result["voice_result"])
    for column, label, branch in zip(st.columns(2), labels, branches):
        with column:
            st.markdown(f"**{label}**")
            st.write(branch["transcript"])
    st.markdown("#### Answers")
    for column, label, branch in zip(st.columns(2), ("Text answer", "Voice answer"), branches):
        with column:
            st.markdown(f"**{label}**")
            st.write(branch["llm_response"])
            _show_sources(branch["retrieved_docs"])
    with st.expander("Metrics and experiment configuration"):
        st.write(f"Raw WER: {result['wer_raw']:.2%} · Embedding similarity: {result['similarity']:.4f}")
        st.write("WER and CER measure transcription errors and can exceed 100%. Normalized CER omits whitespace. Answer correctness requires separate review.")
        st.json(result["marker"])


def _render_single_sample_analysis() -> None:
    st.subheader("Single sample analysis")
    st.write("Compare the reference transcript with speech recognition on the same sample. Both use the original language.")
    asr_notice = _asr_configuration_notice()
    if asr_notice:
        st.info("Speech recognition is unavailable. You can still preview a sample. See Setup in the sidebar.")
    loader, sample, index, language = _sample_controls()
    marker = {"sample_id": sample["id"], "locale": language, **_model_configuration()} if sample else None
    if sample is not None and st.button("Run experiment", key="run_single_experiment", type="primary", disabled=asr_notice is not None):
        attempt = {"sample_id": sample["id"], "locale": language}
        with st.status("Checking the knowledge base…", expanded=False) as progress:
            try:
                require_asr_model()
                health = _refresh_health()
                if not _health_is_ready(health):
                    raise UserMessageError("The knowledge base is unavailable. Check Setup in the sidebar and try again.")
                physical = health.get('physical_collection', get_settings().qdrant_collection)
                progress.update(label="Running the reference transcript…")
                text_result = _run_text_pipeline(sample["transcription"], physical)
                progress.update(label="Transcribing and running the voice comparison…")
                voice_result = _run_voice_pipeline(loader.get_audio_bytes(index, sample=sample), physical)
                marker = {**attempt, **_model_configuration()}
                st.session_state.last_experiment = {
                    "id": uuid4().hex, "marker": marker, "sample": sample,
                    "ground_truth": sample["transcription"], "text_result": text_result,
                    "voice_result": voice_result,
                    "wer": calculate_wer(sample["transcription"], voice_result["transcript"]),
                    "wer_raw": calculate_raw_wer(sample["transcription"], voice_result["transcript"]),
                    "cer": calculate_cer(sample["transcription"], voice_result["transcript"]),
                    "similarity": cosine_similarity_score(text_result["embedding"], voice_result["embedding"]),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
                progress.update(label="Experiment complete", state="complete")
            except UserMessageError as exc:
                attempt["error"] = str(exc)
            except Exception as exc:
                logger.warning("Single sample experiment failed (%s)", type(exc).__name__)
                attempt["error"] = "Check the model and database settings, then retry."
            if attempt.get("error"):
                progress.update(label="Experiment failed", state="error")
        st.session_state.last_single_attempt = attempt
    _render_last_experiment(marker)


def _review_candidates() -> dict[str, dict[str, Any]]:
    candidates = {}
    experiment = st.session_state.get("last_experiment")
    if experiment:
        experiment_id = experiment.get("id", experiment["created_utc"])
        marker = experiment["marker"]
        for branch in ("text", "voice"):
            result = experiment[f"{branch}_result"]
            candidates[f"Last experiment · {branch}"] = {
                "experiment_id": experiment_id, "locale": marker["locale"],
                "sample_id": marker["sample_id"], "branch": branch,
                "query": result["transcript"], "answer": result["llm_response"],
                "sources": result["retrieved_docs"],
            }
    history = st.session_state.get("chat_history", [])
    for index in range(len(history) - 1, 0, -1):
        question, answer = history[index - 1:index + 1]
        if (question.get("role") == "user" and answer.get("role") == "assistant"
                and answer.get("status", "success") == "success" and answer.get("content")):
            candidates["Latest chat answer"] = {
                "experiment_id": answer.get("turn_id", f"chat:{index}"),
                "locale": "", "sample_id": "", "branch": question.get("origin", "chat"),
                "query": question["content"], "answer": answer["content"],
                "sources": answer.get("sources", []),
            }
            break
    return candidates


def _review_score(label: str, key: str) -> str:
    choices = ("Not reviewed", "0", "1", "2")
    saved = saved_widget_value(key, choices[0], allowed=choices)
    return st.selectbox(label, choices, index=choices.index(saved), key=key,
                        on_change=remember_widget, args=(key,))


def _render_answer_quality_review() -> None:
    st.subheader("Manual answer review")
    candidates = _review_candidates()
    if not candidates:
        st.info("Run a single sample experiment or ask the assistant a question to get an answer to review.")
    else:
        _render_review_draft(candidates)
    records = st.session_state.get("answer_quality_reviews", [])
    if records:
        frame = pd.DataFrame(records)
        st.subheader(f"Saved reviews ({len(records)})")
        columns = ["candidate_source", "support", "correctness", "relevance", "reviewer"]
        st.dataframe(frame[columns], width="stretch", hide_index=True, column_config={
            "candidate_source": "Answer", "support": "Source support", "correctness": "Correctness",
            "relevance": "Relevance", "reviewer": "Reviewer",
        })
        st.download_button("Download reviews CSV", _spreadsheet_safe_csv(frame), file_name="manual_answer_reviews.csv",
                           mime="text/csv", key="manual_reviews_csv", on_click="ignore")


def _render_review_draft(candidates: dict[str, dict[str, Any]]) -> None:
    key = "quality_candidate"
    choices = list(candidates)
    saved = saved_widget_value(key, choices[0], allowed=choices)
    selected = st.selectbox("Answer to review", choices, index=choices.index(saved), key=key,
                            on_change=remember_widget, args=(key,))
    candidate = candidates[selected]
    candidate_id = hashlib.sha256(json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    if candidate["sample_id"]:
        st.write(f"{candidate['locale']} · Sample `{candidate['sample_id']}` · Experiment `{candidate['experiment_id']}`")
    st.markdown("**Question**")
    st.write(candidate["query"])
    st.markdown("**Answer**")
    st.write(candidate["answer"])
    _show_sources(candidate["sources"])
    st.markdown("#### Review rubric")
    st.write("Score each item from 0 (unsupported, incorrect, or off topic) to 2 (fully supported, correct, or relevant). Use 1 for partial credit.")
    st.write("Reviews are saved for this session. Download the CSV to keep a copy.")
    reviewer_key = "quality_reviewer"
    reviewer = st.text_input("Reviewer", value=saved_widget_value(reviewer_key, ""), max_chars=80, key=reviewer_key,
                             on_change=remember_widget, args=(reviewer_key,))
    draft = f"review_{candidate_id}"
    scores = {
        "support": _review_score("Support from displayed sources", f"{draft}_support"),
        "correctness": _review_score("Factual correctness", f"{draft}_correctness"),
        "relevance": _review_score("Relevance to the question", f"{draft}_relevance"),
    }
    notes_key = f"{draft}_notes"
    notes = st.text_area("Evidence and notes", value=saved_widget_value(notes_key, ""), max_chars=4000, key=notes_key,
                         on_change=remember_widget, args=(notes_key,))
    if st.button("Save review", key="save_quality_review", type="primary"):
        if "Not reviewed" in scores.values():
            st.error("Select all three rubric scores before saving the review.")
            return
        sources = candidate["sources"]
        record = {
            "candidate_id": candidate_id, "candidate_source": selected,
            "query": candidate["query"], "answer": candidate["answer"],
            "source_ids": json.dumps([doc.get("id") for doc in sources]),
            "observed_sources": json.dumps(sources, ensure_ascii=False, sort_keys=True, default=str),
            **{name: int(score) for name, score in scores.items()},
            "reviewer": reviewer, "notes": notes, "reviewed_utc": datetime.now(timezone.utc).isoformat(),
            **{name: candidate[name] for name in ("experiment_id", "locale", "sample_id", "branch")},
        }
        st.session_state.answer_quality_reviews = [*st.session_state.get("answer_quality_reviews", []), record]
        st.success("Review saved in this session.")


def render_research_tab() -> None:
    st.header("Research Lab")
    with st.sidebar:
        health_panel = st.container()
    # Only the chosen journey executes: opening Batch never loads a single-sample dataset.
    mode = navigation_choice("Research workflow", ("Single Sample", "Batch", "Review"), "research_mode")
    try:
        if mode == "Single Sample":
            _render_single_sample_analysis()
        elif mode == "Batch":
            render_batch_evaluation_tab()
        else:
            _render_answer_quality_review()
    except Exception as exc:
        _safe_error("This view could not be displayed. Check the configuration or choose another analysis. Saved results remain in this session.", exc)
    with health_panel:
        _render_health()
