"""User-visible Streamlit regressions; all services/data are deterministic offline seams."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import src.tabs.live_demo as live
import src.tabs.research_lab as research
import src.evaluation_engine as evaluation

APP = Path(__file__).resolve().parents[1] / "main.py"


def test_research_marker_tracks_sumopod_routes_without_credentials(monkeypatch):
    from src.config import Settings
    monkeypatch.setattr(research.st,'session_state',{})
    cfg=Settings(_env_file=None,api_key='private-marker-secret',asr_model='fixture-transcription-model',chat_model='same-chat-model')
    monkeypatch.setattr(research,'get_settings',lambda:cfg)
    marker=research._model_configuration()
    for purpose in ('embedding','asr','chat'):
        assert marker[f'{purpose}_base_url']=='https://ai.sumopod.com/v1'
    assert marker['embedding_send_dimensions'] is False
    assert 'private-marker-secret' not in json.dumps(marker)


@pytest.fixture
def ui(monkeypatch):
    # Explicit fake configuration does not assert availability of any real ASR model.
    monkeypatch.setenv('API_KEY','offline-ui-fixture-key')
    monkeypatch.setenv('ASR_MODEL','fixture-transcription-model')
    live.get_settings.cache_clear()
    state = SimpleNamespace(calls=[], loaders=[], exports={}, docs=[{"id": "policy-1", "text": "Informasi saldo rekening.", "score": 0.91, "intent_ids": [4], "source": "bank_policy.txt", "kb_version": "fixture-v1"}], health={"status": "ready", "points_count": 14, "labelled": True, "kb_version": "fixture-v1", "kb_fingerprint": "fixture-snapshot-v1"}, fail=None, empty_asr=False, batch=None, batch_error=False)
    def health():
        state.calls.append(("health", {}))
        return dict(state.health)
    def embedding(text):
        state.calls.append(("embedding", {"text": text}))
        if state.fail == "embedding": raise RuntimeError("secret-api-token-must-not-appear")
        if not text.strip(): raise ValueError("Empty input")
        return [1.0, 0.5, 0.25]
    def documents(**kwargs):
        state.calls.append(("retrieval", kwargs))
        if state.fail == "retrieval": raise RuntimeError("secret-api-token-must-not-appear")
        threshold = kwargs.get("score_threshold")
        return [doc for doc in state.docs if threshold is None or doc["score"] >= threshold]
    def answer(**kwargs):
        state.calls.append(("llm", kwargs))
        if state.fail == "llm": raise RuntimeError("secret-api-token-must-not-appear")
        return "The policy explains this [1]."
    def asr(buffer):
        state.calls.append(("asr", {}))
        if state.fail == "asr": raise RuntimeError("secret-api-token-must-not-appear")
        if state.empty_asr: return " "
        return "Wie ist mein Kontostand?" if "de-DE" in getattr(buffer, "name", "") else "What is my balance?"
    class Loader:
        def __init__(self, language="en-US", **kwargs):
            state.loaders.append(language)
            if state.fail == "loader": raise RuntimeError("secret-api-token-must-not-appear")
            self.language = language
        def __len__(self): return 3
        def get_sample(self, i, **kwargs):
            if state.fail == "sample": raise RuntimeError("secret-api-token-must-not-appear")
            return dict(id=f"{self.language}-{i}", transcription="Wie ist mein Kontostand?" if self.language == "de-DE" else "What is my balance?", english_transcription="What is my balance?", intent_class=4, intent_name="balance", sampling_rate=8000, audio_path=b"RIFFfixture")
        def get_audio_bytes(self, i, **kwargs):
            buffer=BytesIO(b"fixture audio"); buffer.name=self.language+".wav"; return buffer
    def batch(**kwargs):
        state.calls.append(("batch", kwargs))
        if state.batch_error: raise RuntimeError("secret-api-token-must-not-appear")
        frame = state.batch.copy(deep=True)
        if kwargs.get("progress_callback"): kwargs["progress_callback"](len(frame), len(frame), "en-US", "fixture")
        return frame
    for module in (live, research):
        monkeypatch.setattr(module, "get_embedding", embedding)
        monkeypatch.setattr(module, "search_documents", documents)
        monkeypatch.setattr(module, "generate_grounded_response", answer)
        monkeypatch.setattr(module, "transcribe_audio", asr)
    monkeypatch.setattr(live, "collection_health", health)
    monkeypatch.setattr(research, "MINDS14Loader", Loader)
    monkeypatch.setattr(evaluation, "run_batch_evaluation", batch)
    original_download = research.st.download_button
    def capture_download(*args, **kwargs):
        state.exports[kwargs.get("file_name")] = kwargs.get("data", args[1] if len(args)>1 else None)
        return original_download(*args, **kwargs)
    monkeypatch.setattr(research.st, "download_button", capture_download)
    return state


def start():
    at=AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    return at


def open_research(at, mode="Single Sample"):
    at.button_group(key="mode_selector").set_value(["Research Lab"]).run()
    at.button_group(key="research_mode").set_value([mode]).run()
    assert not at.exception
    return at


def load_single(at):
    at.button(key="load_single_dataset").click().run()
    assert not at.exception
    return at


def trials(statuses):
    rows=[]
    for i,status in enumerate(statuses):
        measured=status!="failed"
        rows.append(dict(locale="en-US", sample_id=f"fixture-{i}", intent_name="balance", status=status, stage="complete" if measured else "asr", error_type="" if measured else "MockError", wer_score=0.0 if measured else None, wer_raw_score=0.0 if measured else None, hit_rate_text=1 if measured else None, hit_rate_voice=1 if measured else None, text_relevance="measured" if measured else "error", voice_relevance="measured" if measured else "error", reference_words=5, substitutions=0, insertions=0, deletions=0))
    frame=pd.DataFrame(rows,columns=["locale","sample_id","intent_name","status","stage","error_type","wer_score","wer_raw_score","hit_rate_text","hit_rate_voice","text_relevance","voice_relevance","reference_words","substitutions","insertions","deletions"])
    frame.attrs.update(manifest={"run_id":"fixture","seed":42},checkpoint_path="results/fixture/trials.jsonl")
    return frame


def test_initial_and_batch_views_do_not_connect_or_load_datasets(ui):
    at=start()
    assert ui.calls==[] and ui.loaders==[]
    open_research(at,"Batch")
    assert ui.calls==[] and ui.loaders==[]
    at.multiselect(key="batch_locales").set_value([]).run()
    assert at.button(key="run_batch").disabled


def test_missing_asr_model_disables_audio_and_runs_without_blocking_text(ui,monkeypatch):
    monkeypatch.delenv('ASR_MODEL',raising=False)
    live.get_settings.cache_clear()
    at=start()
    assert not at.get('audio_input')
    assert any('Set ASR_MODEL' in item.value for item in at.caption)
    assert ui.calls==[]
    at.chat_input(key='chat_text_input').set_value('How do I check my balance?').run()
    assert not at.exception and len(at.chat_message)==2
    assert any(kind=='llm' for kind,_ in ui.calls)
    assert not any(kind=='asr' for kind,_ in ui.calls)
    ui.calls.clear()
    load_single(open_research(at))
    assert at.button(key='run_single_experiment').disabled
    assert any('Set ASR_MODEL' in item.value for item in at.caption)
    at.button_group(key='research_mode').set_value(['Batch']).run()
    assert at.button(key='run_batch').disabled
    assert ui.calls==[]


def test_unconfigured_asr_never_processes_retained_audio_or_voice_pipeline(ui,monkeypatch):
    monkeypatch.delenv('ASR_MODEL',raising=False)
    live.get_settings.cache_clear()
    audio=BytesIO(b'retained fixture audio'); audio.name='recording.wav'
    # Defend against a retained widget value as well as a disabled native widget.
    monkeypatch.setattr(live,'_render_audio_input',lambda:audio)
    at=start(); at.run()
    assert at.session_state['chat_history']==[] and ui.calls==[]
    with pytest.raises(ValueError,match='ASR_MODEL'):
        research._run_voice_pipeline(audio)
    assert ui.calls==[]


def test_missing_sumopod_key_disables_voice_before_any_request(ui,monkeypatch):
    monkeypatch.delenv('API_KEY',raising=False)
    live.get_settings.cache_clear()
    at=start()
    assert not at.get('audio_input')
    assert any('API_KEY' in item.value for item in at.caption)
    assert ui.calls==[]


@pytest.mark.parametrize("status", ["error", "missing", "empty", "incompatible"])
def test_database_status_persists_and_refresh_recovers(ui,status):
    ui.health={"status":status,"points_count":0}
    at=start(); at.button(key="refresh_database").click().run()
    assert at.error
    at.run(); assert at.error and len(ui.calls)==1
    ui.health={"status":"ready","points_count":14,"labelled":True}
    at.button(key="refresh_database").click().run()
    assert not at.error and any("Knowledge base ready" in e.value for e in at.success)


def test_chat_history_followup_and_sources(ui):
    at=start(); at.chat_input(key="chat_text_input").set_value("How do I freeze my card?").run()
    at.chat_input(key="chat_text_input").set_value("How do I undo that?").run()
    assert not at.exception and len(at.chat_message)==4
    calls=[kwargs for kind,kwargs in ui.calls if kind=="llm"]
    assert calls[-1]["history"][0]["content"]=="How do I freeze my card?"
    assert any("Previous question: How do I freeze my card?" in kwargs["text"] for kind,kwargs in ui.calls if kind=="embedding")
    assert at.session_state["chat_history"][-1]["sources"][0]["id"]=="policy-1"


def test_database_status_explains_embedding_provider_migration(ui):
    ui.health = {"status": "incompatible", "reason": "embedding_provider", "points_count": 12}
    at = start()
    at.button(key="refresh_database").click().run()
    assert not at.exception
    assert any("Re-ingest" in error.value and "embedding provider" in error.value for error in at.error)


def test_submit_updates_health_badge_in_the_same_render(ui):
    at = start()
    assert not any('Knowledge base ready' in e.value for e in at.success)
    at.chat_input(key='chat_text_input').set_value('How do I check my balance?').run()
    assert not at.exception
    assert any('Knowledge base ready' in e.value for e in at.success)
    assert not any('has not been checked' in e.value for e in at.caption)


def test_no_context_declines_without_llm(ui):
    ui.docs=[]
    at=start(); at.chat_input(key="chat_text_input").set_value("What is my balance?").run()
    assert not at.exception and not any(kind=="llm" for kind,_ in ui.calls)
    assert at.session_state["chat_history"][-1]["content"]==live.NO_CONTEXT


def test_configured_score_cutoff_declines_low_similarity_without_llm(ui,monkeypatch):
    monkeypatch.setattr(live.get_settings(), "retrieval_score_threshold", 0.99)
    at=start(); at.chat_input(key="chat_text_input").set_value("What is my balance?").run()
    assert not at.exception and not any(kind=="llm" for kind,_ in ui.calls)
    assert next(kwargs for kind,kwargs in ui.calls if kind=="retrieval")["score_threshold"]==0.99
    assert at.session_state["chat_history"][-1]["content"]==live.NO_CONTEXT


@pytest.mark.parametrize("failure", ["embedding", "retrieval", "llm"])
def test_live_service_errors_hide_sensitive_details(ui,failure):
    ui.fail=failure
    at=start(); at.chat_input(key="chat_text_input").set_value("How do I check my balance?").run()
    assert not at.exception and len(at.chat_message)==2
    assert "secret-api-token" not in at.session_state["chat_history"][-1]["content"]
    assert "could not be completed" in at.session_state["chat_history"][-1]["content"]


def test_audio_deduplicates_text_is_single_event_and_clear_resets(ui,monkeypatch):
    audio=BytesIO(b"one retained recording"); audio.name="recording.wav"
    monkeypatch.setattr(live,"_render_audio_input",lambda:audio if live.st.session_state.audio_epoch==0 else None)
    at=start(); assert len(at.chat_message)==0
    at.run(); assert len(at.chat_message)==0
    assert [kind for kind,_ in ui.calls]==["asr"]
    assert at.session_state["voice_draft"]["status"]=="ready"
    at.button(key="send_voice").click().run()
    assert len(at.chat_message)==2
    at.chat_input(key="chat_text_input").set_value("Another question").run()
    assert len(at.chat_message)==4 and sum(kind=="asr" for kind,_ in ui.calls)==1
    at.button(key="clear_chat").click().run()
    assert not at.exception and at.session_state["chat_history"]==[] and len(at.chat_message)==0


@pytest.mark.parametrize("failure", ["asr", "empty"])
def test_audio_errors_are_safe_and_do_not_repeat(ui,monkeypatch,failure):
    audio=BytesIO(b"retained"); audio.name="recording.wav"
    monkeypatch.setattr(live,"_render_audio_input",lambda:audio)
    ui.fail="asr" if failure=="asr" else None; ui.empty_asr=failure=="empty"
    at=start(); assert at.error and not at.exception
    assert "secret-api-token" not in " ".join(e.value for e in at.error)
    at.run(); assert sum(kind=="asr" for kind,_ in ui.calls)==1


def test_single_results_persist_and_label_previous_selection(ui):
    at=open_research(start()); assert ui.loaders==[]
    load_single(at); at.button(key="run_single_experiment").click().run()
    assert not at.exception and any("Word error rate (WER)" in e.value for e in at.markdown)
    at.run(); assert any("Word error rate (WER)" in e.value for e in at.markdown)
    at.slider(key="research_sample_index_en-US").set_value(1).run()
    assert any("previous experiment" in e.value for e in at.info)
    assert any("Word error rate (WER)" in e.value for e in at.markdown)


def test_research_navigation_keeps_selection_without_loading_or_paid_actions(ui):
    at = open_research(start())
    at.selectbox(key="research_language_select").select("de-DE").run()
    load_single(at)
    at.slider(key="research_sample_index_de-DE").set_value(2).run()
    at.button_group(key="research_mode").set_value(["Review"]).run()
    at.button_group(key="mode_selector").set_value(["Assistant"]).run()
    at.button_group(key="mode_selector").set_value(["Research Lab"]).run()

    assert at.button_group(key="research_mode").value == "Review"
    at.button_group(key="research_mode").set_value(["Single Sample"]).run()
    assert not at.exception and not at.warning and ui.calls == [] and ui.loaders == ["de-DE"]
    assert at.selectbox(key="research_language_select").value == "de-DE"
    assert at.slider(key="research_sample_index_de-DE").value == 2


def test_refreshed_knowledge_version_marks_saved_experiment_as_previous(ui):
    at=load_single(open_research(start())); at.button(key="run_single_experiment").click().run()
    ui.health["kb_fingerprint"]="fixture-snapshot-v2"
    at.button(key="refresh_database").click().run()
    assert not at.exception and any("previous experiment" in e.value for e in at.info)
    assert at.session_state["last_experiment"]["marker"]["kb_version"]=="fixture-v1"
    assert at.session_state["last_experiment"]["marker"]["kb_fingerprint"]=="fixture-snapshot-v1"


def test_native_language_used_for_both_single_retrievals(ui):
    at=open_research(start()); at.selectbox(key="research_language_select").select("de-DE").run()
    load_single(at); at.button(key="run_single_experiment").click().run()
    assert not at.exception
    inputs=[kwargs["text"] for kind,kwargs in ui.calls if kind=="embedding"]
    assert inputs==["Wie ist mein Kontostand?","Wie ist mein Kontostand?"]
    assert at.session_state["last_experiment"]["wer"]==0


def test_single_pipelines_share_frozen_physical_collection(ui):
    ui.health['physical_collection'] = 'immutable-policy-v1'
    at = load_single(open_research(start()))
    at.button(key='run_single_experiment').click().run()
    assert not at.exception
    calls = [kwargs for kind, kwargs in ui.calls if kind == 'retrieval']
    assert len(calls) == 2
    assert all(call['collection_name'] == 'immutable-policy-v1' for call in calls)


@pytest.mark.parametrize("failure", ["embedding", "asr", "llm", "sample"])
def test_single_stage_errors_are_contained_and_keep_previous(ui,failure):
    at=load_single(open_research(start())); at.button(key="run_single_experiment").click().run()
    previous=at.session_state["last_experiment"]["created_utc"]
    ui.fail=failure
    if failure=="sample": at.run()
    else: at.button(key="run_single_experiment").click().run()
    assert not at.exception
    if failure == "sample":
        assert at.error
    else:
        assert any("Latest attempt failed" in e.value for e in at.warning)
    assert "secret-api-token" not in " ".join(e.value for e in [*at.error, *at.warning])
    assert at.session_state["last_experiment"]["created_utc"]==previous


def _paired_summary(at):
    return next(element.value for element in at.dataframe if "Paired trials" in element.value.columns)


def test_partial_batch_exposes_failures_denominators_and_exports(ui, settle_batch):
    ui.batch=trials(["success","failed"])
    at=open_research(start(),"Batch"); at.button(key="run_batch").click().run()
    settle_batch(at)
    assert not at.exception and at.warning
    assert _paired_summary(at).iloc[0]["Paired trials"] == 1
    diagnostic = next(e.value for e in at.dataframe if "text_coverage_%" in e.value.columns).iloc[0]
    assert diagnostic["failed_samples"] == 1 and diagnostic["text_coverage_%"] == 50.0
    csv = next(value for name, value in ui.exports.items() if name.endswith("trials.csv"))
    manifest = next(value for name, value in ui.exports.items() if name.endswith("manifest.json"))
    assert len(csv.splitlines()) == 3
    assert json.loads(manifest)["run_id"] == "fixture"


def test_unlabeled_relevance_shows_unknown_instead_of_accuracy(ui, settle_batch):
    ui.batch=trials(["partial"])
    for name in ("hit_rate_text","hit_rate_voice"):
        ui.batch[name]=None
    ui.batch["text_relevance"]="unlabeled"; ui.batch["voice_relevance"]="unlabeled"
    at=open_research(start(),"Batch"); at.button(key="run_batch").click().run()
    settle_batch(at)
    row = _paired_summary(at).iloc[0]
    assert pd.isna(row["Text baseline (%)"]) and pd.isna(row["Voice experiment (%)"])
    assert any("No paired retrieval" in e.value for e in at.info)


def test_voice_improvement_displays_from_matching_denominators(ui, settle_batch):
    ui.batch = trials(["partial", "partial", "success"])
    ui.batch["hit_rate_text"] = [1, 1, 0]
    ui.batch["hit_rate_voice"] = [None, None, 1]
    at=open_research(start(),"Batch"); at.button(key="run_batch").click().run()
    settle_batch(at)
    row = _paired_summary(at).iloc[0]
    assert row["Paired trials"] == 1 and row["Degradation (pp)"] == -100
    assert row["Text baseline (%)"] == 0 and row["Voice experiment (%)"] == 100


def test_unpaired_retrieval_results_do_not_imply_degradation(ui, settle_batch):
    ui.batch=trials(["partial","partial"])
    ui.batch["hit_rate_text"]=[1,None]; ui.batch["hit_rate_voice"]=[None,0]
    at=open_research(start(),"Batch"); at.button(key="run_batch").click().run()
    settle_batch(at)
    row = _paired_summary(at).iloc[0]
    assert row["Paired trials"] == 0 and pd.isna(row["Degradation (pp)"])


@pytest.mark.parametrize("failed_statuses", [["failed","failed"], []])
def test_failed_or_empty_batch_never_overwrites_valid_results(ui,failed_statuses,settle_batch):
    ui.batch=trials(["success"])
    at=open_research(start(),"Batch"); at.button(key="run_batch").click().run()
    settle_batch(at)
    ui.batch=trials(failed_statuses)
    at.button(key="batch_replicate").click().run()
    settle_batch(at)
    assert len(at.session_state["batch_results"])==1
    assert any("Previous valid results" in e.value for e in at.info)
    at.run(); assert not at.exception


def test_resume_seed_and_config_contract(ui, tmp_path, settle_batch):
    ui.batch=trials(["success"])
    path = tmp_path / "trials.jsonl"
    path.write_text("", encoding="utf-8")
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "run_id": "saved-run", "config_hash": "fixture", "completed": False,
        "config": {"locales": ["en-US"], "samples_per_locale": 3, "seed": 17, "top_k": 5},
    }), encoding="utf-8")
    at=open_research(start(),"Batch")
    at.checkbox(key="batch_resume").check().run()
    assert at.button(key="run_batch").disabled
    at.text_input(key="batch_checkpoint").set_value(str(path)).run()
    at.button(key="batch_inspect").click().run()
    assert ui.calls == []
    at.button(key="batch_continue").click().run()
    settle_batch(at)
    kwargs=next(kwargs for kind,kwargs in ui.calls if kind=="batch")
    assert kwargs["resume"] and kwargs["seed"]==17 and Path(kwargs["checkpoint_path"]) == path
    assert kwargs["run_config"]["top_k"] == 5 and "retrieval_top_k" not in kwargs["run_config"]
    assert "kb_fingerprint" not in kwargs["run_config"]


def test_manual_rubric_records_observed_sources_and_exports(ui):
    at=load_single(open_research(start())); at.button(key="run_single_experiment").click().run()
    at.button_group(key="research_mode").set_value(["Review"]).run()
    for label in ("Support from displayed sources","Factual correctness","Relevance to the question"):
        next(e for e in at.selectbox if e.label==label).select("2")
    next(e for e in at.text_area if e.label=="Evidence and notes").set_value("=unsafe spreadsheet formula")
    at.button(key="save_quality_review").click().run()
    assert not at.exception
    record=at.session_state["answer_quality_reviews"][0]
    assert record["support"]==2 and json.loads(record["source_ids"])==["policy-1"]
    assert json.loads(record["observed_sources"])[0]["kb_version"]=="fixture-v1"
    assert record["experiment_id"] == at.session_state["last_experiment"]["id"]
    assert (record["locale"], record["sample_id"], record["branch"]) == ("en-US", "en-US-0", "text")
    assert "'=unsafe spreadsheet formula" in ui.exports["manual_answer_reviews.csv"]


def test_manual_review_drafts_are_restored_only_for_the_same_answer(ui):
    at = load_single(open_research(start()))
    at.button(key="run_single_experiment").click().run()
    at.button_group(key="research_mode").set_value(["Review"]).run()
    support = lambda: next(e for e in at.selectbox if e.label == "Support from displayed sources")
    notes = lambda: next(e for e in at.text_area if e.label == "Evidence and notes")
    support().select("2")
    notes().set_value("Reference transcript review")
    at.selectbox(key="quality_candidate").select("Last experiment · voice").run()
    assert support().value == "Not reviewed" and notes().value == ""
    at.selectbox(key="quality_candidate").select("Last experiment · text").run()
    assert not at.warning and support().value == "2" and notes().value == "Reference transcript review"

    at.button_group(key="research_mode").set_value(["Single Sample"]).run()
    at.button(key="run_single_experiment").click().run()
    at.button_group(key="research_mode").set_value(["Review"]).run()
    assert support().value == "Not reviewed" and notes().value == ""
