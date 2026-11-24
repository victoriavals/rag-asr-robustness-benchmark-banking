"""Offline batch-worker, recovery and paired-display regressions."""
import json
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.tabs import batch_evaluation as batch


def request(seed=17):
    return batch.BatchRequest.create(["en-US"], 2, seed, {
        "asr_model": "asr-fixture", "embedding_model": "embedding-fixture",
        "embedding_dimensions": 2, "collection_name": "policy", "retrieval_top_k": 3,
    })


def health():
    return {"status": "ready", "points_count": 12, "kb_version": "kb1", "labelled": True}


def frame(statuses=("success", "success")):
    return pd.DataFrame([dict(
        trial_id=f"en-US:{i}", locale="en-US", sample_id=f"sample-{i}", intent_name="balance",
        status=status, stage="complete", error_type="", wer_score=0., cer_score=0., wer_raw_score=0.,
        hit_rate_text=1, hit_rate_voice=1, reference_words=3, substitutions=0, insertions=0,
        deletions=0, text_relevance="measured", voice_relevance="measured", config_hash="hash1",
    ) for i, status in enumerate(statuses)])


def write_checkpoint(kwargs, rows, *, completed=False):
    path = Path(kwargs["checkpoint_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(kwargs["run_config"], locales=kwargs["locales"],
                  samples_per_locale=kwargs["samples_per_locale"], seed=kwargs["seed"],
                  selected={"en-US": [{"index": 0}, {"index": 1}]})
    manifest = dict(run_id="engine-manifest-id", config=config, config_hash="hash1",
                    completed=completed, created_at="2026-11-24T00:00:00+00:00")
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def finish(kwargs):
    data = frame()
    manifest = write_checkpoint(kwargs, data.to_dict("records"), completed=True)
    data.attrs.update(manifest=manifest, checkpoint_path=kwargs["checkpoint_path"])
    return data


def start(controller, path, runner, **options):
    return controller.start(request(), path, runner=runner, health_checker=health,
                            preflight=lambda: "asr-fixture", **options)


def test_worker_is_background_and_snapshot_is_immutable(tmp_path):
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    ids = []
    def runner(**kwargs):
        ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return finish(kwargs)
    controller = batch.BatchController()
    try:
        assert start(controller, tmp_path / "run.jsonl", runner)
        assert entered.wait(5)
        snapshot = controller.snapshot()
        assert snapshot.busy and ids != [caller]
        with pytest.raises(FrozenInstanceError):
            snapshot.phase = "changed"
        assert not start(controller, tmp_path / "duplicate.jsonl", runner)
    finally:
        release.set(); controller.join(5)
    assert controller.snapshot().phase == "completed"
    copy = controller.snapshot().dataframe()
    copy.loc[0, "status"] = "changed"
    assert controller.snapshot().dataframe().loc[0, "status"] == "success"


def test_identical_run_is_guarded_but_replication_uses_new_path(tmp_path):
    calls = []
    def runner(**kwargs):
        calls.append(kwargs); return finish(kwargs)
    controller = batch.BatchController()
    assert start(controller, tmp_path / "first.jsonl", runner)
    controller.join(5)
    first_id = controller.snapshot().job_id
    assert not start(controller, tmp_path / "unwanted.jsonl", runner)
    assert len(calls) == 1
    assert start(controller, tmp_path / "replicate.jsonl", runner, replication=True)
    controller.join(5)
    assert len(calls) == 2 and controller.snapshot().job_id != first_id
    assert first_id == batch._checkpoint_identity(tmp_path / "first.jsonl")


def test_stop_recovers_checkpoint_then_resume_uses_frozen_setup(tmp_path):
    committed, release = threading.Event(), threading.Event()
    calls = []
    path = tmp_path / "run.jsonl"
    def runner(**kwargs):
        calls.append(kwargs)
        if not kwargs["resume"]:
            write_checkpoint(kwargs, frame().iloc[:1].to_dict("records"))
            committed.set()
            assert release.wait(5)
            kwargs["progress_callback"](1, 2, "en-US", "sample-0")
            pytest.fail("Stop must interrupt after its recorded trial")
        return finish(kwargs)
    controller = batch.BatchController()
    try:
        assert start(controller, path, runner)
        assert committed.wait(5)
        stable_id = controller.snapshot().job_id
        controller.request_stop()
    finally:
        release.set(); controller.join(5)
    saved = controller.snapshot()
    assert saved.phase == "interrupted" and saved.current == 1 and saved.total == 2
    assert len(saved.dataframe()) == 1 and saved.job_id == stable_id
    assert saved.request.seed == 17
    assert controller.start(saved.request, saved.checkpoint_path, runner=runner,
                            health_checker=health, preflight=lambda: None, resume=True)
    controller.join(5)
    assert controller.snapshot().job_id == stable_id
    assert calls[-1]["seed"] == 17 and calls[-1]["checkpoint_path"] == str(path.resolve())
    assert calls[-1]["resume"] is True


def test_stop_after_completed_cannot_overwrite_terminal_state(tmp_path):
    controller = batch.BatchController()
    start(controller, tmp_path / "run.jsonl", lambda **kwargs: finish(kwargs))
    controller.join(5)
    controller.request_stop()
    assert controller.snapshot().phase == "completed"


def test_selection_callback_records_actual_total_before_trials(tmp_path):
    entered, release = threading.Event(), threading.Event()
    controller = batch.BatchController()
    def runner(**kwargs):
        kwargs["selection_callback"](2)
        entered.set()
        assert release.wait(5)
        return finish(kwargs)
    try:
        start(controller, tmp_path / "run.jsonl", runner)
        assert entered.wait(5)
        assert controller.snapshot().total == 2 and controller.snapshot().current == 0
    finally:
        release.set(); controller.join(5)


def test_preflight_failure_makes_no_health_or_engine_call(tmp_path):
    called = []
    def fail(): raise ValueError("sensitive-provider-secret")
    controller = batch.BatchController()
    controller.start(request(), tmp_path / "run.jsonl", runner=lambda **k: called.append("engine"),
                     health_checker=lambda: called.append("health"), preflight=fail)
    controller.join(5)
    snapshot = controller.snapshot()
    assert snapshot.phase == "blocked" and not called
    assert "sensitive-provider-secret" not in snapshot.message


def test_inspection_is_read_only_and_loads_legacy_explicit_path(tmp_path):
    path = tmp_path / "older-layout" / "trials.jsonl"
    kwargs = dict(checkpoint_path=str(path), run_config=json.loads(request().config_json),
                  locales=["en-US"], samples_per_locale=2, seed=17)
    write_checkpoint(kwargs, frame().iloc[:1].to_dict("records"))
    with path.open("ab") as handle: handle.write(b'{"trial_id":')
    before = path.read_bytes()
    snapshot = batch.inspect_checkpoint(path)
    assert snapshot.phase == "interrupted" and snapshot.current == 1
    assert snapshot.request.seed == 17
    assert path.read_bytes() == before
    assert "Incomplete final record" in snapshot.message


def test_corrupt_complete_checkpoint_error_hides_file_content(tmp_path):
    path = tmp_path / "trials.jsonl"
    kwargs = dict(checkpoint_path=str(path), run_config=json.loads(request().config_json),
                  locales=["en-US"], samples_per_locale=2, seed=17)
    write_checkpoint(kwargs, [])
    path.write_text("sensitive-file-content\n")
    with pytest.raises(ValueError) as error: batch.inspect_checkpoint(path)
    assert "sensitive-file-content" not in str(error.value)


def test_manifest_restore_preserves_logical_alias_for_resume(tmp_path):
    path = tmp_path / "trials.jsonl"
    config = dict(json.loads(request().config_json), collection_name="immutable-physical-v1",
                  logical_collection="bank-policy")
    kwargs = dict(checkpoint_path=str(path), run_config=config,
                  locales=["en-US"], samples_per_locale=2, seed=17)
    write_checkpoint(kwargs, frame().iloc[:1].to_dict("records"))
    snapshot = batch.inspect_checkpoint(path)
    assert json.loads(snapshot.request.config_json)["collection_name"] == "bank-policy"
    captured = []
    controller = batch.BatchController()
    controller.load(snapshot)
    def runner(**values):
        captured.append(values["run_config"]["collection_name"])
        return finish(values)
    assert controller.start(snapshot.request, path, runner=runner, health_checker=health,
                            preflight=lambda: None, resume=True)
    controller.join(5)
    assert captured == ["bank-policy"]


def test_paired_display_uses_common_denominator_and_keeps_diagnostics():
    from src.evaluation_engine import summarize_results
    data = frame(("partial", "partial", "success"))
    data["hit_rate_text"] = [1, 1, 0]
    data["hit_rate_voice"] = [None, None, 1]
    summary = summarize_results(data)
    primary = batch._paired_comparison(summary).iloc[0]
    assert primary["Paired trials"] == 1
    assert primary["Text baseline (%)"] == 0 and primary["Voice experiment (%)"] == 100
    assert primary["Degradation (pp)"] == -100
    assert summary.iloc[0]["text_hit_rate_%"] == pytest.approx(200 / 3)


def test_passive_batch_view_does_not_capture_provider_or_runner(monkeypatch):
    monkeypatch.setenv("API_KEY", "offline-test-key")
    monkeypatch.setenv("ASR_MODEL", "asr-fixture")
    monkeypatch.setattr(batch, "_capture_callables", lambda: pytest.fail("Passive rendering captured a provider"))
    app = AppTest.from_string("from src.tabs.batch_evaluation import render_batch_evaluation_tab\nrender_batch_evaluation_tab()")
    app.run(timeout=30)
    assert not app.exception
    assert app.button(key="run_batch")
    assert app.session_state["_batch_controller"].snapshot().phase == "idle"
    assert any(item.label == "Advanced settings and recovery" for item in app.expander)


def test_ui_background_stop_frozen_resume_and_explicit_replication(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "offline-batch-ui-key")
    monkeypatch.setenv("ASR_MODEL", "asr-fixture")
    committed, release = threading.Event(), threading.Event()
    calls, exports = [], []
    def runner(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            write_checkpoint(kwargs, [])
            kwargs["selection_callback"](2)
            write_checkpoint(kwargs, frame().iloc[:1].to_dict("records"))
            committed.set()
            assert release.wait(10)
            kwargs["progress_callback"](1, 2, "en-US", "sample-0")
        return finish(kwargs)
    monkeypatch.setattr(batch, "_capture_callables", lambda: (runner, health, lambda: None))
    original = batch.st.download_button
    def capture(*args, **kwargs):
        exports.append((args[0], kwargs.get("file_name")))
        return original(*args, **kwargs)
    monkeypatch.setattr(batch.st, "download_button", capture)
    app = AppTest.from_string("from src.tabs.batch_evaluation import render_batch_evaluation_tab\nrender_batch_evaluation_tab()")
    app.run(timeout=30)
    try:
        app.button(key="run_batch").click().run()
        assert committed.wait(5)
        controller = app.session_state["_batch_controller"]
        stable_id = controller.snapshot().job_id
        assert app.button(key="run_batch").disabled
        app.button(key="batch_stop").click().run()
    finally:
        release.set()
        app.session_state["_batch_controller"].join(5)
    app.run()
    assert not app.exception
    assert controller.snapshot().phase == "interrupted"
    assert any(value.value == controller.snapshot().checkpoint_path for value in app.code)
    saved_seed = controller.snapshot().request.seed
    app.number_input(key="batch_seed").set_value(999).run()
    app.button(key="batch_continue").click().run()
    controller.join(5)
    app.run()
    assert calls[-1]["resume"] is True and calls[-1]["seed"] == saved_seed
    assert controller.snapshot().job_id == stable_id
    assert calls[-1]["checkpoint_path"] == calls[0]["checkpoint_path"]
    # Restore the displayed new-run seed, making the duplicate protection visible.
    app.number_input(key="batch_seed").set_value(saved_seed).run()
    assert app.button(key="run_batch").disabled
    before = len(calls)
    app.run()
    assert len(calls) == before
    app.button(key="batch_replicate").click().run()
    controller.join(5)
    app.run()
    assert len(calls) == before + 1 and calls[-1]["checkpoint_path"] != calls[0]["checkpoint_path"]
    assert any("current" in label and stable_id in label for label, _ in exports)
    assert not app.exception


def test_explicit_database_refresh_survives_later_batch_rerender(monkeypatch):
    monkeypatch.setenv("API_KEY", "offline-batch-ui-key")
    monkeypatch.setenv("ASR_MODEL", "asr-fixture")
    monkeypatch.setattr(batch, "_capture_callables", lambda: (lambda **kwargs: finish(kwargs), health, lambda: None))
    app = AppTest.from_string("from src.tabs.batch_evaluation import render_batch_evaluation_tab\nrender_batch_evaluation_tab()")
    app.run(timeout=30)
    app.button(key="run_batch").click().run()
    controller = app.session_state["_batch_controller"]
    controller.join(5)
    app.run()
    app.session_state["database_health"] = {"status": "ready", "kb_version": "explicitly-refreshed"}
    app.run()
    assert app.session_state["database_health"]["kb_version"] == "explicitly-refreshed"
    assert not app.exception
