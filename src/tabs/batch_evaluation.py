"""Session-owned batch jobs, recoverable checkpoints, and paired research results.

Only the main Streamlit thread reads session state or renders widgets. A worker
receives frozen JSON configuration and publishes immutable, lock-protected values.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.tabs.ui_state import (
    _spreadsheet_safe_csv, model_configuration, remember_widget, saved_widget_value,
)

logger = logging.getLogger(__name__)
BUSY_PHASES = frozenset({"preparing", "running", "stopping"})
LOCALES = ("en-US", "en-GB", "en-AU")
ENGINE_FIELDS = frozenset({
    "embedding_model", "embedding_dimensions", "embedding_send_dimensions",
    "asr_model", "chat_model", "embedding_base_url", "asr_base_url",
    "chat_base_url", "collection_name", "dataset_revision", "top_k",
    "score_threshold", "kb_version", "asr_language",
})


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _engine_configuration(configuration: dict) -> dict:
    config = dict(configuration)
    if config.get("logical_collection"):
        config["collection_name"] = config["logical_collection"]
    config["top_k"] = config.pop("retrieval_top_k", config.get("top_k", 3))
    config["score_threshold"] = config.pop(
        "retrieval_score_threshold", config.get("score_threshold")
    )
    return {key: value for key, value in config.items() if key in ENGINE_FIELDS}


def _checkpoint_identity(path) -> str:
    return hashlib.sha256(str(Path(path).expanduser().resolve()).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class BatchRequest:
    locales: tuple[str, ...]
    samples_per_locale: int
    seed: int
    config_json: str

    @classmethod
    def create(cls, locales, samples_per_locale, seed, configuration):
        if not locales or len(locales) != len(set(locales)) or any(x not in LOCALES for x in locales):
            raise ValueError("Choose at least one supported locale without duplicates.")
        if (isinstance(samples_per_locale, bool) or not isinstance(samples_per_locale, int)
                or samples_per_locale != -1 and samples_per_locale < 1):
            raise ValueError("Samples per locale must be positive or full dataset.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Sampling seed must be an integer.")
        return cls(tuple(locales), int(samples_per_locale), seed, _json(_engine_configuration(configuration)))

    @property
    def signature(self) -> str:
        # A signature guards accidental repeats; intentional replications stay possible.
        config = {key: value for key, value in json.loads(self.config_json).items() if key != "kb_version"}
        data = [sorted(self.locales), self.samples_per_locale,
                0 if self.samples_per_locale == -1 else self.seed, config]
        return hashlib.sha256(_json(data).encode()).hexdigest()


@dataclass(frozen=True)
class BatchSnapshot:
    job_id: str = ""
    phase: str = "idle"
    request: BatchRequest | None = None
    checkpoint_path: str = ""
    current: int = 0
    total: int = 0
    locale: str = ""
    sample_id: str = ""
    message: str = ""
    error_type: str = ""
    rows_json: str = "[]"
    columns: tuple[str, ...] = ()
    manifest_json: str = "{}"
    health_json: str = "{}"
    started_utc: str = ""

    @property
    def busy(self) -> bool:
        return self.phase in BUSY_PHASES

    def dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(json.loads(self.rows_json), columns=list(self.columns) or None)
        frame.attrs.update(
            manifest=json.loads(self.manifest_json), checkpoint_path=self.checkpoint_path,
            ui_run_id=self.job_id,
            manifest_path=str(Path(self.checkpoint_path).with_suffix(".manifest.json")) if self.checkpoint_path else "",
        )
        return frame


class BatchStopped(InterruptedError):
    """A cooperative stop after the engine has checkpointed its current trial."""


def inspect_checkpoint(path: str | Path) -> BatchSnapshot:
    """Read only the explicitly selected run; never repair or scan directories."""
    checkpoint = Path(path).expanduser().resolve()
    manifest_path = checkpoint.with_suffix(".manifest.json")
    try:
        if checkpoint.suffix != ".jsonl":
            raise ValueError
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        request = BatchRequest.create(config["locales"], config["samples_per_locale"], config["seed"], config)
        lines = checkpoint.read_bytes().splitlines(keepends=True) if checkpoint.exists() else []
        rows, seen = [], set()
        partial = False
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if i == len(lines) - 1 and not line.endswith(b"\n"):
                    partial = True
                    break
                raise ValueError from None
            if (not isinstance(row, dict) or not isinstance(row.get("trial_id"), str) or not row["trial_id"]
                    or row.get("config_hash") != manifest.get("config_hash")
                    or row.get("trial_id") in seen or row.get("status") not in {"success", "partial", "failed"}):
                raise ValueError
            required = {"locale", "sample_id", "intent_name", "wer_score", "hit_rate_text", "hit_rate_voice"}
            if not required.issubset(row):
                raise ValueError
            for name in ("hit_rate_text", "hit_rate_voice"):
                if row[name] is not None and row[name] not in (0, 1):
                    raise ValueError
            if row["wer_score"] is not None and (
                not isinstance(row["wer_score"], (int, float)) or
                not math.isfinite(row["wer_score"]) or row["wer_score"] < 0
            ):
                raise ValueError
            seen.add(row.get("trial_id"))
            rows.append(row)
        selected = config.get("selected", {})
        total = sum(len(value) for value in selected.values()) if isinstance(selected, dict) else 0
        total = max(total, len(rows))
        completed = bool(manifest.get("completed")) and not partial and len(rows) >= total
        frame = pd.DataFrame(rows)
        return BatchSnapshot(
            job_id=_checkpoint_identity(checkpoint),
            phase="completed" if completed else "interrupted", request=request,
            checkpoint_path=str(checkpoint), current=len(rows), total=total,
            rows_json=frame.to_json(orient="records"), columns=tuple(frame.columns),
            manifest_json=_json(manifest), started_utc=str(manifest.get("created_at", "")),
            message=("Incomplete final record retained; the engine will validate recovery on resume." if partial else "Saved run loaded locally. No provider requests were made."),
        )
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("This checkpoint could not be inspected. Select its trials.jsonl and matching manifest, or check the saved run files.") from None


class BatchController:
    """One active worker per browser session; no Streamlit dependency in workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = BatchSnapshot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> BatchSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, **changes):
        with self._lock:
            self._snapshot = replace(self._snapshot, **changes)

    def load(self, saved: BatchSnapshot):
        with self._lock:
            if self._snapshot.busy:
                raise RuntimeError("A batch is already running in this session.")
            self._snapshot = saved

    def start(self, request: BatchRequest, checkpoint_path, *, runner: Callable,
              health_checker: Callable, preflight: Callable, resume=False, replication=False) -> bool:
        with self._lock:
            previous = self._snapshot
            if previous.busy:
                return False
            if (not resume and not replication and previous.request
                    and previous.request.signature == request.signature
                    and previous.phase in {"completed", "interrupted"}):
                return False
            path = str(Path(checkpoint_path).expanduser().resolve())
            self._snapshot = BatchSnapshot(
                job_id=_checkpoint_identity(path),
                phase="preparing", request=request, checkpoint_path=path,
                current=previous.current if resume else 0, total=previous.total if resume else 0,
                rows_json=previous.rows_json if resume else "[]",
                columns=previous.columns if resume else (), manifest_json=previous.manifest_json if resume else "{}",
                started_utc=previous.started_utc if resume else datetime.now(timezone.utc).isoformat(),
                message="Checking configuration and knowledge base before model requests.",
            )
            self._stop = threading.Event()
            stop = self._stop
        worker = threading.Thread(
            target=self._work, args=(request, runner, health_checker, preflight, bool(resume), stop),
            name="banking-batch-worker", daemon=True,
        )
        self._thread = worker
        worker.start()
        return True

    def request_stop(self):
        with self._lock:
            if self._snapshot.busy:
                self._stop.set()
                self._snapshot = replace(self._snapshot, phase="stopping", message="Stopping after the current trial is checkpointed.")

    def join(self, timeout=None):
        worker = self._thread
        if worker is not None:
            worker.join(timeout)

    def _work(self, request, runner, health_checker, preflight, resume, stop):
        try:
            preflight()
            if stop.is_set():
                raise BatchStopped
            health = health_checker()
            self._publish(health_json=_json(health))
            if stop.is_set():
                raise BatchStopped
            if health.get("status") != "ready" or not health.get("points_count", 0):
                self._publish(phase="blocked", message="The knowledge base is not ready. Check its database status before retrying.")
                return
            config = json.loads(request.config_json)
            if not resume and health.get("kb_version"):
                config["kb_version"] = health["kb_version"]
            frozen = replace(request, config_json=_json(config))
            self._publish(request=frozen, phase="running", message="Trials are running; recorded attempts will not be repeated by resume.")

            def progress(current, total, locale, sample_id):
                self._publish(current=int(current), total=int(total), locale=str(locale), sample_id=str(sample_id))
                if stop.is_set():
                    raise BatchStopped

            def selection(total):
                self._publish(total=int(total), message=f"Prepared {int(total)} trials. Each nonempty trial uses up to two embeddings and one ASR request.")
                if stop.is_set():
                    raise BatchStopped

            frame = runner(
                locales=list(frozen.locales), samples_per_locale=frozen.samples_per_locale,
                seed=frozen.seed, checkpoint_path=self.snapshot().checkpoint_path,
                resume=resume, run_config=json.loads(frozen.config_json), progress_callback=progress,
                selection_callback=selection,
            )
            manifest = frame.attrs.get("manifest", {})
            self._publish(
                phase="completed",
                rows_json=frame.to_json(orient="records"), columns=tuple(frame.columns),
                manifest_json=_json(manifest), current=len(frame), total=max(len(frame), self.snapshot().total),
                message="All planned attempts finished. Inspect failures and coverage before interpreting rates.",
            )
        except Exception as exc:
            logger.warning("Batch worker stopped (%s)", type(exc).__name__)
            snapshot = self.snapshot()
            try:
                recovered = inspect_checkpoint(snapshot.checkpoint_path)
            except ValueError:
                recovered = None
            if recovered is not None:
                self._publish(
                    job_id=recovered.job_id, request=recovered.request or snapshot.request,
                    phase="interrupted", rows_json=recovered.rows_json, columns=recovered.columns,
                    manifest_json=recovered.manifest_json, current=recovered.current, total=recovered.total,
                    error_type=type(exc).__name__,
                    message="Run interrupted. Recorded trials and its checkpoint are retained; resume uses the same frozen setup.",
                )
            else:
                self._publish(
                    phase="blocked" if snapshot.phase == "preparing" or isinstance(exc, BatchStopped) else "failed",
                    error_type=type(exc).__name__,
                    message="The run could not start or finish. Check the saved configuration and service readiness. No earlier result was replaced.",
                )


def _controller() -> BatchController:
    if "_batch_controller" not in st.session_state:
        st.session_state._batch_controller = BatchController()
    return st.session_state._batch_controller


def _attempt_counts(frame):
    status = frame.get("status", pd.Series(["success"] * len(frame)))
    return {"Attempted": len(frame), "Successful": int(status.eq("success").sum()),
            "Partial": int(status.eq("partial").sum()), "Failed": int(status.eq("failed").sum())}


def _has_usable_results(frame):
    return bool(not frame.empty and (
        ("wer_score" in frame and frame.wer_score.notna().any()) or
        (all(key in frame for key in ("hit_rate_text", "hit_rate_voice")) and
         (frame.hit_rate_text.notna() & frame.hit_rate_voice.notna()).any())
    ))


def _percent(value):
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f}%"


def _display_attempt_counts(frame):
    st.caption(" · ".join(f"{value} {name.lower()}" for name, value in _attempt_counts(frame).items()))
    if "status" in frame and not frame.status.eq("success").all():
        st.warning("Partial and failed trials are retained. Unknown metrics are excluded from their measured denominators.")
        with st.expander("Incomplete trials"):
            names = [name for name in ("locale", "sample_id", "status", "stage", "error_type") if name in frame]
            st.dataframe(frame.loc[~frame.status.eq("success"), names], width="stretch", hide_index=True)


def _paired_comparison(summary):
    names = {"locale": "Locale", "paired_samples": "Paired trials", "paired_text_hit_rate_%": "Text baseline (%)",
             "paired_voice_hit_rate_%": "Voice experiment (%)", "paired_degradation_pp": "Degradation (pp)"}
    return summary.loc[:, list(names)].rename(columns=names)


def _display_batch_results(frame):
    from src.evaluation_engine import summarize_by_intent, summarize_results
    _display_attempt_counts(frame)
    if frame.empty:
        st.info("No trials have been recorded yet.")
        return
    summary = summarize_results(frame)
    st.subheader("Paired results by locale")
    st.caption("Chapter-level intent labels are a retrieval proxy. They do not establish question-level relevance or answer correctness.")
    for _, row in summary.iterrows():
        delta = row["paired_degradation_pp"]
        delta_label = "N/A" if pd.isna(delta) else f"{delta:+.1f} pp"
        st.markdown(f"**{row['locale']} · paired n={int(row['paired_samples'])}** — Text {_percent(row['paired_text_hit_rate_%'])} · Voice {_percent(row['paired_voice_hit_rate_%'])} · Δ {delta_label}")
        low, high = row["paired_degradation_ci_low_pp"], row["paired_degradation_ci_high_pp"]
        if pd.notna(low) and pd.notna(high):
            st.caption(f"95% paired bootstrap interval: {low:+.1f} to {high:+.1f} pp; independent utterances are assumed.")
        st.caption(f"WER {_percent(row['avg_wer_%'])} · CER {_percent(row.get('avg_cer_%'))} · transcription n={int(row['wer_samples'])}. Retrieval coverage: text {_percent(row['text_coverage_%'])}, voice {_percent(row['voice_coverage_%'])}.")
    st.caption("Degradation = paired text minus paired voice. Negative values mean voice scored higher on those same trials.")
    st.dataframe(_paired_comparison(summary), width="stretch", hide_index=True)
    paired = summary.loc[summary.paired_samples.gt(0)]
    if not paired.empty:
        import plotly.express as px
        chart = paired.rename(columns={"paired_text_hit_rate_%": "Text baseline", "paired_voice_hit_rate_%": "Voice experiment"})
        st.plotly_chart(px.bar(chart, x="locale", y=["Text baseline", "Voice experiment"], barmode="group",
                              labels={"locale": "Locale", "value": "Paired hit rate (%)", "variable": "Pipeline"}), width="stretch")
    else:
        st.info("No paired retrieval measurements are available. A comparison chart cannot be calculated.")
    with st.expander("Coverage and full summary"):
        st.caption("Independent rates below use each branch's measured denominator and are diagnostic only.")
        st.dataframe(summary, width="stretch", hide_index=True)
    with st.expander("Results by intent"):
        st.dataframe(summarize_by_intent(frame), width="stretch", hide_index=True)
    with st.expander("All recorded trials"):
        st.dataframe(frame, width="stretch", hide_index=True)


def _export_batch(frame, key="batch"):
    manifest = frame.attrs.get("manifest", {})
    identity = str(frame.attrs.get("ui_run_id") or _checkpoint_identity(frame.attrs.get("checkpoint_path", "run.jsonl")))
    identity = "".join(c for c in identity if c.isalnum() or c in "-_")[:80] or "run"
    label = {"current": "current", "previous": "previous valid"}.get(key, key)
    if len(frame.columns):
        st.download_button(f"Download {label} trials CSV ({identity})", _spreadsheet_safe_csv(frame), file_name=f"{identity}-{key}-trials.csv",
                           mime="text/csv", key=f"{key}_{identity}_csv", on_click="ignore")
    st.download_button(f"Download {label} manifest ({identity})", _json(manifest), file_name=f"{identity}-{key}-manifest.json",
                       mime="application/json", key=f"{key}_{identity}_manifest", on_click="ignore")


def _capture_callables():
    # Capture at the explicit submission boundary, so each session binds its job.
    from src import evaluation_engine, sumopod_services
    from src.tabs import live_demo
    return evaluation_engine.run_batch_evaluation, live_demo.collection_health, sumopod_services.require_asr_model


def _start(controller, request, *, resume=False, replication=False):
    snapshot = controller.snapshot()
    path = snapshot.checkpoint_path if resume else get_settings().results_dir / uuid.uuid4().hex / "trials.jsonl"
    runner, health, preflight = _capture_callables()
    return controller.start(request, path, runner=runner, health_checker=health, preflight=preflight,
                            resume=resume, replication=replication)


def _adopt_snapshot(snapshot):
    health_token = (snapshot.job_id, snapshot.health_json)
    if snapshot.health_json != "{}" and st.session_state.get("_batch_health_seen") != health_token:
        st.session_state.database_health = json.loads(snapshot.health_json)
        st.session_state._batch_health_seen = health_token
    token = (snapshot.job_id, snapshot.phase, snapshot.current)
    if snapshot.busy or not snapshot.job_id or st.session_state.get("_batch_adopted") == token:
        return
    st.session_state._batch_adopted = token
    frame = snapshot.dataframe()
    st.session_state.last_batch_attempt = frame
    st.session_state.last_batch_attempt_config = json.loads(snapshot.request.config_json) if snapshot.request else {}
    if snapshot.phase == "completed" and _has_usable_results(frame):
        st.session_state.batch_results = frame
        st.session_state.batch_configuration = st.session_state.last_batch_attempt_config


def _render_status(controller):
    snapshot = controller.snapshot()
    _adopt_snapshot(snapshot)
    if not snapshot.job_id:
        return
    st.subheader(f"Run status: {snapshot.phase}")
    if snapshot.busy and st.button("Stop after current trial", key="batch_stop", disabled=snapshot.phase == "stopping"):
        controller.request_stop()
        st.rerun()
    st.caption(f"Run {snapshot.job_id} · started {snapshot.started_utc}")
    if snapshot.phase in {"blocked", "failed"}:
        st.error(snapshot.message)
    elif snapshot.phase == "interrupted":
        st.warning(snapshot.message)
    else:
        st.info(snapshot.message)
    if snapshot.total:
        st.progress(min(1., snapshot.current / snapshot.total), text=f"{snapshot.current}/{snapshot.total} recorded trials")
    elif snapshot.busy:
        st.caption("Preparing data and checking support; model requests have not been counted as completed trials.")
    if snapshot.locale:
        st.caption(f"Latest checkpoint: {snapshot.locale} · {snapshot.sample_id}")
    st.caption("This run's checkpoint" if snapshot.manifest_json != "{}" else "Reserved checkpoint path")
    st.code(snapshot.checkpoint_path, language=None, wrap_lines=True)
    if snapshot.request:
        label = "all samples" if snapshot.request.samples_per_locale == -1 else f"cap {snapshot.request.samples_per_locale}/locale"
        st.caption(f"Saved setup: {', '.join(snapshot.request.locales)} · {label} · seed {snapshot.request.seed}.")
        with st.expander("Saved run configuration"):
            st.json(json.loads(snapshot.request.config_json))
            manifest_id = json.loads(snapshot.manifest_json).get("run_id")
            if manifest_id:
                st.caption(f"Engine manifest ID: {manifest_id}")
    if snapshot.phase == "interrupted" and snapshot.request:
        from src.tabs.live_demo import _asr_configuration_notice
        st.caption("Resume continues unrecorded trials only. Recorded failures are not retried. An in-flight request lost before checkpointing can be billed again.")
        if st.button("Resume pending trials with saved setup", key="batch_continue", disabled=_asr_configuration_notice() is not None):
            _start(controller, snapshot.request, resume=True)
            st.rerun()


def render_batch_evaluation_tab():
    from src.tabs.live_demo import _asr_configuration_notice
    controller = _controller()
    snapshot = controller.snapshot()
    _adopt_snapshot(snapshot)
    st.subheader("Batch evaluation")
    notice = _asr_configuration_notice()
    if notice:
        st.info("Batch speech recognition is unavailable. Check Setup in the sidebar. Saved checkpoints can still be inspected.")
    busy = snapshot.busy
    locales = st.multiselect("Locales", LOCALES, default=saved_widget_value("batch_locales", list(LOCALES)), key="batch_locales", disabled=busy,
                            on_change=remember_widget, args=("batch_locales",))
    full = st.checkbox("Use full datasets", value=saved_widget_value("use_full_dataset", False), key="use_full_dataset", disabled=busy,
                       on_change=remember_widget, args=("use_full_dataset",))
    samples = -1 if full else int(st.number_input("Samples per locale", min_value=1, max_value=10000, value=saved_widget_value("batch_samples", 28),
        step=14, key="batch_samples", disabled=busy, on_change=remember_widget, args=("batch_samples",)))
    if full:
        st.caption("Full datasets retain their original intent distribution. The sampling seed does not select or randomize a subset.")
    else:
        st.caption(f"Seeded intent quotas are shared across locales. Requested cap: {samples * len(locales)} trials; actual shared support may be lower. Each nonempty successful trial uses up to two embeddings and one ASR request.")
    with st.expander("Advanced settings and recovery"):
        seed = int(st.number_input("Sampling seed", min_value=0, max_value=2**32 - 1, value=saved_widget_value("batch_seed", get_settings().evaluation_seed),
            key="batch_seed", disabled=busy or full, on_change=remember_widget, args=("batch_seed",)))
        resume = st.checkbox("Resume an existing run", value=saved_widget_value("batch_resume", False), key="batch_resume", disabled=busy,
                             on_change=remember_widget, args=("batch_resume",))
        if resume:
            path = st.text_input("Path to existing trials.jsonl", value=saved_widget_value("batch_checkpoint", ""), key="batch_checkpoint", disabled=busy,
                                 on_change=remember_widget, args=("batch_checkpoint",))
            if st.button("Inspect saved checkpoint", key="batch_inspect", disabled=busy or not path.strip()):
                try:
                    controller.load(inspect_checkpoint(path))
                    st.rerun()
                except (ValueError, RuntimeError):
                    st.error("The checkpoint could not be loaded. Check its saved manifest and path. No provider was called.")
            st.caption("Inspect first to view a saved run without model requests. Resume uses that run's frozen configuration, not these new-run controls.")
    configuration = model_configuration(get_settings(), {})
    try:
        request = BatchRequest.create(locales, samples, seed, configuration)
    except ValueError:
        request = None
    same = bool(request and snapshot.request and snapshot.request.signature == request.signature)
    protected = same and snapshot.phase in {"completed", "interrupted"}
    if st.button("Start batch evaluation", key="run_batch", type="primary",
                 disabled=bool(notice) or busy or not request or resume or protected):
        _start(controller, request)
        st.rerun()
    if protected and not busy:
        st.caption("This setup already has a saved run. Continue pending trials or deliberately create a separate replication.")
        if st.button("Start a new replication", key="batch_replicate", disabled=bool(notice)):
            _start(controller, request, replication=True)
            st.rerun()

    initial = controller.snapshot()

    @st.fragment(run_every=0.75 if initial.busy else None)
    def poll():
        before = st.session_state.get("_batch_adopted")
        _render_status(controller)
        after = controller.snapshot()
        if initial.busy and not after.busy and before != st.session_state.get("_batch_adopted"):
            st.rerun()

    poll()
    snapshot = controller.snapshot()
    if not snapshot.busy and snapshot.job_id:
        frame = snapshot.dataframe()
        if not frame.empty:
            _display_batch_results(frame)
        elif snapshot.phase == "completed":
            st.warning("This run completed without recorded measurements. No rate can be calculated.")
        if snapshot.manifest_json != "{}":
            _export_batch(frame, key="current")
    previous = st.session_state.get("batch_results")
    if previous is not None and (snapshot.busy or snapshot.phase != "completed" or not _has_usable_results(snapshot.dataframe())):
        st.info("Previous valid results are retained separately from the current attempt.")
        with st.expander("Previous valid results"):
            _display_batch_results(previous)
            _export_batch(previous, key="previous")
