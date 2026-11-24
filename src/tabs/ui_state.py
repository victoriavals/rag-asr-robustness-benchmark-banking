"""Session-owned widget values and shared, credential-free UI data."""
from typing import Any

import pandas as pd
import streamlit as st


def saved_widget_value(key: str, default: Any, *, allowed=None) -> Any:
    value = st.session_state.get("ui_widget_values", {}).get(key, default)
    return value if allowed is None or value in allowed else default


def restore_widget(key: str, *, allowed=None) -> None:
    values = st.session_state.get("ui_widget_values", {})
    if key not in st.session_state and key in values:
        value = values[key]
        if allowed is None or value in allowed:
            st.session_state[key] = value


def remember_widget(key: str) -> None:
    values = st.session_state.get("ui_widget_values", {})
    st.session_state.ui_widget_values = {**values, key: st.session_state[key]}


def navigation_choice(label: str, options: tuple[str, ...], key: str) -> str:
    restore_widget(key, allowed=options)

    def remember_selection() -> None:
        if st.session_state[key] is None:
            saved = st.session_state.get("ui_widget_values", {}).get(key)
            st.session_state[key] = saved if saved in options else options[0]
        remember_widget(key)

    return st.segmented_control(
        label, options, default=None if key in st.session_state else options[0], key=key,
        on_change=remember_selection, label_visibility="collapsed",
    ) or options[0]


def model_configuration(settings, health) -> dict[str, Any]:
    names = (
        "embedding_model", "embedding_dimensions", "embedding_send_dimensions",
        "asr_model", "chat_model", "retrieval_top_k", "retrieval_score_threshold",
        "dataset_revision",
    )
    result = {name: getattr(settings, name) for name in names}
    result.update({
        f"{purpose}_base_url": settings.provider_address(purpose)
        for purpose in ("embedding", "asr", "chat")
    })
    result["collection_name"] = settings.qdrant_collection
    if version := health.get("kb_version"):
        result["kb_version"] = version
    if fingerprint := health.get("kb_fingerprint", health.get("fingerprint")):
        result["kb_fingerprint"] = fingerprint
    return result


def _spreadsheet_safe_csv(df: pd.DataFrame) -> str:
    """Escape text formula prefixes while retaining numeric columns."""
    safe = df.copy()
    for column in safe.columns:
        if safe[column].dtype == object or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(
                lambda value: "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
            )
    return safe.to_csv(index=False)
