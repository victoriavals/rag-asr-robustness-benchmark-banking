"""Streamlit entry point. Opening the app never contacts a model or database."""
import streamlit as st
from src.tabs.live_demo import cancel_pending_chat, render_live_demo_tab
from src.tabs.research_lab import render_research_tab
from src.tabs.ui_state import navigation_choice


def main() -> None:
    st.set_page_config(page_title="Banking RAG · ASR Benchmark", layout="wide", initial_sidebar_state="auto")
    # Streamlit 1.52 moves a collapsed sidebar offscreen while leaving its controls focusable.
    st.html("""<style>
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="false"] * { visibility: hidden !important; }
    </style>""")
    st.markdown("**Banking RAG**")
    mode = navigation_choice("Workspace", ("Assistant", "Research Lab"), "mode_selector")
    if mode == "Research Lab" and st.session_state.get("active_workspace") == "Assistant":
        cancel_pending_chat()
    st.session_state.active_workspace = mode
    if mode == "Assistant":
        render_live_demo_tab()
    else:
        render_research_tab()
    with st.sidebar.expander("Setup"):
        st.write("If the knowledge base is missing or incompatible, run the policy ingestion command and refresh its status.")
        st.code("python ingest_data.py", language="shell")
        st.caption("Configure the SumoPod API_KEY. Set ASR_MODEL to a transcription model enabled for your account to use voice features.")


if __name__ == "__main__":
    main()
