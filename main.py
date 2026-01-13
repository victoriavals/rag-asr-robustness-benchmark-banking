"""
Main Entry Point for the Hybrid Banking RAG System.

This is the main Streamlit application that integrates the Live Demo
and Research Lab modules into a unified interface.

Usage:
    streamlit run main.py
"""

import streamlit as st

from src.tabs.live_demo import render_live_demo_tab
from src.tabs.research_lab import render_research_tab


def _configure_page() -> None:
    """
    Configure the Streamlit page settings.

    Sets up the page title, layout, and favicon.

    Params:
        None

    Returns:
        None
    """
    st.set_page_config(
        page_title="Hybrid Banking RAG System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_sidebar() -> str:
    """
    Render the sidebar with mode selection and project info.

    Params:
        None

    Returns:
        str: The selected mode from the dropdown.
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/bank-building.png",
            width=80,
        )
        st.title("🏦 Banking RAG")

        st.markdown("---")

        # Mode selection
        selected_mode: str = st.selectbox(
            "🔀 Select Mode",
            options=[
                "Mode 1: Live Banking Assistant",
                "Mode 2: Research Lab (Robustness Analysis)",
            ],
            key="mode_selector",
        )

        st.markdown("---")

        # Project description
        st.markdown("### 📖 About This Project")
        st.markdown(
            """
            This is a **Dual-Mode RAG Application** designed for banking 
            customer service and research.
            
            **Features:**
            - 💬 **Live Demo**: Interactive chatbot with text & voice input
            - 🔬 **Research Lab**: Compare ASR vs Ground Truth performance
            
            **Tech Stack:**
            - 🤖 OpenAI GPT-4o & Whisper
            - 📊 Qdrant Vector Database
            - 🎨 Streamlit UI
            
            **Metrics:**
            - Word Error Rate (WER)
            - Embedding Cosine Similarity
            """
        )

        st.markdown("---")

        # Footer
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 12px;'>
                Built with ❤️ using Streamlit<br>
                Indonesia AI Final Project
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_mode


def main() -> None:
    """
    Main entry point for the Streamlit application.

    Configures the page, renders the sidebar, and displays
    the appropriate tab based on user selection.

    Params:
        None

    Returns:
        None
    """
    # Configure page settings (must be first Streamlit command)
    _configure_page()

    # Render sidebar and get selected mode
    selected_mode: str = _render_sidebar()

    # Render the appropriate tab based on selection
    if selected_mode == "Mode 1: Live Banking Assistant":
        render_live_demo_tab()
    elif selected_mode == "Mode 2: Research Lab (Robustness Analysis)":
        render_research_tab()


if __name__ == "__main__":
    main()
