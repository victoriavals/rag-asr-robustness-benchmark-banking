"""
Live Demo Tab for the Dual-Mode RAG Application.

This module provides the interactive chat interface where users can
ask questions via text or voice input and receive AI-powered responses
based on the bank policy knowledge base.
"""

from io import BytesIO
from typing import TypedDict

import streamlit as st

from src.openai_services import get_embedding, get_llm_response, transcribe_audio
from src.vector_db import search_similar_docs


class ChatMessage(TypedDict):
    """
    Represents a single chat message.

    Attributes:
        role (str): Either 'user' or 'assistant'.
        content (str): The message content.
    """

    role: str
    content: str


def _initialize_session_state() -> None:
    """
    Initialize session state variables for chat history.

    Sets up the chat_history list if it doesn't exist in session state.

    Params:
        None

    Returns:
        None
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[ChatMessage] = []


def _display_chat_history() -> None:
    """
    Display all messages in the chat history.

    Renders each message using Streamlit's chat_message component
    with appropriate role styling.

    Params:
        None

    Returns:
        None
    """
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _add_message(role: str, content: str) -> None:
    """
    Add a message to the chat history.

    Params:
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The message content.

    Returns:
        None
    """
    message: ChatMessage = {"role": role, "content": content}
    st.session_state.chat_history.append(message)


def _process_query(query: str) -> str:
    """
    Process a user query through the RAG pipeline.

    Generates an embedding for the query, retrieves relevant context
    from the vector database, and generates an LLM response.

    Params:
        query (str): The user's question.

    Returns:
        str: The generated response from the LLM.

    Raises:
        ConnectionError: If Qdrant server is not accessible.
    """
    try:
        # Generate embedding for the query
        query_embedding: list[float] = get_embedding(query)

        # Retrieve relevant documents from vector database
        relevant_docs: list[str] = search_similar_docs(
            query_vector=query_embedding,
            top_k=3,
        )

        # Combine retrieved documents into context
        context: str = "\n\n---\n\n".join(relevant_docs) if relevant_docs else ""

        # Generate response using LLM
        response: str = get_llm_response(prompt=query, context=context)

        return response

    except ConnectionRefusedError as e:
        raise ConnectionError(
            "❌ **Qdrant Server Not Available**\n\n"
            "The vector database (Qdrant) is not accessible. Please check:\n\n"
            "1. Qdrant server is running at the configured URL\n"
            "2. Check your `.env` file for correct `QDRANT_URL`\n"
            "3. Ensure your firewall allows the connection\n\n"
            f"Technical details: {str(e)}"
        )
    except Exception as e:
        # Re-raise if it's already a connection error
        if "connection" in str(e).lower() or "10061" in str(e):
            raise ConnectionError(
                "❌ **Cannot Connect to Qdrant Server**\n\n"
                "The Qdrant vector database is not accessible.\n\n"
                "**Steps to fix:**\n"
                "1. Make sure Qdrant server is running\n"
                "2. Verify `QDRANT_URL` in your `.env` file\n"
                "3. Run `python ingest_data.py` to create the collection\n\n"
                f"Error: {str(e)}"
            )
        raise


def _render_audio_input() -> str | None:
    """
    Render audio input section and process uploaded audio.

    Provides a file uploader for audio files and transcribes
    the audio if provided.

    Params:
        None

    Returns:
        str | None: The transcribed text if audio was provided, None otherwise.
    """
    st.markdown("#### 🎤 Voice Input")
    
    # Try to use st.audio_input if available (Streamlit >= 1.33)
    try:
        audio_data = st.audio_input(
            "Record your question",
            key="audio_recorder",
        )
        if audio_data is not None:
            with st.spinner("Transcribing audio..."):
                # Convert to BytesIO and add name attribute
                audio_buffer: BytesIO = BytesIO(audio_data.getvalue())
                audio_buffer.name = "recording.wav"
                transcribed_text: str = transcribe_audio(audio_buffer)
                st.success(f"Transcribed: {transcribed_text}")
                return transcribed_text
    except AttributeError:
        # Fallback to file uploader if audio_input is not available
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            key="audio_uploader",
        )
        if uploaded_audio is not None:
            with st.spinner("Transcribing audio..."):
                audio_buffer = BytesIO(uploaded_audio.getvalue())
                audio_buffer.name = uploaded_audio.name
                transcribed_text = transcribe_audio(audio_buffer)
                st.success(f"Transcribed: {transcribed_text}")
                return transcribed_text

    return None


def _check_qdrant_connection() -> bool:
    """
    Check if Qdrant server is accessible.

    Params:
        None

    Returns:
        bool: True if connection successful, False otherwise.
    """
    try:
        from src.vector_db import get_client
        client = get_client()
        # Try to list collections as a connection test
        client.get_collections()
        return True
    except Exception:
        return False


def render_live_demo_tab() -> None:
    """
    Render the Live Demo tab with chat interface.

    Provides a complete chat experience with:
    - Chat history display
    - Text input via chat_input
    - Voice input via audio recorder/uploader
    - RAG-powered responses

    Params:
        None

    Returns:
        None
    """
    # Initialize session state
    _initialize_session_state()

    # Header
    st.header("💬 Live Demo - Banking Assistant")
    st.markdown(
        "Ask questions about our banking policies using text or voice input."
    )

    # Check Qdrant connection on first load
    if "qdrant_checked" not in st.session_state:
        st.session_state.qdrant_checked = True
        if not _check_qdrant_connection():
            st.error(
                "⚠️ **Qdrant Server Not Connected**\n\n"
                "The vector database (Qdrant) is currently not accessible. "
                "This may affect the quality of responses as the system cannot retrieve relevant context.\n\n"
                "**Please check:**\n"
                "- Qdrant server is running\n"
                "- `.env` file has correct `QDRANT_URL`\n"
                "- Run `python ingest_data.py` to set up the database"
            )

    # Sidebar for audio input
    with st.sidebar:
        st.markdown("---")
        transcribed_query: str | None = _render_audio_input()
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    _display_chat_history()

    # Process transcribed audio query
    if transcribed_query:
        # Add user message
        _add_message("user", transcribed_query)
        with st.chat_message("user"):
            st.markdown(transcribed_query)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response: str = _process_query(transcribed_query)
                    st.markdown(response)
                    _add_message("assistant", response)
                except Exception as e:
                    error_msg: str = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    _add_message("assistant", error_msg)

    # Text input
    user_input: str | None = st.chat_input(
        "Type your question here...",
        key="chat_text_input",
    )

    if user_input:
        # Add user message
        _add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = _process_query(user_input)
                    st.markdown(response)
                    _add_message("assistant", response)
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    _add_message("assistant", error_msg)
