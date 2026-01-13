"""
OpenAI Service Wrappers for the Dual-Mode RAG Application.

This module provides wrapper functions for interacting with OpenAI APIs,
including embeddings, audio transcription, and chat completions.
All functions use the configuration from src/config.py.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from src.config import settings


# Initialize OpenAI client with API key from settings
_client: OpenAI = OpenAI(api_key=settings.openai_api_key)


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text using OpenAI's embedding model.

    Uses the text-embedding-3-small model to create a vector representation
    of the input text for semantic search and similarity comparisons.

    Params:
        text (str): The text to generate an embedding for.

    Returns:
        list[float]: A list of floats representing the embedding vector.

    Raises:
        ValueError: If the input text is empty.
        APIError: If the OpenAI API returns an error.
        APIConnectionError: If there's a connection issue with the API.
        RateLimitError: If the API rate limit is exceeded.
    """
    if not text or text.strip() == "":
        raise ValueError("Input text cannot be empty")

    try:
        response = _client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        embedding: list[float] = response.data[0].embedding
        return embedding

    except RateLimitError as e:
        raise RateLimitError(
            f"OpenAI API rate limit exceeded. Please try again later. Error: {e}"
        ) from e
    except APIConnectionError as e:
        raise APIConnectionError(
            f"Failed to connect to OpenAI API. Check your network. Error: {e}"
        ) from e
    except APIError as e:
        raise APIError(
            f"OpenAI API error during embedding generation: {e}"
        ) from e


def transcribe_audio(file_path_or_buffer: Union[str, Path, BytesIO]) -> str:
    """
    Transcribe audio to text using OpenAI's Whisper API.

    Accepts either a file path to an audio file or a BytesIO buffer
    (useful for Streamlit file uploads).

    Params:
        file_path_or_buffer (Union[str, Path, BytesIO]): Either a path to an
            audio file (as string or Path object) or a BytesIO buffer
            containing audio data.

    Returns:
        str: The transcribed text from the audio.

    Raises:
        FileNotFoundError: If a file path is provided but the file doesn't exist.
        ValueError: If the input is neither a valid path nor a BytesIO object.
        APIError: If the OpenAI API returns an error.
        APIConnectionError: If there's a connection issue with the API.
        RateLimitError: If the API rate limit is exceeded.
    """
    try:
        if isinstance(file_path_or_buffer, BytesIO):
            # Handle BytesIO buffer (e.g., from Streamlit file uploader)
            # Ensure the buffer has a name attribute for the API
            if not hasattr(file_path_or_buffer, "name"):
                file_path_or_buffer.name = "audio.wav"
            
            audio_file = file_path_or_buffer
            response = _client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            transcription: str = response.text
            return transcription

        elif isinstance(file_path_or_buffer, (str, Path)):
            # Handle file path
            file_path: Path = Path(file_path_or_buffer)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            with open(file_path, "rb") as audio_file:
                response = _client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            transcription = response.text
            return transcription

        else:
            raise ValueError(
                f"Invalid input type: {type(file_path_or_buffer)}. "
                "Expected str, Path, or BytesIO."
            )

    except RateLimitError as e:
        raise RateLimitError(
            f"OpenAI API rate limit exceeded. Please try again later. Error: {e}"
        ) from e
    except APIConnectionError as e:
        raise APIConnectionError(
            f"Failed to connect to OpenAI API. Check your network. Error: {e}"
        ) from e
    except APIError as e:
        raise APIError(
            f"OpenAI API error during audio transcription: {e}"
        ) from e


def get_llm_response(prompt: str, context: str) -> str:
    """
    Generate a response from GPT-4o based on the prompt and context.

    Uses a system prompt configured for a banking assistant and provides
    the context for answering user queries.

    Params:
        prompt (str): The user's question or prompt.
        context (str): The relevant context retrieved from the knowledge base
            to help answer the prompt.

    Returns:
        str: The generated response from the LLM.

    Raises:
        ValueError: If the prompt is empty.
        APIError: If the OpenAI API returns an error.
        APIConnectionError: If there's a connection issue with the API.
        RateLimitError: If the API rate limit is exceeded.
    """
    if not prompt or prompt.strip() == "":
        raise ValueError("Prompt cannot be empty")

    system_prompt: str = (
        "You are a helpful banking assistant. "
        "Answer based on the context provided."
    )

    user_message: str = f"""Context:
{context}

Question:
{prompt}

Please provide a helpful and accurate answer based on the context above."""

    try:
        response = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        
        answer: str = response.choices[0].message.content
        return answer if answer else ""

    except RateLimitError as e:
        raise RateLimitError(
            f"OpenAI API rate limit exceeded. Please try again later. Error: {e}"
        ) from e
    except APIConnectionError as e:
        raise APIConnectionError(
            f"Failed to connect to OpenAI API. Check your network. Error: {e}"
        ) from e
    except APIError as e:
        raise APIError(
            f"OpenAI API error during LLM response generation: {e}"
        ) from e
