"""
Data Ingestion Script for the Dual-Mode RAG Application.

This script loads text documents, chunks them, generates embeddings,
and upserts the data into the Qdrant vector database.

Usage:
    python ingest_data.py
"""

import uuid
from pathlib import Path

from qdrant_client.http import models

from src.config import settings
from src.openai_services import get_embedding
from src.vector_db import recreate_collection, upsert_points


def load_text_file(file_path: str | Path) -> str:
    """
    Load text content from a file.

    Params:
        file_path (str | Path): The path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path: Path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        content: str = f.read()

    return content


def chunk_text(text: str, separator: str = "\n\n") -> list[str]:
    """
    Split text into chunks based on a separator.

    Filters out empty chunks and strips whitespace from each chunk.

    Params:
        text (str): The text to split into chunks.
        separator (str): The separator to split on. Default is double newline.

    Returns:
        list[str]: A list of non-empty text chunks.
    """
    raw_chunks: list[str] = text.split(separator)

    # Filter empty chunks and strip whitespace
    chunks: list[str] = [
        chunk.strip()
        for chunk in raw_chunks
        if chunk.strip()
    ]

    return chunks


def create_points(chunks: list[str]) -> list[models.PointStruct]:
    """
    Create Qdrant PointStruct objects from text chunks.

    Generates embeddings for each chunk and creates points with
    unique IDs and text payloads.

    Params:
        chunks (list[str]): A list of text chunks to embed.

    Returns:
        list[models.PointStruct]: A list of PointStruct objects ready
            for upserting into Qdrant.
    """
    points: list[models.PointStruct] = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")

        # Generate embedding for the chunk
        vector: list[float] = get_embedding(chunk)

        # Create a unique ID for the point
        point_id: str = str(uuid.uuid4())

        # Create the point with text payload
        point: models.PointStruct = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={"text": chunk, "chunk_index": i},
        )

        points.append(point)

    return points


def ingest_document(file_path: str | Path) -> int:
    """
    Ingest a document into the Qdrant vector database.

    Loads the document, chunks it, generates embeddings, and upserts
    the points into the configured Qdrant collection.

    Params:
        file_path (str | Path): The path to the document to ingest.

    Returns:
        int: The number of chunks ingested.
    """
    collection_name: str = settings.qdrant_collection

    print(f"Loading document: {file_path}")
    text: str = load_text_file(file_path)

    print("Chunking document...")
    chunks: list[str] = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print(f"Recreating collection: {collection_name}")
    recreate_collection(collection_name)

    print("Creating embeddings and points...")
    points: list[models.PointStruct] = create_points(chunks)

    print(f"Upserting {len(points)} points into Qdrant...")
    upsert_points(collection_name, points)

    print("Ingestion complete!")
    return len(chunks)


def main() -> None:
    """
    Main entry point for the ingestion script.

    Ingests the bank policy document into the vector database.

    Params:
        None

    Returns:
        None
    """
    # Path to the bank policy document
    data_file: Path = Path("data/bank_policy.txt")

    try:
        num_chunks: int = ingest_document(data_file)
        print(f"\nSuccessfully ingested {num_chunks} chunks into Qdrant!")
        print(f"Collection: {settings.qdrant_collection}")
        print(f"Qdrant URL: {settings.qdrant_url}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists.")

    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
