"""
Vector Database module for Qdrant interactions.

This module provides functions for interacting with the Qdrant vector database,
including searching for similar documents based on query vectors.
"""

import time

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from src.config import settings


# Initialize Qdrant client using URL from settings
# Note: check_compatibility=False suppresses version mismatch warnings
_client: QdrantClient = QdrantClient(
    url=settings.qdrant_url,
    check_compatibility=False,
)


def get_client() -> QdrantClient:
    """
    Get the Qdrant client instance.

    Provides access to the initialized Qdrant client for advanced operations
    not covered by the wrapper functions in this module.

    Params:
        None

    Returns:
        QdrantClient: The initialized Qdrant client instance.
    """
    return _client


def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant.

    Params:
        collection_name (str): The name of the collection to check.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    try:
        _client.get_collection(collection_name=collection_name)
        return True
    except UnexpectedResponse:
        return False


def create_collection(
    collection_name: str,
    vector_size: int = 1536,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Create a new collection in Qdrant.

    Params:
        collection_name (str): The name of the collection to create.
        vector_size (int): The dimension of the vectors. Default is 1536
            (for text-embedding-3-small).
        distance (models.Distance): The distance metric to use.
            Default is COSINE.

    Returns:
        None

    Raises:
        UnexpectedResponse: If the collection creation fails.
    """
    _client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


def delete_collection(collection_name: str) -> None:
    """
    Delete a collection from Qdrant.

    Params:
        collection_name (str): The name of the collection to delete.

    Returns:
        None

    Raises:
        UnexpectedResponse: If the collection deletion fails.
    """
    _client.delete_collection(collection_name=collection_name)


def recreate_collection(
    collection_name: str,
    vector_size: int = 1536,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Recreate a collection by deleting it if it exists and creating a new one.

    This is useful for clean ingestion during testing to avoid duplicate data.

    Params:
        collection_name (str): The name of the collection to recreate.
        vector_size (int): The dimension of the vectors. Default is 1536
            (for text-embedding-3-small).
        distance (models.Distance): The distance metric to use.
            Default is COSINE.

    Returns:
        None
    """
    if collection_exists(collection_name):
        delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")

    create_collection(collection_name, vector_size, distance)
    print(f"Created new collection: {collection_name}")


def upsert_points(
    collection_name: str,
    points: list[models.PointStruct],
) -> None:
    """
    Upsert points (vectors with payloads) into a Qdrant collection.

    Params:
        collection_name (str): The name of the collection to upsert into.
        points (list[models.PointStruct]): A list of PointStruct objects
            containing the id, vector, and payload for each point.

    Returns:
        None

    Raises:
        UnexpectedResponse: If the upsert operation fails.
    """
    _client.upsert(
        collection_name=collection_name,
        points=points,
    )


def search_similar_docs(
    query_vector: list[float],
    top_k: int = 3,
    collection_name: str | None = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Search for similar documents in the Qdrant collection.

    Performs a vector similarity search and returns the text payloads
    of the most similar documents. Includes retry logic for transient
    connection errors.

    Params:
        query_vector (list[float]): The query vector to search with.
        top_k (int): The number of top results to return. Default is 3.
        collection_name (str | None): The collection to search in.
            If None, uses the collection from settings.
        max_retries (int): Maximum number of retry attempts for transient errors.

    Returns:
        list[str]: A list of text payloads from the most similar documents.

    Raises:
        ResponseHandlingException: If all retry attempts fail.
        UnexpectedResponse: If the search operation fails with a non-transient error.
    """
    if collection_name is None:
        collection_name = settings.qdrant_collection

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            # Use query_points for qdrant-client 1.7+ (replaces deprecated search method)
            results = _client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
            )

            # Extract text payloads from results
            documents: list[str] = []
            for point in results.points:
                if point.payload and "text" in point.payload:
                    text: str = point.payload["text"]
                    documents.append(text)

            return documents

        except ResponseHandlingException as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time: float = 2 ** attempt
                print(f"Qdrant connection error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            continue

        except Exception as e:
            # For non-transient errors, raise immediately
            raise e

    # If all retries failed, raise the last exception
    if last_exception:
        raise last_exception

    return []
