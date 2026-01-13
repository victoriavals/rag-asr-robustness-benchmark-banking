"""
Configuration module for the Dual-Mode RAG Application.

This module uses pydantic-settings to load and validate environment variables
from a .env file. It ensures that all required API keys and configuration
values are present before the application starts.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class validates that all required configuration values exist
    and provides type-safe access to environment variables.

    Attributes:
        openai_api_key (str): The OpenAI API key for Whisper, Embeddings, and GPT-4o.
        qdrant_url (str): The URL of the Qdrant vector database instance.
        qdrant_collection (str): The name of the Qdrant collection to use.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for Whisper, Embeddings, and GPT-4o",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant vector database instance",
    )
    qdrant_collection: str = Field(
        default="documents",
        description="Name of the Qdrant collection",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: str) -> str:
        """
        Validate that the OpenAI API key is not empty or a placeholder.

        Params:
            value (str): The OpenAI API key value to validate.

        Returns:
            str: The validated API key.

        Raises:
            ValueError: If the API key is empty or contains a placeholder value.
        """
        if not value or value.strip() == "":
            raise ValueError("OPENAI_API_KEY cannot be empty")
        if value in ("api_key", "your_api_key", "your-api-key", "sk-xxx"):
            raise ValueError(
                "OPENAI_API_KEY appears to be a placeholder. "
                "Please set a valid API key."
            )
        return value

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        """
        Validate that the Qdrant URL is properly formatted.

        Params:
            value (str): The Qdrant URL to validate.

        Returns:
            str: The validated Qdrant URL.

        Raises:
            ValueError: If the URL is empty or not properly formatted.
        """
        if not value or value.strip() == "":
            raise ValueError("QDRANT_URL cannot be empty")
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                "QDRANT_URL must start with http:// or https://"
            )
        return value

    @field_validator("qdrant_collection")
    @classmethod
    def validate_qdrant_collection(cls, value: str) -> str:
        """
        Validate that the Qdrant collection name is not empty.

        Params:
            value (str): The Qdrant collection name to validate.

        Returns:
            str: The validated collection name.

        Raises:
            ValueError: If the collection name is empty.
        """
        if not value or value.strip() == "":
            raise ValueError("QDRANT_COLLECTION cannot be empty")
        return value


def get_settings() -> Settings:
    """
    Create and return a Settings instance.

    This function loads the settings from environment variables and the .env file.
    It validates all required values and raises an error if any are missing or invalid.

    Params:
        None

    Returns:
        Settings: A validated Settings instance with all configuration values.

    Raises:
        ValidationError: If any required environment variable is missing or invalid.
    """
    return Settings()


# Singleton instance for easy access throughout the application
settings: Settings = get_settings()
