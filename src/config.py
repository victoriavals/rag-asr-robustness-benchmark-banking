"""Validated, lazy configuration; importing the app never requires credentials."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_REVISION = '40ce77cb32a384e4d50a568e1ec39ac804019d33'
SUMOPOD_BASE_URL = 'https://ai.sumopod.com/v1'
SUMOPOD_CHAT_MODEL = 'glm-5.3-flash'
ServicePurpose = Literal['embedding', 'asr', 'chat']

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / '.env', env_file_encoding='utf-8',
        case_sensitive=False, extra='ignore', populate_by_name=True, hide_input_in_errors=True)
    api_key: SecretStr | None = None
    chat_model: str = Field(default=SUMOPOD_CHAT_MODEL, validation_alias=AliasChoices('CHAT_MODEL', 'MODEL'))
    embedding_model: str = 'text-embedding-3-small'
    embedding_dimensions: int = Field(default=1536, ge=1, le=3072)
    embedding_send_dimensions: bool = False
    asr_model: str | None = None
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    chat_max_tokens: int = Field(default=1024, ge=1)
    request_timeout_seconds: float = Field(default=30, gt=0, le=600)
    max_audio_bytes: int = Field(default=25*1024*1024, ge=1, le=25*1024*1024)
    max_history_messages: int = Field(default=12, ge=0, le=50)
    max_context_chars: int = Field(default=24000, ge=100, le=100000)
    qdrant_url: str = 'http://localhost:6333'
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = 'documents'
    retrieval_top_k: int = Field(default=3, ge=1, le=20)
    retrieval_score_threshold: float | None = Field(default=None, ge=-1, le=1)
    evaluation_seed: int = 42
    dataset_revision: str = Field(default=DATASET_REVISION,
        validation_alias=AliasChoices('DATASET_REVISION', 'MINDS14_REVISION'))
    results_dir: Path = Path('results')

    def provider_address(self, purpose: ServicePurpose) -> str:
        """Return the fixed gateway recorded in retrieval and benchmark provenance."""
        return SUMOPOD_BASE_URL

    @field_validator('api_key', 'qdrant_api_key', mode='before')
    @classmethod
    def normalize_secret(cls, value):
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if value is None:
            return None
        value = str(value).strip()
        placeholders = {'', 'api', 'api_key', 'your_api_key', 'your-api-key',
                        'your_qdrant_api_key_here', 'your_sumopod_api_key_here', 'sk-xxx'}
        return None if value.lower() in placeholders else value

    @field_validator('qdrant_url', mode='before')
    @classmethod
    def validate_url(cls, value):
        if value is None or not str(value).strip():
            return None
        value = str(value).strip().rstrip('/')
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme in {'http', 'https'} and parsed.hostname
                and not any(char.isspace() for char in value)
                and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment)
            parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('Use an HTTP(S) URL with a valid host and no embedded credentials')
        return value

    @field_validator('qdrant_url')
    @classmethod
    def require_qdrant_url(cls, value):
        if value is None:
            raise ValueError('QDRANT_URL cannot be empty')
        return value

    @field_validator('qdrant_collection', 'chat_model', 'embedding_model', 'dataset_revision')
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value or any(char in value for char in '\r\n\x00'):
            raise ValueError('Configuration must be nonempty and contain no control characters')
        return value

    @field_validator('asr_model')
    @classmethod
    def validate_asr_model(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return cls.nonempty_text(value)

    def public_metadata(self) -> dict:
        return {name: getattr(self, name) for name in ('chat_model', 'embedding_model', 'embedding_dimensions',
            'asr_model', 'llm_temperature', 'retrieval_top_k', 'retrieval_score_threshold', 'evaluation_seed', 'dataset_revision')}

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
