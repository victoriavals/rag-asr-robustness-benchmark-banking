import pytest
from pydantic import ValidationError
from src.config import SUMOPOD_BASE_URL, Settings


def test_optional_key_and_asr_model_do_not_prevent_opening_app():
    cfg = Settings(_env_file=None)
    assert cfg.api_key is None
    assert cfg.asr_model is None
    assert cfg.chat_model == 'glm-5.3-flash'


def test_secrets_are_masked_and_not_exported():
    cfg = Settings(_env_file=None, api_key='  fixture-secret  ', qdrant_api_key='qdrant-secret')
    assert cfg.api_key.get_secret_value() == 'fixture-secret'
    assert 'fixture-secret' not in repr(cfg)
    assert 'qdrant-secret' not in repr(cfg)
    assert not any('key' in name or 'url' in name for name in cfg.public_metadata())


@pytest.mark.parametrize('value', ['', 'API', 'your_sumopod_api_key_here', 'sk-xxx'])
def test_placeholders_are_unconfigured(value):
    assert Settings(_env_file=None, api_key=value).api_key is None


@pytest.mark.parametrize('url', ['https://', 'http:// bad host', 'ftp://localhost',
                                'http://user:private@localhost:6333', 'http://localhost:99999'])
def test_invalid_urls_fail_without_exposing_input(url):
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, qdrant_url=url)
    assert 'input_value=' not in str(caught.value)
    assert 'user:private' not in str(caught.value)


def test_explicit_chat_name_takes_precedence(monkeypatch):
    monkeypatch.setenv('MODEL', 'glm-5.3-flash')
    monkeypatch.setenv('CHAT_MODEL', 'explicit-chat-model')
    assert Settings(_env_file=None).chat_model == 'explicit-chat-model'


def test_single_key_and_models_loaded_from_env_file(tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text('API_KEY=fixture-sumo-secret\nMODEL=qa-chat\nASR_MODEL=qa-audio\n', encoding='utf-8')
    cfg = Settings(_env_file=env_file)
    assert cfg.api_key.get_secret_value() == 'fixture-sumo-secret'
    assert cfg.chat_model == 'qa-chat'
    assert cfg.asr_model == 'qa-audio'
    assert cfg.embedding_model == 'text-embedding-3-small'
    assert cfg.embedding_dimensions == 1536
    assert cfg.embedding_send_dimensions is False
    assert cfg.llm_temperature == 0.7
    for purpose in ('chat', 'embedding', 'asr'):
        assert cfg.provider_address(purpose) == SUMOPOD_BASE_URL


@pytest.mark.parametrize('value', ['', '   ', None])
def test_unconfigured_asr_model_leaves_voice_disabled(value):
    assert Settings(_env_file=None, asr_model=value).asr_model is None


def test_asr_model_rejects_control_characters():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, asr_model='audio\nmodel')


def test_custom_embedding_dimensions_can_be_requested():
    cfg = Settings(_env_file=None, embedding_dimensions=768, embedding_send_dimensions=True)
    assert cfg.embedding_send_dimensions is True
    assert cfg.embedding_dimensions == 768
