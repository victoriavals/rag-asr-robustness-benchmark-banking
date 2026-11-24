"""Exercise the real provider wrappers at the evaluator boundary, offline."""

import json
import wave
from io import BytesIO
import httpx
import pytest

from src import evaluation_engine as engine
from src import sumopod_services as services
from src.config import Settings, SUMOPOD_BASE_URL
from src.knowledge_base import INTENT_NAMES


def audio_buffer():
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b'\x00\x00' * 80)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize('transcription,category,wer,voice_hit', [
    ('', 'empty_asr', 1.0, 0),
    (None, 'infrastructure', None, None),
])
def test_real_asr_wrapper_distinguishes_no_speech_from_provider_failure(
    monkeypatch, transcription, category, wer, voice_hit
):
    config = Settings(_env_file=None, api_key='fixture-api-key', embedding_dimensions=2,
                      asr_model='fixture-transcription-model', request_timeout_seconds=5)
    monkeypatch.setattr(services, '_get_settings', lambda: config)
    embedded = []

    def respond(request):
        if request.url.path == '/v1/embeddings':
            text = json.loads(request.content)['input']
            embedded.append(text)
            assert text.strip(), 'An empty ASR result must never be embedded'
            return httpx.Response(200, json={'data': [{'embedding': [1., 0.]}]})
        assert request.url.path == '/v1/audio/transcriptions'
        return httpx.Response(200, json={'text': transcription})
    monkeypatch.setattr(engine, 'get_embedding', services.get_embedding)
    monkeypatch.setattr(engine, 'transcribe_audio', services.transcribe_audio)
    monkeypatch.setattr(engine, 'search_documents', lambda **kwargs: [{
        'id': 'balance-policy', 'text': 'Check your balance in the app.', 'score': 1.0,
        'intent_ids': ['balance'], 'kb_version': 'v1', 'source': 'policy.txt', 'chunk_index': 0,
    }])
    sample = {'id': 'en-US:0', 'transcription': 'check my balance',
        'intent_class': INTENT_NAMES.index('balance'), 'intent_name': 'balance'}
    with httpx.Client(base_url=SUMOPOD_BASE_URL, transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(services, '_get_client', lambda: client)
        row = engine.evaluate_sample(sample, 'en-US', audio_buffer())
    assert embedded == ['check my balance']
    assert row['hit_rate_text'] == 1
    assert row['wer_score'] == wer
    assert row['hit_rate_voice'] == voice_hit
    assert row['error_category'] == category
    assert json.loads(row['errors_json'])[-1]['category'] == category
