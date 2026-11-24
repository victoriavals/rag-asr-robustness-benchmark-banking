"""Actual UI, evaluation, and HTTP serialization with a simulated provider."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import wave
import json
import httpx
from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

from src import evaluation_engine as evaluation
from src import sumopod_services as services
from src import vector_db
from src.config import SUMOPOD_BASE_URL, get_settings
from src.minds14_loader import MINDS14Loader
import ingest_data


def test_real_batch_contract_empty_asr_and_recovery(monkeypatch, tmp_path, request, settle_batch):
    monkeypatch.setenv('EMBEDDING_DIMENSIONS', '2')
    monkeypatch.setenv('API_KEY', 'offline-integration-key')
    monkeypatch.setenv('ASR_MODEL', 'fixture-transcription-model')
    get_settings.cache_clear()
    client = QdrantClient(':memory:')
    request.addfinalizer(client.close)
    monkeypatch.setattr(vector_db, '_client', client)
    state = SimpleNamespace(asr_mode='normal', asr_calls=0, embedding_calls=0)

    def respond(http_request):
        assert http_request.url.host == 'ai.sumopod.com'
        if http_request.url.path == '/v1/embeddings':
            assert json.loads(http_request.content)['input'].strip()
            state.embedding_calls += 1
            return httpx.Response(200, json={'data': [{'embedding': [1., 0.]}]})
        assert http_request.url.path == '/v1/audio/transcriptions'
        assert b'fixture-transcription-model' in http_request.content
        state.asr_calls += 1
        if state.asr_mode == 'failure':
            raise httpx.ConnectError('Simulated network failure', request=http_request)
        return httpx.Response(200, json={
            'text': '' if state.asr_mode == 'empty' else 'Check my balance.'
        })

    transport_client = httpx.Client(base_url=SUMOPOD_BASE_URL, transport=httpx.MockTransport(respond))
    request.addfinalizer(transport_client.close)
    monkeypatch.setattr(services, '_get_client', lambda: transport_client)
    kb = Path(__file__).resolve().parents[1] / 'data' / 'bank_policy.txt'
    ingest_data.ingest_document(kb)
    state.embedding_calls = 0
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b'\0\0' * 80)
    raw = buffer.getvalue()

    class CachedFixtureLoader(MINDS14Loader):
        def __init__(self, language='en-US', revision=None):
            super().__init__(language, cache_dir=str(tmp_path / 'audio'), revision=revision)
            self._dataset = [dict(id=f'fixture-{i}', transcription='Check my balance',
                english_transcription='Check my balance', intent_class=i,
                audio={'bytes': raw, 'path': f'fixture-{i}.wav'}) for i in range(14)]

    monkeypatch.setattr(evaluation, 'MINDS14Loader', CachedFixtureLoader)
    app = Path(__file__).resolve().parents[1] / 'main.py'
    at = AppTest.from_file(str(app), default_timeout=60).run()
    at.button_group(key='mode_selector').set_value(['Research Lab']).run()
    at.button_group(key='research_mode').set_value(['Batch']).run()
    at.multiselect(key='batch_locales').set_value(['en-US']).run()
    at.number_input(key='batch_samples').set_value(14).run()

    run_batch = evaluation.run_batch_evaluation

    def interrupt_first(**kwargs):
        report = kwargs['progress_callback']

        def interrupt(current, total, locale, sample_id):
            report(current, total, locale, sample_id)
            raise InterruptedError('Simulated stop after the first durable trial')

        return run_batch(**{**kwargs, 'progress_callback': interrupt})

    monkeypatch.setattr(evaluation, 'run_batch_evaluation', interrupt_first)
    at.button(key='run_batch').click().run()
    settle_batch(at)
    interrupted = at.session_state['last_batch_attempt']
    assert len(interrupted) == 1 and state.asr_calls == 1
    assert Path(interrupted.attrs['checkpoint_path']).is_file()
    assert interrupted.attrs['manifest']['config']['logical_collection'] == 'documents'

    monkeypatch.setattr(evaluation, 'run_batch_evaluation', run_batch)
    at.button(key='batch_continue').click().run()
    settle_batch(at)
    assert not at.exception and not at.error
    first = at.session_state['batch_results']
    assert len(first) == 14 and first['status'].eq('success').all()
    assert first['wer_score'].eq(0).all()
    assert state.asr_calls == 14 and state.embedding_calls == 28
    assert first.attrs['manifest']['completed'] is True
    assert Path(first.attrs['checkpoint_path']).is_file()

    # A real wrapper returns an empty string; the actual engine must retain the
    # model outcome rather than misclassifying it as an infrastructure exception.
    state.asr_mode = 'empty'
    state.embedding_calls = 0
    at.button(key='batch_replicate').click().run()
    settle_batch(at)
    assert not at.exception
    empty = at.session_state['last_batch_attempt']
    assert len(empty) == 14 and empty['wer_score'].eq(1).all()
    assert empty['hit_rate_voice'].eq(0).all()
    assert empty['error_category'].eq('empty_asr').all()
    assert state.embedding_calls == 14  # No empty-text voice embedding calls.

    # Infrastructure-only voice failures preserve the previous speech results,
    # while keeping the new baseline/failure records available for export.
    previous_id = at.session_state['batch_results'].attrs['manifest']['run_id']
    state.asr_mode = 'failure'
    at.button(key='batch_replicate').click().run()
    settle_batch(at)
    assert not at.exception
    failed = at.session_state['last_batch_attempt']
    assert len(failed) == 14 and failed['wer_score'].isna().all()
    assert failed['hit_rate_text'].notna().all()
    assert at.session_state['batch_results'].attrs['manifest']['run_id'] == previous_id
    assert any('Previous valid results' in element.value for element in at.info)
