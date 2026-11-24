from pathlib import Path
from unittest.mock import Mock
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import Settings
from src import vector_db as db
import ingest_data as ingest


@pytest.fixture
def memory(monkeypatch):
    client = QdrantClient(':memory:')
    cfg = Settings(_env_file=None, embedding_dimensions=2)
    monkeypatch.setattr(db, '_client', client)
    monkeypatch.setattr(db, 'get_settings', lambda: cfg)
    monkeypatch.setattr(ingest, 'get_settings', lambda: cfg)
    db.create_collection('documents', 2)
    db.upsert_points('documents', [models.PointStruct(id=1, vector=[1., 0.], payload={'text': 'old KB'})])
    yield client
    client.close()


def policy(tmp_path):
    p = tmp_path / 'bank_policy.txt'
    p.write_text('1. Cek Saldo\nSaldo bisa dicek melalui aplikasi.', encoding='utf-8')
    return p


def test_success_activates_staging_and_preserves_old(memory, tmp_path):
    count = ingest.ingest_document(policy(tmp_path), embedding_fn=lambda _: [1., 0.])
    assert count == 1
    assert memory.count('documents').count == 1
    assert memory.count('documents__active').count == 1
    assert db.search_documents([1., 0.])[0]['intent_ids'] == ['balance']
    assert db.search_documents([1., 0.])[0]['kb_version']


def test_empty_input_does_not_create_or_delete_collections(memory, tmp_path):
    p = tmp_path / 'empty.txt'
    p.write_text(' \n ', encoding='utf-8')
    with pytest.raises(ValueError):
        ingest.ingest_document(p, embedding_fn=Mock())
    assert len(memory.get_collections().collections) == 1
    assert memory.count('documents').count == 1


def test_embedding_failure_leaves_old_collection_untouched(memory, tmp_path):
    embed = Mock(side_effect=RuntimeError('embedding unavailable'))
    with pytest.raises(RuntimeError):
        ingest.ingest_document(policy(tmp_path), embedding_fn=embed)
    assert len(memory.get_collections().collections) == 1
    assert memory.count('documents').count == 1


def test_upsert_failure_never_activates_or_deletes_old(memory, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, 'upsert_points', Mock(side_effect=RuntimeError('upsert failure')))
    with pytest.raises(RuntimeError):
        ingest.ingest_document(policy(tmp_path), embedding_fn=lambda _: [1., 0.])
    assert memory.count('documents').count == 1
    assert memory.get_aliases().aliases == []


def test_bad_embedding_dimensions_precede_db_changes(memory, tmp_path):
    with pytest.raises(ValueError):
        ingest.ingest_document(policy(tmp_path), embedding_fn=lambda _: [1.])
    assert len(memory.get_collections().collections) == 1


@pytest.mark.parametrize('vector', [[0., 0.], [True, False], ['1', '0'], None])
def test_invalid_vector_precedes_db_changes(memory, tmp_path, vector):
    with pytest.raises(ValueError):
        ingest.ingest_document(policy(tmp_path), embedding_fn=lambda _: vector)
    assert len(memory.get_collections().collections) == 1


def test_cli_missing_file_fails_without_exposing_details(tmp_path, capsys):
    assert ingest.main([str(tmp_path / 'missing.txt')]) == 1
    assert 'Ingestion failed' in capsys.readouterr().err


def test_ingestion_records_sumopod_provider_and_preserves_previous_index(memory,tmp_path,monkeypatch):
    cfg=Settings(_env_file=None,embedding_dimensions=2)
    monkeypatch.setattr(db,'get_settings',lambda:cfg)
    monkeypatch.setattr(ingest,'get_settings',lambda:cfg)
    ingest.ingest_document(policy(tmp_path),embedding_fn=lambda _:[1.,0.])
    snapshot=db.get_collection_snapshot()
    assert snapshot['embedding_base_url']==cfg.provider_address('embedding')
    assert snapshot['embedding_provider_verification']=='explicit-metadata-and-payload'
    physical=snapshot['physical_collection']
    info=memory.get_collection(physical)
    assert info.config.metadata['embedding_base_url']==cfg.provider_address('embedding')
    records,_=memory.scroll(physical,with_payload=True)
    assert all(p.payload['embedding_base_url']==cfg.provider_address('embedding') for p in records)
    assert memory.count('documents').count==1
    assert db.search_documents([1.,0.])[0]['embedding_base_url']==cfg.provider_address('embedding')
