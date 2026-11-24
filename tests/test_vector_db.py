from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from src.config import Settings
from src import vector_db as db


@pytest.fixture
def memory_db(monkeypatch):
    client = QdrantClient(':memory:')
    monkeypatch.setattr(db, '_client', client)
    monkeypatch.setattr(db, 'get_settings', lambda: Settings(_env_file=None, embedding_dimensions=2))
    yield client
    client.close()


def point(number=1, *, labels=None, text='Cek saldo', model='text-embedding-3-small'):
    return models.PointStruct(id=number, vector=[1., 0.], payload={
        'text': text, 'intent_ids': labels or ['balance'], 'kb_version': 'v1',
        'embedding_model': model, 'embedding_base_url':'https://ai.sumopod.com/v1',
        'source': 'bank_policy.txt', 'chunk_index': 0})


def mock_remote_client(query):
    """Provide real vector schema/provenance rather than bypassing validation."""
    payload = {'embedding_model': 'text-embedding-3-small', 'kb_version': 'v1',
               'embedding_base_url':'https://ai.sumopod.com/v1'}
    info = SimpleNamespace(points_count=1, config=SimpleNamespace(
        params=SimpleNamespace(vectors=models.VectorParams(size=2, distance=models.Distance.COSINE)),
        metadata=payload))
    return SimpleNamespace(query_points=query,
        get_collection=lambda *args, **kwargs: info,
        scroll=lambda **kwargs: ([SimpleNamespace(payload=payload)], None),
        get_aliases=lambda: SimpleNamespace(aliases=[]))


def test_alias_switch_keeps_legacy_and_previous_collections(memory_db):
    for name in ('documents', 'staged1', 'staged2'):
        db.create_collection(name, 2)
        db.upsert_points(name, [point()])
    db.activate_collection('documents', 'staged1')
    first_fingerprint = db.get_collection_fingerprint()
    db.activate_collection('documents', 'staged2')
    assert db.resolve_collection() == 'documents__active'
    assert db.get_collection_fingerprint() != first_fingerprint
    assert memory_db.count('documents').count == 1
    assert memory_db.count('staged1').count == 1
    docs = db.search_documents([1., 0.])
    assert docs[0]['intent_ids'] == ['balance']
    assert docs[0]['id'] == '1'
    assert docs[0]['score'] == 1
    assert db.collection_health()['status'] == 'ready'


def test_health_missing_empty_and_incompatible(memory_db):
    assert db.collection_health()['status'] == 'missing'
    db.create_collection('documents', 2)
    assert db.collection_health()['status'] == 'empty'
    db.upsert_points('documents', [point(model='other-model')])
    assert db.collection_health()['status'] == 'incompatible'


def test_recreate_and_empty_upsert_refuse_data_loss(memory_db):
    db.create_collection('documents', 2)
    db.upsert_points('documents', [point()])
    with pytest.raises(ValueError):
        db.recreate_collection('documents', 2)
    with pytest.raises(ValueError):
        db.upsert_points('documents', [])
    assert memory_db.count('documents').count == 1


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_collection_check_preserves_non_404_failures(monkeypatch, status):
    exc = UnexpectedResponse(status, 'error', b'private response', httpx.Headers())
    monkeypatch.setattr(db, '_client', SimpleNamespace(get_collection=Mock(side_effect=exc)))
    with pytest.raises(UnexpectedResponse) as caught:
        db.collection_exists('documents')
    assert caught.value is exc


def test_missing_remote_collection_is_false(monkeypatch):
    exc = UnexpectedResponse(404, 'not found', b'', httpx.Headers())
    monkeypatch.setattr(db, '_client', SimpleNamespace(get_collection=Mock(side_effect=exc)))
    assert db.collection_exists('documents') is False


@pytest.mark.parametrize('exc', [UnexpectedResponse(500, 'server', b'', httpx.Headers()),
                               ResponseHandlingException(RuntimeError('transport'))])
def test_search_retries_transient_failures(monkeypatch, exc):
    query = Mock(side_effect=[exc, SimpleNamespace(points=[])])
    monkeypatch.setattr(db, '_client', mock_remote_client(query))
    monkeypatch.setattr(db, 'get_settings', lambda: Settings(_env_file=None, embedding_dimensions=2))
    sleep = Mock()
    monkeypatch.setattr(db.time, 'sleep', sleep)
    assert db.search_documents([1., 0.]) == []
    assert query.call_count == 2
    sleep.assert_called_once_with(1)


def test_search_filters_invalid_payload_and_preserves_provenance(monkeypatch):
    origin={'embedding_base_url':'https://ai.sumopod.com/v1'}
    points = [SimpleNamespace(id='a', score=.9, payload={'text': None,**origin}),
              SimpleNamespace(id='b', score=.8, payload={'text': 42,**origin}),
              SimpleNamespace(id='c', score=.7, payload={'text': 'valid', 'intent_ids': 'bad',**origin})]
    monkeypatch.setattr(db, '_client', mock_remote_client(lambda **kwargs: SimpleNamespace(points=points)))
    monkeypatch.setattr(db, 'get_settings', lambda: Settings(_env_file=None, embedding_dimensions=2))
    docs = db.search_documents([1., 0.])
    assert len(docs) == 1 and docs[0]['id'] == 'c' and docs[0]['intent_ids'] == []


@pytest.mark.parametrize('kwargs', [{'query_vector': []}, {'query_vector': [float('nan')]},
    {'query_vector': [1], 'max_retries': 0}, {'query_vector': [1], 'top_k': 0},
    {'query_vector': [0., 0.]}, {'query_vector': [True, False]}])
def test_invalid_search_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        db.search_documents(**kwargs)


@pytest.mark.parametrize('kind', ['model', 'dimension', 'distance', 'named_vectors',
                                 'payload_model_conflict', 'payload_version_conflict'])
def test_incompatible_schema_refuses_query_fingerprint_and_health(memory_db, monkeypatch, kind):
    size = 3 if kind == 'dimension' else 2
    distance = models.Distance.DOT if kind == 'distance' else models.Distance.COSINE
    vectors = models.VectorParams(size=size, distance=distance)
    metadata = {'embedding_model': 'text-embedding-3-small', 'kb_version': 'v1'}
    if kind == 'named_vectors':
        vectors = {'dense': vectors}
    memory_db.create_collection('documents', vectors_config=vectors, metadata=metadata)
    payload = {'text': 'balance', 'embedding_model': 'text-embedding-3-small', 'kb_version': 'v1',
               'embedding_base_url':'https://ai.sumopod.com/v1'}
    if kind in ('model', 'payload_model_conflict'):
        payload['embedding_model'] = 'different-model'
    if kind == 'payload_version_conflict':
        payload['kb_version'] = 'v2'
    vector = [1.] + [0.] * (size - 1)
    if kind == 'named_vectors':
        vector = {'dense': vector}
    memory_db.upsert('documents', points=[models.PointStruct(id=1, vector=vector, payload=payload)])
    query = Mock(wraps=memory_db.query_points)
    monkeypatch.setattr(memory_db, 'query_points', query)
    with pytest.raises(ValueError):
        db.search_documents([1., 0.])
    with pytest.raises(ValueError):
        db.get_collection_fingerprint()
    query.assert_not_called()
    assert db.collection_health()['status'] == 'incompatible'


def stage(memory_db, name, text='balance'):
    db.create_collection(name, 2)
    db.upsert_points(name, [point(text=text)])


def test_preexisting_configured_alias_is_preserved_on_migration(memory_db):
    stage(memory_db, 'legacy')
    stage(memory_db, 'new')
    memory_db.update_collection_aliases(change_aliases_operations=[models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name='legacy', alias_name='documents'))])
    assert db.resolve_collection() == 'documents'
    assert db.get_collection_snapshot()['physical_collection'] == 'legacy'
    db.activate_collection('documents', 'new')
    aliases = {alias.alias_name: alias.collection_name for alias in memory_db.get_aliases().aliases}
    assert aliases == {'documents': 'legacy', 'documents__active': 'new'}
    assert memory_db.count('legacy').count == 1
    assert db.get_collection_snapshot()['physical_collection'] == 'new'


@pytest.mark.parametrize('target', ['missing', 'empty', 'incompatible'])
def test_invalid_staged_target_cannot_remove_current_alias(memory_db, target):
    stage(memory_db, 'old')
    db.activate_collection('documents', 'old')
    if target == 'empty':
        db.create_collection(target, 2)
    elif target == 'incompatible':
        db.create_collection(target, 2)
        db.upsert_points(target, [point(model='different-model')])
    with pytest.raises(ValueError):
        db.activate_collection('documents', target)
    aliases = {alias.alias_name: alias.collection_name for alias in memory_db.get_aliases().aliases}
    assert aliases['documents__active'] == 'old'
    assert db.search_documents([1., 0.])[0]['text'] == 'balance'


def test_ambiguous_alias_failure_does_not_roll_back_committed_replacement(memory_db, monkeypatch):
    stage(memory_db, 'old')
    stage(memory_db, 'new')
    db.activate_collection('documents', 'old')
    update = memory_db.update_collection_aliases

    def committed_but_connection_lost(**kwargs):
        update(**kwargs)
        raise ResponseHandlingException(RuntimeError('connection lost after commit'))

    monkeypatch.setattr(memory_db, 'update_collection_aliases', committed_but_connection_lost)
    with pytest.raises(ResponseHandlingException):
        db.activate_collection('documents', 'new')
    assert db.get_collection_snapshot()['physical_collection'] == 'new'
    assert memory_db.count('old').count == 1


def test_known_local_partial_alias_failure_restores_only_missing_alias(memory_db, monkeypatch):
    stage(memory_db, 'old')
    stage(memory_db, 'new')
    db.activate_collection('documents', 'old')
    update = memory_db.update_collection_aliases
    calls = []

    def partially_apply_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            update(change_aliases_operations=[kwargs['change_aliases_operations'][0]])
            raise ValueError('known local partial failure')
        return update(**kwargs)

    monkeypatch.setattr(memory_db, 'update_collection_aliases', partially_apply_once)
    with pytest.raises(ValueError):
        db.activate_collection('documents', 'new')
    assert db.get_collection_snapshot()['physical_collection'] == 'old'
    assert len(calls) == 2


def test_known_failure_does_not_replace_an_observed_new_alias(memory_db, monkeypatch):
    stage(memory_db, 'old')
    stage(memory_db, 'new')
    db.activate_collection('documents', 'old')
    update = memory_db.update_collection_aliases

    def applied_before_validation_error(**kwargs):
        update(**kwargs)
        raise ValueError('local failure after alias installation')

    monkeypatch.setattr(memory_db, 'update_collection_aliases', applied_before_validation_error)
    with pytest.raises(ValueError):
        db.activate_collection('documents', 'new')
    assert db.get_collection_snapshot()['physical_collection'] == 'new'


def test_snapshot_pins_physical_collection_across_alias_changes(memory_db):
    stage(memory_db, 'first', 'old policy')
    stage(memory_db, 'second', 'new policy')
    db.activate_collection('documents', 'first')
    snapshot = db.get_collection_snapshot()
    assert snapshot['collection_name'] == snapshot['physical_collection'] == 'first'
    assert snapshot['logical_collection'] == 'documents'
    assert snapshot['dimensions'] == 2
    assert snapshot['embedding_model'] == 'text-embedding-3-small'
    db.activate_collection('documents', 'second')
    assert db.search_documents([1., 0.], collection_name=snapshot['physical_collection'])[0]['text'] == 'old policy'
    assert db.search_documents([1., 0.])[0]['text'] == 'new policy'
    assert db.get_collection_fingerprint('first') == snapshot['fingerprint']
    assert db.collection_health()['kb_fingerprint'] == db.get_collection_fingerprint()


def test_fingerprint_detects_payload_change_without_count_or_metadata_change(memory_db):
    stage(memory_db, 'documents', 'initial policy')
    before = db.get_collection_fingerprint()
    db.upsert_points('documents', [point(text='revised policy')])
    assert db.get_collection_fingerprint() != before


def test_mixed_payload_provenance_is_rejected_even_when_first_point_matches(memory_db):
    db.create_collection('documents', 2, metadata={
        'embedding_model': 'text-embedding-3-small', 'kb_version': 'v1'})
    db.upsert_points('documents', [point(1), point(2, model='other-model')])
    with pytest.raises(ValueError):
        db.get_collection_snapshot()
    with pytest.raises(ValueError):
        db.search_documents([1., 0.])
    assert db.collection_health()['status'] == 'incompatible'


def test_snapshot_hashes_every_page_and_is_stable(memory_db):
    db.create_collection('documents', 2)
    db.upsert_points('documents', [point(index) for index in range(130)])
    snapshot = db.get_collection_snapshot()
    assert snapshot['points_count'] == 130
    assert snapshot['fingerprint'] == db.get_collection_snapshot()['fingerprint']


def test_unknown_embedding_provenance_is_rejected_without_assumptions(memory_db,monkeypatch):
    db.create_collection('documents',2)
    p=point(); p.payload.pop('embedding_base_url')
    db.upsert_points('documents',[p])
    query=Mock(wraps=memory_db.query_points)
    monkeypatch.setattr(memory_db,'query_points',query)
    with pytest.raises(db.ProviderProvenanceError,match='re-ingest'):
        db.search_documents([1.,0.])
    query.assert_not_called()
    health=db.collection_health()
    assert health['status']=='incompatible' and health['reason']=='embedding_provider'
    assert 'Re-ingest' in health['message']
    assert memory_db.count('documents').count==1


def test_collection_metadata_cannot_bless_legacy_unknown_points_on_sumopod(memory_db,monkeypatch):
    cfg=Settings(_env_file=None,embedding_dimensions=2)
    monkeypatch.setattr(db,'get_settings',lambda:cfg)
    db.create_collection('documents',2,metadata={'embedding_base_url':cfg.provider_address('embedding')})
    p=point(); p.payload.pop('embedding_base_url')
    db.upsert_points('documents',[p])
    with pytest.raises(db.ProviderProvenanceError): db.get_collection_snapshot()


def test_explicit_provider_mismatch_same_model_and_dimensions_blocks_query(memory_db,monkeypatch):
    db.create_collection('documents',2)
    p=point(); p.payload['embedding_base_url']='https://different-provider.example.invalid/v1'
    db.upsert_points('documents',[p])
    query=Mock(wraps=memory_db.query_points)
    monkeypatch.setattr(memory_db,'query_points',query)
    with pytest.raises(db.ProviderProvenanceError): db.search_documents([1.,0.])
    query.assert_not_called()


@pytest.mark.parametrize('url',['https://user:private-password@example.invalid/v1',
    'https://example.invalid/v1?api_key=private-password','',42])
def test_invalid_stored_provider_never_exposes_secret(memory_db,url):
    db.create_collection('documents',2)
    p=point(); p.payload['embedding_base_url']=url
    db.upsert_points('documents',[p])
    with pytest.raises(db.ProviderProvenanceError) as caught: db.get_collection_snapshot()
    assert 'private-password' not in str(caught.value)
    assert 'private-password' not in db.collection_health()['message']


def test_explicit_point_provenance_can_supply_missing_collection_metadata(memory_db,monkeypatch):
    cfg=Settings(_env_file=None,embedding_dimensions=2)
    monkeypatch.setattr(db,'get_settings',lambda:cfg)
    db.create_collection('documents',2)
    p=point(); p.payload['embedding_base_url']=cfg.provider_address('embedding')+'/'
    db.upsert_points('documents',[p])
    snapshot=db.get_collection_snapshot()
    assert snapshot['embedding_base_url']==cfg.provider_address('embedding')
    assert snapshot['embedding_provider_verification']=='explicit-payload'
