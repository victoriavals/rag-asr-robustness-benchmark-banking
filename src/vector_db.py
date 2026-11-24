"""Qdrant adapters with safe alias activation and auditable retrieval."""
import hashlib
import json
import math
import time
from numbers import Real
from typing import TypedDict
from urllib.parse import urlsplit
from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from src.config import SUMOPOD_BASE_URL, get_settings

EMBEDDING_PROVENANCE_VERSION = 'sumopod-embedding-endpoint-v2'


class ProviderProvenanceError(ValueError):
    """The stored vector space cannot be verified against the active provider."""


def _provider_address(value: str) -> str:
    """Canonicalize public provider URLs without accepting credentials or queries."""
    try:
        parsed=urlsplit(value)
        if (not isinstance(value,str) or parsed.scheme not in {'http','https'} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None or parsed.query
                or parsed.fragment or any(c.isspace() for c in value)):
            raise ValueError
        host=parsed.hostname.lower()
        host=f'[{host}]' if ':' in host else host
        port=parsed.port
        if port is not None and port != (443 if parsed.scheme=='https' else 80):
            host+=f':{port}'
        return f'{parsed.scheme.lower()}://{host}{parsed.path.rstrip("/")}'
    except (AttributeError,TypeError,ValueError):
        raise ProviderProvenanceError('Knowledge-base embedding provider provenance is invalid; re-ingest it') from None

class RetrievedDocument(TypedDict):
    id: str
    text: str
    score: float
    intent_ids: list[str]
    kb_version: str
    source: str
    chunk_index: int
    embedding_base_url: str

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        cfg = get_settings()
        _client = QdrantClient(url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None,
            timeout=cfg.request_timeout_seconds, check_compatibility=False)
    return _client

def collection_exists(collection_name: str) -> bool:
    try:
        get_client().get_collection(collection_name=collection_name)
        return True
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return False
        raise
    except ValueError as exc:
        if str(exc).startswith('Collection ') and 'not found' in str(exc):
            return False
        raise

def active_alias(collection_name: str) -> str:
    return f'{collection_name}__active'

def resolve_collection(collection_name: str | None = None) -> str:
    return _resolve_target(collection_name)[1]


def _resolve_target(collection_name: str | None = None) -> tuple[str, str, str]:
    name = collection_name or get_settings().qdrant_collection
    aliases = {item.alias_name: item.collection_name for item in get_client().get_aliases().aliases}
    selected = active_alias(name) if active_alias(name) in aliases else name
    return name, selected, aliases.get(selected, selected)

def create_collection(collection_name: str, vector_size: int = 1536,
        distance: models.Distance = models.Distance.COSINE, metadata: dict | None = None) -> None:
    if vector_size <= 0:
        raise ValueError('vector_size must be positive')
    get_client().create_collection(collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance), metadata=metadata)

def delete_collection(collection_name: str) -> None:
    """Explicit admin operation; ingestion never deletes active data."""
    get_client().delete_collection(collection_name=collection_name)

def recreate_collection(collection_name: str, vector_size: int = 1536,
        distance: models.Distance = models.Distance.COSINE) -> None:
    if collection_exists(collection_name):
        raise ValueError('Collection already exists; ingest into a staging collection instead')
    create_collection(collection_name, vector_size, distance)

def activate_collection(logical_name: str, staged_name: str) -> None:
    client = get_client()
    # A missing/invalid target must fail before removing an existing alias. The
    # local SDK does not roll back partially applied alias operations itself.
    info = _validate_collection_schema(staged_name)
    if not info.points_count:
        raise ValueError('Cannot activate an empty knowledge base')
    _collection_snapshot(logical_name, staged_name, info)
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    alias = active_alias(logical_name)
    operations = []
    if alias in aliases:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    operations.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=staged_name, alias_name=alias)))
    try:
        client.update_collection_aliases(change_aliases_operations=operations)
    except ValueError:
        # Recover only a known local failure, only if the alias is absent. Never
        # remove a replacement alias or roll back an ambiguous network outcome.
        if isinstance(getattr(client, '_client', None), QdrantLocal) and alias in aliases:
            try:
                current = {item.alias_name for item in client.get_aliases().aliases}
                if alias not in current:
                    client.update_collection_aliases(change_aliases_operations=[
                        models.CreateAliasOperation(create_alias=models.CreateAlias(
                            collection_name=aliases[alias], alias_name=alias))])
            except Exception:
                pass
        raise

def upsert_points(collection_name: str, points: list[models.PointStruct]) -> None:
    if not points:
        raise ValueError('Cannot upsert an empty knowledge base')
    get_client().upsert(collection_name=collection_name, points=points, wait=True)

def _first_payload(name: str) -> dict:
    records, _ = get_client().scroll(collection_name=name, limit=1, with_payload=True, with_vectors=False)
    return (records[0].payload or {}) if records else {}

def _payload_identity(payload: dict, expected: dict, *, is_point: bool = False) -> dict:
    """Reject conflicting provenance and require explicit point origin."""
    cfg = get_settings()
    identity = dict(expected)
    for key in ('embedding_model', 'kb_version'):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError('Knowledge base provenance is invalid; re-ingest it')
        if value and identity.get(key) and value != identity[key]:
            raise ValueError('Knowledge base contains conflicting provenance; re-ingest it')
        if value:
            identity[key] = value
    if identity.get('embedding_model') and identity['embedding_model'] != cfg.embedding_model:
        raise ValueError('Embedding models differ; re-ingest the knowledge base')
    dimension = payload.get('embedding_dimensions')
    if dimension is not None and (isinstance(dimension, bool) or dimension != cfg.embedding_dimensions):
        raise ValueError('Knowledge base embedding dimensions conflict; re-ingest it')
    stored=payload.get('embedding_base_url')
    address=_provider_address(stored) if stored is not None else None
    active=_provider_address(cfg.provider_address('embedding'))
    if active != SUMOPOD_BASE_URL:
        raise ProviderProvenanceError('Only the configured SumoPod embedding provider is supported')
    if address is None and is_point:
        raise ProviderProvenanceError('Knowledge-base embedding provider is unknown; re-ingest it with SumoPod')
    if address:
        if address != active or (identity.get('embedding_base_url') and address != identity['embedding_base_url']):
            raise ProviderProvenanceError('Knowledge-base embedding provider differs from the current route; re-ingest it')
        identity['embedding_base_url']=address
    return identity


def _inspect_collection(name: str, info=None, payload: dict | None = None):
    cfg = get_settings()
    info = info or get_client().get_collection(name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, models.VectorParams) or vectors.size != cfg.embedding_dimensions:
        raise ValueError('Embedding dimensions differ; re-ingest the knowledge base')
    if vectors.distance != models.Distance.COSINE:
        raise ValueError('The knowledge base must use cosine distance')
    metadata = getattr(info.config, 'metadata', None) or {}
    identity = _payload_identity(metadata, {})
    if payload is None:
        payload = _first_payload(name) if info.points_count else {}
    identity = _payload_identity(payload, identity, is_point=bool(info.points_count))
    return info, identity


def _validate_collection_schema(name: str, info=None):
    return _inspect_collection(name, info)[0]

def collection_health() -> dict:
    cfg = get_settings()
    name = cfg.qdrant_collection
    base = {'collection': name, 'points_count': 0, 'vector_size': None, 'labelled': False,
            'kb_version': None, 'embedding_model': None, 'embedding_base_url': None,
            'embedding_provider_verification': None}
    try:
        logical, selected, physical = _resolve_target(name)
        base['collection'] = selected
        if not collection_exists(physical):
            return {**base, 'status': 'missing', 'message': 'Knowledge base has not been ingested.'}
        info = get_client().get_collection(physical)
        vectors = info.config.params.vectors
        dimension = vectors.size if isinstance(vectors, models.VectorParams) else None
        base.update(points_count=info.points_count or 0, vector_size=dimension)
        _validate_collection_schema(physical, info)
        if not base['points_count']:
            return {**base, 'status': 'empty', 'message': 'Knowledge base is empty.'}
        snapshot = _collection_snapshot(logical, physical, info)
        base.update(labelled=snapshot['labelled'], kb_version=snapshot['kb_version'],
            embedding_model=snapshot['embedding_model'], kb_fingerprint=snapshot['fingerprint'],
            physical_collection=physical, embedding_base_url=snapshot['embedding_base_url'],
            embedding_provider_verification=snapshot['embedding_provider_verification'])
        return {**base, 'status': 'ready', 'message': 'Knowledge base is ready.'}
    except ProviderProvenanceError:
        return {**base, 'status': 'incompatible', 'reason': 'embedding_provider',
            'message': 'Knowledge-base embedding provider is missing or differs from the current route. Re-ingest it; existing collections are retained.'}
    except ValueError:
        return {**base, 'status': 'incompatible', 'message': 'Knowledge base schema or provenance differs; re-ingest it.'}
    except Exception:
        return {**base, 'status': 'error', 'message': 'Cannot read Qdrant. Check the URL, credentials, and server.'}

def _collection_snapshot(logical_name: str, physical_name: str, info) -> dict:
    client = get_client()
    info, identity = _inspect_collection(physical_name, info)
    if not info.points_count:
        raise ValueError('Knowledge base is empty')
    metadata = getattr(info.config, 'metadata', None) or {}
    vectors = info.config.params.vectors
    records = []
    offset = None
    labelled = True
    while True:
        page, next_offset = client.scroll(collection_name=physical_name, limit=128,
            offset=offset, with_payload=True, with_vectors=True)
        for point in page:
            payload = point.payload or {}
            identity = _payload_identity(payload, identity, is_point=True)
            labels = payload.get('intent_ids')
            labelled = labelled and isinstance(labels, list) and bool(labels) and all(
                isinstance(label, str) and bool(label.strip()) for label in labels)
            records.append({'id': str(point.id), 'payload': payload, 'vector': point.vector})
        if next_offset is None:
            break
        if next_offset == offset:
            raise ValueError('Knowledge base pagination did not advance')
        offset = next_offset
    if len(records) != info.points_count:
        raise ValueError('Knowledge base changed while its snapshot was being read; retry')
    verification=('explicit-metadata-and-payload' if metadata.get('embedding_base_url') else 'explicit-payload')
    record = {'physical_collection': physical_name, 'points': sorted(records, key=lambda x: x['id']),
              'schema': vectors.model_dump(), 'metadata': metadata,
              'embedding_base_url': identity.get('embedding_base_url'),
              'embedding_provider_verification': verification,
              'embedding_provenance_version': EMBEDDING_PROVENANCE_VERSION}
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return {'fingerprint': fingerprint, 'collection_name': physical_name,
            'physical_collection': physical_name, 'logical_collection': logical_name,
            'kb_version': identity.get('kb_version'), 'embedding_model': identity.get('embedding_model'),
            'embedding_base_url': identity.get('embedding_base_url'),
            'embedding_provider_verification': verification,
            'embedding_provenance_version': EMBEDDING_PROVENANCE_VERSION,
            'dimensions': vectors.size, 'labelled': labelled, 'points_count': len(records)}


def get_collection_snapshot(collection_name: str | None = None) -> dict:
    """Freeze a physical target and fingerprint its schema, payloads, and vectors."""
    logical, _, physical = _resolve_target(collection_name)
    return _collection_snapshot(logical, physical, get_client().get_collection(physical))


def get_collection_fingerprint(collection_name: str | None = None) -> str:
    return get_collection_snapshot(collection_name)['fingerprint']

def search_documents(query_vector: list[float], top_k: int = 3, collection_name: str | None = None,
        max_retries: int = 3, score_threshold: float | None = None) -> list[RetrievedDocument]:
    if (not isinstance(query_vector, (list, tuple)) or not query_vector
            or not all(not isinstance(x, bool) and isinstance(x, Real) and math.isfinite(x) for x in query_vector)
            or not any(query_vector)):
        raise ValueError('query_vector must contain finite numbers and be nonzero')
    if top_k < 1 or max_retries < 1:
        raise ValueError('top_k and max_retries must be positive')
    if score_threshold is not None and not -1 <= score_threshold <= 1:
        raise ValueError('score_threshold must be between -1 and 1')
    for attempt in range(max_retries):
        try:
            _, _, name = _resolve_target(collection_name)
            _, identity = _inspect_collection(name)
            if len(query_vector) != get_settings().embedding_dimensions:
                raise ValueError('Query vector dimension does not match the configured embedding model')
            results = get_client().query_points(collection_name=name,
                query=query_vector, limit=top_k, score_threshold=score_threshold)
            documents = []
            for point in results.points:
                payload = point.payload or {}
                point_identity=_payload_identity(payload, identity, is_point=True)
                text = payload.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                labels = payload.get('intent_ids', [])
                labels = [x for x in labels if isinstance(x, str)] if isinstance(labels, list) else []
                documents.append({'id': str(point.id), 'text': text, 'score': float(point.score),
                    'intent_ids': labels, 'kb_version': str(payload.get('kb_version', '')),
                    'source': str(payload.get('source', '')), 'chunk_index': payload.get('chunk_index', -1)
                    if isinstance(payload.get('chunk_index', -1), int) else -1,
                    'embedding_base_url': point_identity.get('embedding_base_url','')})
            return documents
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            retryable = isinstance(exc, ResponseHandlingException) or exc.status_code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 4))
    return []

def search_similar_docs(query_vector: list[float], top_k: int = 3,
        collection_name: str | None = None, max_retries: int = 3) -> list[str]:
    """Compatibility adapter for callers that only need document text."""
    return [doc['text'] for doc in search_documents(query_vector, top_k, collection_name, max_retries)]
