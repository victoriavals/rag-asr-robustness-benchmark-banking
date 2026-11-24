"""Stage and validate a knowledge base before atomically activating its alias."""
import argparse
import math
import sys
import uuid
from numbers import Real
from pathlib import Path
from collections.abc import Callable
from qdrant_client.http import models
from src.config import PROJECT_ROOT, get_settings
from src.knowledge_base import prepare_chunks
from src.sumopod_services import get_embedding
from src.vector_db import activate_collection, create_collection, get_client, upsert_points


def load_text_file(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding='utf-8-sig')


def chunk_text(text: str, separator: str = '\n\n') -> list[str]:
    if not separator:
        raise ValueError('separator cannot be empty')
    return [chunk.strip() for chunk in text.split(separator) if chunk.strip()]


def create_points(chunks: list[dict] | list[str], embedding_fn: Callable | None = None) -> list[models.PointStruct]:
    cfg = get_settings()
    embed = embedding_fn or get_embedding
    points = []
    if not chunks:
        raise ValueError('Knowledge base contains no usable sections')
    if isinstance(chunks[0], str):
        chunks = prepare_chunks('\n\n'.join(chunks))
    for chunk in chunks:
        vector = embed(chunk['text'])
        if (not isinstance(vector, (list, tuple)) or len(vector) != cfg.embedding_dimensions
                or not all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
                or not any(vector)):
            raise ValueError('Embedding dimension or values do not match the configured schema')
        payload = {key: value for key, value in chunk.items() if key != 'id'}
        payload.update(embedding_model=cfg.embedding_model, embedding_dimensions=cfg.embedding_dimensions,
                       embedding_base_url=cfg.provider_address('embedding'))
        points.append(models.PointStruct(id=chunk['id'], vector=vector, payload=payload))
    return points


def ingest_document(file_path: str | Path, *, embedding_fn: Callable | None = None,
                    collection_name: str | None = None) -> int:
    cfg = get_settings()
    path = Path(file_path)
    text = load_text_file(path)
    chunks = prepare_chunks(text, source_name=path.name)
    # Finish all embedding work before creating or modifying database state.
    points = create_points(chunks, embedding_fn=embedding_fn)
    logical = collection_name or cfg.qdrant_collection
    kb_version = chunks[0]['kb_version']
    staged = f'{logical}__{kb_version[:12]}_{uuid.uuid4().hex[:8]}'
    create_collection(staged, vector_size=cfg.embedding_dimensions,
        metadata={'kb_version': kb_version, 'embedding_model': cfg.embedding_model,
                  'embedding_base_url': cfg.provider_address('embedding'),
                  'embedding_dimensions': cfg.embedding_dimensions,
                  'source': path.name, 'chunk_count': len(points)})
    # Failed staging collections are retained for inspection. Never remove an
    # active collection after an ambiguous server/alias-update failure.
    upsert_points(staged, points)
    actual = get_client().count(collection_name=staged, exact=True).count
    if actual != len(points):
        raise RuntimeError('Staged point count differs from the prepared knowledge base')
    activate_collection(logical, staged)
    print(f'Activated {len(points)} sections through {logical}__active; previous collections retained.')
    return len(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?', default=str(PROJECT_ROOT / 'data' / 'bank_policy.txt'))
    args = parser.parse_args(argv)
    try:
        ingest_document(args.file)
    except (OSError, ValueError, RuntimeError):
        print('Ingestion failed. Check the input, service configuration, and staging collection. Active data was not deleted.', file=sys.stderr)
        return 1
    except Exception:
        print('Ingestion failed during a service request. Check credentials and service availability; existing collections were retained.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
