from pathlib import Path
import uuid
import pytest
from src.knowledge_base import prepare_chunks, INTENT_NAMES


def test_real_policy_has_stable_ids_and_all_explicit_intents():
    path=Path(__file__).resolve().parents[1]/'data/bank_policy.txt'
    text=path.read_text(encoding='utf-8')
    chunks=prepare_chunks(text)
    assert len(chunks)==12
    assert chunks==prepare_chunks(text.replace('\n','\r\n'))
    assert set().union(*(set(c['intent_ids']) for c in chunks))==set(INTENT_NAMES)
    assert {tuple(c['intent_ids']) for c in chunks} >= {('atm_limit','cash_deposit'),('direct_debit','high_value_payment')}
    assert len({c['id'] for c in chunks})==12
    for c in chunks: assert uuid.UUID(c['id']).version==5
    changed=prepare_chunks(text+'\n\nGeneric extra paragraph')
    assert changed[0]['kb_version'] != chunks[0]['kb_version']
    assert changed[0]['id'] != chunks[0]['id']


def test_generic_content_is_not_keyword_labeled():
    chunk=prepare_chunks('Pinjaman includes pin. Tabungan includes bunga.')[0]
    assert chunk['intent_ids']==[]


@pytest.mark.parametrize('text',['','  ','=== TITLE ONLY ===\n\n[KATEGORI: UMUM]'])
def test_empty_content_rejected(text):
    with pytest.raises(ValueError): prepare_chunks(text)
