"""Deterministic knowledge chunks and explicit banking chapter labels."""
import hashlib
import re
import uuid
from typing import TypedDict

GOLD_MAPPING_VERSION = 'bank-policy-chapters-v1'

INTENT_NAMES = ["abroad", "address", "app_error", "atm_limit", "balance",
                "business_loan", "card_issues", "cash_deposit", "direct_debit",
                "freeze", "high_value_payment", "joint_account", "latest_transactions", "pay_bill"]


class KnowledgeChunk(TypedDict):
    id: str
    text: str
    chunk_index: int
    intent_ids: list[str]
    kb_version: str
    source: str


_CHAPTER_INTENTS = {
    "cek saldo": ["balance"],
    "mutasi rekening (latest transactions)": ["latest_transactions"],
    "pembayaran tagihan (pay bill)": ["pay_bill"],
    "transfer & direct debit": ["direct_debit", "high_value_payment"],
    "masalah kartu atm (card issues)": ["card_issues"],
    "limit penarikan & setor tunai": ["atm_limit", "cash_deposit"],
    "masalah aplikasi (app error)": ["app_error"],
    "akun dibekukan (freeze account)": ["freeze"],
    "pinjaman bisnis (business loan)": ["business_loan"],
    "akun gabungan (joint account)": ["joint_account"],
    "ganti alamat (address change)": ["address"],
    "penggunaan di luar negeri (abroad)": ["abroad"],
}


def prepare_chunks(text: str, source_name: str = "bank_policy.txt") -> list[KnowledgeChunk]:
    """Split paragraphs; label exact chapter headings, leaving generic text unlabeled."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Knowledge document cannot be empty")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    version = hashlib.sha256(normalized.encode()).hexdigest()
    chunks = []
    for paragraph in re.split(r"\n\s*\n", normalized):
        lines = [s.strip() for s in paragraph.splitlines() if s.strip()]
        content = [s for s in lines if not (s.startswith("===") or s.startswith("[KATEGORI:"))]
        if not content:
            continue
        heading = re.sub(r"^\d+\.\s*", "", content[0]).casefold()
        chunk_text = "\n".join(lines)
        index = len(chunks)
        identifier = uuid.uuid5(uuid.NAMESPACE_URL, f"{source_name}:{version}:{index}:{chunk_text}")
        chunks.append(KnowledgeChunk(id=str(identifier),text=chunk_text,chunk_index=index,
                      intent_ids=list(_CHAPTER_INTENTS.get(heading,[])),kb_version=version,source=source_name))
    if not chunks:
        raise ValueError("Knowledge document has no content chunks")
    return chunks
