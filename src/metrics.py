"""Shared, versioned transcript and embedding metrics."""
import math
import unicodedata
from jiwer import cer, wer

NORMALIZATION_VERSION = "nfkc-casefold-punctuation-v1"
CER_SPACE_POLICY = 'omit-normalized-whitespace'


def normalize_transcript(text: str) -> str:
    """Normalize case/spacing/punctuation, joining apostrophes; keep digits."""
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string")
    chars = []
    for char in unicodedata.normalize("NFKC", text).casefold():
        if char in {"'", "\u2019", "\u2018"}:
            continue
        chars.append(" " if unicodedata.category(char).startswith("P") else char)
    return " ".join("".join(chars).split())


def _wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return float(len(hypothesis.split()))
    return float(wer(reference, hypothesis)) if hypothesis.strip() else 1.0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Normalized utterance WER; may exceed 1 for inserted words."""
    return _wer(normalize_transcript(reference), normalize_transcript(hypothesis))


def calculate_raw_wer(reference: str, hypothesis: str) -> float:
    """Unnormalized WER; empty/empty is 0, empty reference counts insertions."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("Transcripts must be strings")
    return _wer(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Normalized character error rate, omitting all normalized whitespace.

    This supports languages without whitespace word boundaries. Empty reference
    counts inserted characters; empty/empty is zero. May exceed 1.
    """
    reference=normalize_transcript(reference).replace(' ','')
    hypothesis=normalize_transcript(hypothesis).replace(' ','')
    if not reference:
        return float(len(hypothesis))
    return float(cer(reference,hypothesis)) if hypothesis else 1.0


def cosine_similarity_score(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine in [-1, 1]; a zero vector returns 0."""
    if not vec_a or len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must be nonempty and equally sized")
    if not all(math.isfinite(v) for v in [*vec_a, *vec_b]):
        raise ValueError("Embedding values must be finite")
    norm = math.sqrt(sum(v*v for v in vec_a) * sum(v*v for v in vec_b))
    return max(-1.0, min(1.0, sum(a*b for a,b in zip(vec_a,vec_b))/norm)) if norm else 0.0
