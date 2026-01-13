"""
Evaluation Engine for Multi-Accent RAG Analysis.

This module provides functions for evaluating RAG performance across
multiple English locales (en-US, en-GB, en-AU) using the MINDS-14 dataset.
Calculates WER and Retrieval Hit Rate metrics.
"""

from io import BytesIO
from typing import TypedDict

import pandas as pd
from jiwer import wer

from src.minds14_loader import MINDS14Loader, MINDS14Sample, INTENT_NAMES
from src.openai_services import get_embedding, transcribe_audio
from src.vector_db import search_similar_docs


# Map MINDS-14 intent classes to keywords found in bank_policy.txt
# These keywords are used to check if retrieval is relevant to the intent
INTENT_KEYWORDS: dict[str, list[str]] = {
    # Intent 0: abroad - International transactions
    "abroad": [
        "luar negeri", "abroad", "internasional", "international",
        "visa", "mastercard", "transaksi internasional", "overseas",
    ],
    # Intent 1: address - Address change
    "address": [
        "alamat", "address", "domisili", "ubah alamat", "ganti alamat",
        "profil", "data diri", "ktp",
    ],
    # Intent 2: app_error - Application errors
    "app_error": [
        "aplikasi", "app", "error", "tidak bisa dibuka", "update",
        "playstore", "appstore", "koneksi", "internet",
    ],
    # Intent 3: atm_limit - ATM withdrawal limits
    "atm_limit": [
        "limit", "tarik tunai", "penarikan", "atm", "silver", "gold",
        "platinum", "harian", "withdrawal",
    ],
    # Intent 4: balance - Check balance
    "balance": [
        "saldo", "balance", "cek saldo", "informasi saldo", "rekening",
        "dashboard", "sms banking",
    ],
    # Intent 5: business_loan - Business loans
    "business_loan": [
        "pinjaman", "loan", "bisnis", "business", "kredit", "modal kerja",
        "umkm", "plafon", "bunga", "siup", "nib",
    ],
    # Intent 6: card_issues - Card problems
    "card_issues": [
        "kartu", "card", "hilang", "tertelan", "rusak", "blokir",
        "ganti kartu", "call center", "pengaturan kartu",
    ],
    # Intent 7: cash_deposit - Cash deposits
    "cash_deposit": [
        "setor", "deposit", "tunai", "cash", "crm", "setoran",
        "cash recycling", "setor tunai",
    ],
    # Intent 8: direct_debit - Direct debit setup
    "direct_debit": [
        "debit", "auto-debit", "otomatis", "direct debit", "merchant",
        "penarikan otomatis",
    ],
    # Intent 9: freeze - Account freeze/block
    "freeze": [
        "blokir", "freeze", "dibekukan", "terblokir", "pin", "reset pin",
        "lupa pin", "face id", "aktivitas mencurigakan",
    ],
    # Intent 10: high_value_payment - High value transactions
    "high_value_payment": [
        "transfer", "pembayaran", "payment", "bi-fast", "online",
        "biaya admin", "antar bank", "tagihan",
    ],
    # Intent 11: joint_account - Joint accounts
    "joint_account": [
        "gabungan", "joint", "joint account", "rekening gabungan",
        "kedua belah pihak", "npwp", "kantor cabang",
    ],
    # Intent 12: latest_transactions - Transaction history
    "latest_transactions": [
        "mutasi", "transaksi", "riwayat", "history", "statement",
        "e-statement", "latest transactions",
    ],
    # Intent 13: pay_bill - Bill payments
    "pay_bill": [
        "bayar", "tagihan", "bill", "payment", "pembayaran", "pln",
        "pdam", "listrik", "air", "internet", "bukti bayar",
    ],
}


class EvaluationResult(TypedDict):
    """
    Represents evaluation results for a single sample.

    Attributes:
        locale (str): The language locale (e.g., 'en-US').
        sample_id (str): Unique identifier for the sample.
        intent_class (int): The intent class ID.
        intent_name (str): Human-readable intent name.
        ground_truth (str): Original transcription.
        asr_text (str): Whisper-transcribed text.
        wer_score (float): Word Error Rate between ground truth and ASR.
        hit_rate_text (int): 1 if text retrieval matched intent, 0 otherwise.
        hit_rate_voice (int): 1 if voice retrieval matched intent, 0 otherwise.
    """

    locale: str
    sample_id: str
    intent_class: int
    intent_name: str
    ground_truth: str
    asr_text: str
    wer_score: float
    hit_rate_text: int
    hit_rate_voice: int


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis text.

    Uses the jiwer library for accurate WER calculation.

    Params:
        reference (str): The ground truth reference text.
        hypothesis (str): The ASR-generated hypothesis text.

    Returns:
        float: The Word Error Rate (0.0 = perfect, 1.0 = completely wrong).
            Can be greater than 1.0 if there are many insertions.
    """
    if not reference or not reference.strip():
        return 1.0
    if not hypothesis or not hypothesis.strip():
        return 1.0

    # Normalize to lowercase for fair comparison
    reference_clean: str = reference.lower().strip()
    hypothesis_clean: str = hypothesis.lower().strip()

    error_rate: float = wer(reference_clean, hypothesis_clean)
    return error_rate


def check_hit_rate(retrieved_docs: list[str], intent_class: int) -> bool:
    """
    Check if retrieved documents contain keywords related to the intent.

    Verifies that the retrieval is relevant to the query intent by
    checking for keyword matches.

    Params:
        retrieved_docs (list[str]): List of retrieved document texts.
        intent_class (int): The intent class ID (0-13).

    Returns:
        bool: True if any retrieved doc contains intent keywords, False otherwise.
    """
    # Get intent name from class
    if intent_class < 0 or intent_class >= len(INTENT_NAMES):
        return False

    intent_name: str = INTENT_NAMES[intent_class]

    # Get keywords for this intent
    keywords: list[str] = INTENT_KEYWORDS.get(intent_name, [])

    if not keywords:
        return False

    # Combine all documents into one text for searching
    combined_text: str = " ".join(retrieved_docs).lower()

    # Check if any keyword is found in the retrieved documents
    for keyword in keywords:
        if keyword.lower() in combined_text:
            return True

    return False


def _run_retrieval(text: str) -> list[str]:
    """
    Run the retrieval pipeline for a given text.

    Params:
        text (str): The query text.

    Returns:
        list[str]: List of retrieved document texts.
    """
    # Generate embedding
    embedding: list[float] = get_embedding(text)

    # Search for similar documents
    retrieved_docs: list[str] = search_similar_docs(
        query_vector=embedding,
        top_k=3,
    )

    return retrieved_docs


def evaluate_sample(
    sample: MINDS14Sample,
    locale: str,
    audio_buffer: BytesIO,
) -> EvaluationResult:
    """
    Evaluate a single sample from the dataset.

    Runs both text and voice pipelines, calculates WER and hit rates.

    Params:
        sample (MINDS14Sample): The sample to evaluate.
        locale (str): The language locale of the sample.
        audio_buffer (BytesIO): Audio data for Whisper transcription.

    Returns:
        EvaluationResult: Complete evaluation metrics for the sample.
    """
    ground_truth: str = sample["transcription"]
    intent_class: int = sample["intent_class"]

    # Get ASR transcription using Whisper
    asr_text: str = transcribe_audio(audio_buffer)

    # Calculate WER
    wer_score: float = calculate_wer(ground_truth, asr_text)

    # Use English transcription for retrieval if available (better matching)
    text_for_retrieval: str = (
        sample["english_transcription"]
        if sample["english_transcription"]
        else ground_truth
    )

    # Run retrieval for text pipeline
    text_docs: list[str] = _run_retrieval(text_for_retrieval)
    hit_rate_text: int = 1 if check_hit_rate(text_docs, intent_class) else 0

    # Run retrieval for voice pipeline (using ASR text)
    voice_docs: list[str] = _run_retrieval(asr_text)
    hit_rate_voice: int = 1 if check_hit_rate(voice_docs, intent_class) else 0

    result: EvaluationResult = {
        "locale": locale,
        "sample_id": sample["id"],
        "intent_class": intent_class,
        "intent_name": sample["intent_name"],
        "ground_truth": ground_truth,
        "asr_text": asr_text,
        "wer_score": wer_score,
        "hit_rate_text": hit_rate_text,
        "hit_rate_voice": hit_rate_voice,
    }

    return result


def run_batch_evaluation(
    locales: list[str] | None = None,
    samples_per_locale: int = 10,
    progress_callback: callable = None,
) -> pd.DataFrame:
    """
    Run batch evaluation across multiple locales.

    Evaluates samples from each locale and computes WER and hit rate metrics.

    Params:
        locales (list[str] | None): List of locale codes to evaluate.
            Default is ['en-US', 'en-GB', 'en-AU'].
        samples_per_locale (int): Number of samples to evaluate per locale.
            Use -1 to evaluate all samples in the dataset.
            Default is 10.
        progress_callback (callable): Optional callback function for progress updates.
            Called with (current_sample, total_samples, locale, sample_id).

    Returns:
        pd.DataFrame: DataFrame with columns:
            - locale: Language locale
            - sample_id: Sample identifier
            - intent_class: Intent class ID
            - intent_name: Intent name
            - ground_truth: Original transcription
            - asr_text: Whisper transcription
            - wer_score: Word Error Rate
            - hit_rate_text: Text retrieval hit (1/0)
            - hit_rate_voice: Voice retrieval hit (1/0)
    """
    if locales is None:
        locales = ["en-US", "en-GB", "en-AU"]

    results: list[EvaluationResult] = []
    
    # If using full dataset, first calculate total samples across all locales
    if samples_per_locale == -1:
        # Load all loaders first to get total
        loaders: dict[str, MINDS14Loader] = {}
        total_samples: int = 0
        for locale in locales:
            loader = MINDS14Loader(language=locale)
            loaders[locale] = loader
            total_samples += len(loader)
    else:
        loaders = None
        total_samples = len(locales) * samples_per_locale

    current_sample: int = 0

    for locale in locales:
        print(f"\n=== Evaluating locale: {locale} ===")

        # Load dataset for this locale
        if loaders and locale in loaders:
            loader = loaders[locale]
        else:
            loader = MINDS14Loader(language=locale)

        # Determine number of samples to process
        if samples_per_locale == -1:
            num_samples: int = len(loader)  # Use full dataset
        else:
            num_samples = min(samples_per_locale, len(loader))

        print(f"  Processing {num_samples} samples...")

        for i in range(num_samples):
            current_sample += 1

            try:
                # Get sample and audio
                sample: MINDS14Sample = loader.get_sample(i)
                audio_buffer: BytesIO = loader.get_audio_bytes(i)

                # Progress callback
                if progress_callback:
                    progress_callback(current_sample, total_samples, locale, sample["id"])

                print(f"  [{current_sample}/{total_samples}] {sample['id']}: {sample['intent_name']}")

                # Evaluate sample
                result: EvaluationResult = evaluate_sample(sample, locale, audio_buffer)
                results.append(result)

            except Exception as e:
                print(f"  Error evaluating sample {i}: {e}")
                continue

    # Create DataFrame
    df: pd.DataFrame = pd.DataFrame(results)

    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize evaluation results by locale.

    Calculates average WER and hit rates per locale.

    Params:
        df (pd.DataFrame): Raw evaluation results DataFrame.

    Returns:
        pd.DataFrame: Summary DataFrame with aggregated metrics per locale.
    """
    summary: pd.DataFrame = df.groupby("locale").agg(
        total_samples=("sample_id", "count"),
        avg_wer=("wer_score", "mean"),
        std_wer=("wer_score", "std"),
        text_hit_rate=("hit_rate_text", "mean"),
        voice_hit_rate=("hit_rate_voice", "mean"),
    ).reset_index()

    # Convert hit rates to percentages
    summary["text_hit_rate"] = summary["text_hit_rate"] * 100
    summary["voice_hit_rate"] = summary["voice_hit_rate"] * 100
    summary["avg_wer"] = summary["avg_wer"] * 100

    # Rename for clarity
    summary = summary.rename(columns={
        "avg_wer": "avg_wer_%",
        "text_hit_rate": "text_hit_rate_%",
        "voice_hit_rate": "voice_hit_rate_%",
    })

    return summary


def summarize_by_intent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize evaluation results by intent class.

    Calculates average WER and hit rates per intent.

    Params:
        df (pd.DataFrame): Raw evaluation results DataFrame.

    Returns:
        pd.DataFrame: Summary DataFrame with aggregated metrics per intent.
    """
    summary: pd.DataFrame = df.groupby("intent_name").agg(
        total_samples=("sample_id", "count"),
        avg_wer=("wer_score", "mean"),
        text_hit_rate=("hit_rate_text", "mean"),
        voice_hit_rate=("hit_rate_voice", "mean"),
    ).reset_index()

    # Convert to percentages
    summary["text_hit_rate"] = summary["text_hit_rate"] * 100
    summary["voice_hit_rate"] = summary["voice_hit_rate"] * 100
    summary["avg_wer"] = summary["avg_wer"] * 100

    summary = summary.rename(columns={
        "avg_wer": "avg_wer_%",
        "text_hit_rate": "text_hit_rate_%",
        "voice_hit_rate": "voice_hit_rate_%",
    })

    return summary
