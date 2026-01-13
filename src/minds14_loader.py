"""
MINDS-14 Dataset Loader for the Dual-Mode RAG Application.

This module provides functions to load and manage the MINDS-14 dataset
from HuggingFace for ASR benchmarking experiments.

Dataset: https://huggingface.co/datasets/PolyAI/minds14
"""

import os
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import soundfile as sf
from datasets import load_dataset, Dataset


class MINDS14Sample(TypedDict):
    """
    Represents a single sample from the MINDS-14 dataset.

    Attributes:
        id (str): Unique identifier for the sample.
        audio_path (str): Path to the saved audio file.
        audio_array (list[float]): Raw audio data as float array.
        sampling_rate (int): Audio sampling rate in Hz.
        transcription (str): Original language transcription.
        english_transcription (str): English translation of transcription.
        intent_class (int): Intent class ID.
        intent_name (str): Human-readable intent name.
    """

    id: str
    audio_path: str
    audio_array: list[float]
    sampling_rate: int
    transcription: str
    english_transcription: str
    intent_class: int
    intent_name: str


# Intent class names from MINDS-14 dataset
INTENT_NAMES: list[str] = [
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
]


class MINDS14Loader:
    """
    Loader class for the MINDS-14 dataset.

    Handles downloading, caching, and accessing samples from the
    HuggingFace MINDS-14 dataset for ASR benchmarking.

    Attributes:
        language (str): Language code (e.g., 'en-US', 'id-ID').
        dataset (Dataset): The loaded HuggingFace dataset.
        cache_dir (Path): Directory for caching audio files.
    """

    def __init__(
        self,
        language: str = "en-US",
        cache_dir: str = "data/minds14",
    ) -> None:
        """
        Initialize the MINDS-14 dataset loader.

        Params:
            language (str): Language code to load. Default is 'en-US'.
                Available: cs-CZ, de-DE, en-AU, en-GB, en-US, es-ES,
                fr-FR, it-IT, ko-KR, nl-NL, pl-PL, pt-PT, ru-RU, zh-CN.
            cache_dir (str): Directory to cache audio files.

        Returns:
            None
        """
        self.language: str = language
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._dataset: Dataset | None = None

    def load(self) -> None:
        """
        Load the dataset from HuggingFace.

        Downloads the dataset if not cached locally.
        Disables automatic audio decoding to avoid FFmpeg/torchcodec
        dependency issues on Windows.

        Params:
            None

        Returns:
            None
        """
        from datasets import Audio

        print(f"Loading MINDS-14 dataset for language: {self.language}")

        # Load dataset with audio decoding disabled to avoid torchcodec issues
        dataset = load_dataset(
            "PolyAI/minds14",
            self.language,
            split="train",
            trust_remote_code=True,
        )

        # Disable automatic audio decoding - we'll handle it manually with soundfile
        self._dataset = dataset.cast_column(
            "audio",
            Audio(decode=False),
        )
        print(f"Loaded {len(self._dataset)} samples")

    @property
    def dataset(self) -> Dataset:
        """
        Get the loaded dataset, loading it if necessary.

        Params:
            None

        Returns:
            Dataset: The loaded HuggingFace dataset.
        """
        if self._dataset is None:
            self.load()
        return self._dataset

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Params:
            None

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def get_sample(self, index: int) -> MINDS14Sample:
        """
        Get a single sample from the dataset.

        Params:
            index (int): Index of the sample to retrieve.

        Returns:
            MINDS14Sample: The sample data including audio and transcription.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range [0, {len(self.dataset)})")

        item = self.dataset[index]

        # Generate sample ID
        sample_id: str = f"{self.language}_sample_{index:04d}"

        # Audio path for caching
        audio_path: Path = self.cache_dir / f"{sample_id}.wav"

        # Extract audio data - since decode=False, audio is a dict with 'path' and 'bytes'
        audio_info = item["audio"]

        if not audio_path.exists():
            # Get raw audio bytes from the dataset
            if "bytes" in audio_info and audio_info["bytes"] is not None:
                # Audio is stored as bytes
                audio_bytes: bytes = audio_info["bytes"]
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
            elif "path" in audio_info and audio_info["path"] is not None:
                # Audio is stored as a file path - copy it
                import shutil
                shutil.copy(audio_info["path"], audio_path)

        # Read audio file with soundfile to get array and sampling rate
        audio_array_np, sampling_rate = sf.read(str(audio_path))
        audio_array: list[float] = audio_array_np.tolist()

        # Get intent name
        intent_class: int = item["intent_class"]
        intent_name: str = (
            INTENT_NAMES[intent_class]
            if intent_class < len(INTENT_NAMES)
            else f"unknown_{intent_class}"
        )

        sample: MINDS14Sample = {
            "id": sample_id,
            "audio_path": str(audio_path),
            "audio_array": audio_array,
            "sampling_rate": int(sampling_rate),
            "transcription": item["transcription"],
            "english_transcription": item.get("english_transcription", ""),
            "intent_class": intent_class,
            "intent_name": intent_name,
        }

        return sample

    def get_sample_list(self, limit: int = 10) -> list[MINDS14Sample]:
        """
        Get a list of samples from the dataset.

        Params:
            limit (int): Maximum number of samples to retrieve.

        Returns:
            list[MINDS14Sample]: List of sample data.
        """
        samples: list[MINDS14Sample] = []
        for i in range(min(limit, len(self.dataset))):
            samples.append(self.get_sample(i))
        return samples

    def get_audio_bytes(self, index: int) -> BytesIO:
        """
        Get audio data as BytesIO buffer for Whisper API.

        Params:
            index (int): Index of the sample.

        Returns:
            BytesIO: Audio data as BytesIO buffer.
        """
        sample: MINDS14Sample = self.get_sample(index)
        audio_path: Path = Path(sample["audio_path"])

        with open(audio_path, "rb") as f:
            audio_bytes: BytesIO = BytesIO(f.read())
            audio_bytes.name = audio_path.name

        return audio_bytes


def get_available_languages() -> list[str]:
    """
    Get list of available languages in MINDS-14 dataset.

    Params:
        None

    Returns:
        list[str]: List of language codes.
    """
    return [
        "cs-CZ",  # Czech
        "de-DE",  # German
        "en-AU",  # English (Australia)
        "en-GB",  # English (UK)
        "en-US",  # English (US)
        "es-ES",  # Spanish
        "fr-FR",  # French
        "it-IT",  # Italian
        "ko-KR",  # Korean
        "nl-NL",  # Dutch
        "pl-PL",  # Polish
        "pt-PT",  # Portuguese
        "ru-RU",  # Russian
        "zh-CN",  # Chinese
    ]
