"""Pinned MINDS-14 loading with content-verified, revision-aware audio caching."""
import hashlib
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import soundfile as sf
from datasets import Audio, Dataset, DownloadConfig, load_dataset
from src.knowledge_base import INTENT_NAMES
from src.config import get_settings

DEFAULT_DATASET_REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"


class MINDS14Sample(TypedDict):
    id: str
    source_id: str
    audio_path: str
    audio_array: list[float]
    sampling_rate: int
    transcription: str
    english_transcription: str
    intent_class: int
    intent_name: str
    dataset_revision: str
    audio_sha256: str


class MINDS14Loader:
    def __init__(self, language: str = "en-US", cache_dir: str = "data/minds14",
                 revision: str | None = None, offline: bool | None = None) -> None:
        if language not in get_available_languages():
            raise ValueError(f"Unsupported MINDS-14 locale: {language}")
        self.language = language
        self.revision = revision or get_settings().dataset_revision
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision.lower()):
            raise ValueError("MINDS14 revision must be an immutable 40-character commit SHA")
        self.offline = offline
        self.cache_dir = Path(cache_dir)
        self._dataset: Dataset | None = None

    def load(self) -> None:
        offline = bool(self.offline) or any(os.environ.get(k, "").lower() in {"1", "true", "yes"}
                    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))
        if offline:
            from datasets import config as datasets_config
            root = Path(os.environ.get('HF_DATASETS_CACHE', str(datasets_config.HF_DATASETS_CACHE)))
            cached = sorted((root/'PolyAI___minds14'/self.language).glob(
                f'*/{self.revision}/minds14-train.arrow'))
            if not cached:
                raise FileNotFoundError('Pinned MINDS-14 locale is not cached offline; preload this revision first')
            dataset = Dataset.from_file(str(cached[0]))
        else:
            dataset = load_dataset("PolyAI/minds14", self.language, split="train", revision=self.revision,
                                   download_config=DownloadConfig(local_files_only=False))
        labels = list(dataset.features["intent_class"].names)
        if set(labels) != set(INTENT_NAMES) or len(labels) != len(INTENT_NAMES):
            raise ValueError("MINDS-14 intent metadata does not match supported labels")
        self._dataset = dataset.cast_column("audio", Audio(decode=False))

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self.load()
        return self._dataset

    @property
    def intent_labels(self) -> list[str]:
        features = getattr(self.dataset, "features", {})
        feature = features.get("intent_class") if features else None
        return list(feature.names) if feature is not None else list(INTENT_NAMES)

    def get_intent_classes(self) -> list[int]:
        try:
            classes = list(self.dataset["intent_class"])
        except (TypeError, KeyError):
            classes = [row["intent_class"] for row in self.dataset]
        labels = self.intent_labels
        result = []
        for value in classes:
            if isinstance(value,bool) or not isinstance(value, int) or not 0 <= value < len(labels):
                raise ValueError(f"Invalid MINDS-14 intent class: {value}")
            result.append(INTENT_NAMES.index(labels[value]))
        return result

    def __len__(self) -> int:
        return len(self.dataset)

    def get_sample(self, index: int, *, decode_audio: bool = True) -> MINDS14Sample:
        if not 0 <= index < len(self.dataset):
            raise IndexError(f"Index {index} out of range [0, {len(self.dataset)})")
        item = self.dataset[index]
        raw_class = item["intent_class"]
        labels = self.intent_labels
        if isinstance(raw_class,bool) or not isinstance(raw_class, int) or not 0 <= raw_class < len(labels):
            raise ValueError(f"Invalid MINDS-14 intent class: {raw_class}")
        intent_name = labels[raw_class]
        if intent_name not in INTENT_NAMES:
            raise ValueError(f"Unsupported intent label: {intent_name}")
        info = item["audio"]
        raw = info.get("bytes")
        if raw is None and info.get("path"):
            raw = Path(info["path"]).read_bytes()
        if not isinstance(raw, bytes) or not raw:
            raise ValueError(f"MINDS-14 sample {index} has no audio bytes or readable path")
        digest = hashlib.sha256(raw).hexdigest()
        source_id = str(item.get("id") or item.get("path") or info.get("path") or f"row-{index}")
        stable_id = hashlib.sha256(source_id.replace("\\", "/").encode()).hexdigest()[:16]
        sample_id = f"{self.language}_{stable_id}"
        audio_path = self.cache_dir / self.revision / f"{sample_id}_{digest}.wav"
        try:
            audio_info = sf.info(BytesIO(raw))
            array = sf.read(BytesIO(raw))[0].tolist() if decode_audio else []
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid audio for MINDS-14 sample {sample_id}") from exc
        if not audio_path.exists() or hashlib.sha256(audio_path.read_bytes()).hexdigest() != digest:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=audio_path.parent, suffix=".tmp", delete=False) as f:
                temp_path = Path(f.name)
                f.write(raw)
            try:
                temp_path.replace(audio_path)
            finally:
                temp_path.unlink(missing_ok=True)
        return MINDS14Sample(id=sample_id, source_id=source_id, audio_path=str(audio_path), audio_array=array,
            sampling_rate=int(audio_info.samplerate), transcription=item["transcription"],
            english_transcription=item.get("english_transcription", ""), intent_class=INTENT_NAMES.index(intent_name),
            intent_name=intent_name, dataset_revision=self.revision, audio_sha256=digest)

    def get_sample_list(self, limit: int = 10) -> list[MINDS14Sample]:
        if limit < 0:
            raise ValueError("Sample limit cannot be negative")
        return [self.get_sample(i) for i in range(min(limit, len(self)))]

    def get_audio_bytes(self, index: int, sample: MINDS14Sample | None = None) -> BytesIO:
        sample = sample or self.get_sample(index, decode_audio=False)
        path = Path(sample["audio_path"])
        buffer = BytesIO(path.read_bytes())
        buffer.name = path.name
        return buffer


def get_available_languages() -> list[str]:
    return ["cs-CZ", "de-DE", "en-AU", "en-GB", "en-US", "es-ES", "fr-FR",
            "it-IT", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN"]
