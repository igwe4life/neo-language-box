"""Local French text-to-speech.

Engine: Piper (VITS exported to ONNX), running on the same onnxruntime that
faster-whisper already installs. No PyTorch, no network, no API key. A medium
French voice is ~61 MB on disk and synthesises far faster than real time on CPU.

Phase 3 (Yoruba) and Phase 5 (streaming to phones) both plug in here: add
another voice, or take the float32 array from `synthesize` and send it over the
wire instead of to the speakers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TTSConfig


@dataclass
class Speech:
    audio: np.ndarray          # float32 mono in [-1, 1]
    sample_rate: int
    compute_seconds: float

    @property
    def duration(self) -> float:
        return len(self.audio) / max(self.sample_rate, 1)

    @property
    def rtf(self) -> float:
        """< 1.0 means we synthesise faster than the speech plays."""
        return self.compute_seconds / max(self.duration, 1e-6)


class Speaker(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Speech:
        ...


class PiperSpeaker(Speaker):
    def __init__(self, model_path: str | Path, cfg: TTSConfig):
        from piper import PiperVoice, SynthesisConfig

        self.cfg = cfg
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing {self.model_path}. Run: python scripts/fetch_models.py"
            )

        self.voice = PiperVoice.load(str(self.model_path))
        # length_scale > 1 slows delivery down; a sermon reads better unhurried.
        self.syn_config = SynthesisConfig(
            length_scale=cfg.length_scale,
            volume=cfg.volume,
            normalize_audio=True,
        )
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text: str) -> Speech:
        started = time.perf_counter()
        text = text.strip()
        if not text:
            return Speech(np.zeros(0, dtype=np.float32), self.sample_rate, 0.0)

        chunks = list(self.voice.synthesize(text, syn_config=self.syn_config))
        if not chunks:
            return Speech(np.zeros(0, dtype=np.float32), self.sample_rate, 0.0)

        audio = np.concatenate(
            [c.audio_float_array.astype(np.float32, copy=False) for c in chunks]
        )
        return Speech(
            audio=audio,
            sample_rate=chunks[0].sample_rate,
            compute_seconds=time.perf_counter() - started,
        )
