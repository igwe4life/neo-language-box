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


class VitsSpeaker(Speaker):
    """Raw VITS voices for languages Piper has none of.

    Piper covers most of the target list, but not Igbo, Yoruba or Hausa. These
    run on the SAME onnxruntime Piper uses - no torch in the shipped app - but
    they are plain ONNX graphs rather than Piper bundles, so the tokenising and
    the call signature are ours to get right.

    Two export conventions are in the wild and both are supported:

    "coqui"   input / input_lengths / scales -> output
              Symbol table is ['<PAD>','<EOS>','<BOS>','<BLNK>', ...chars...],
              so the blank is id 3 and NOT the last entry. Text is lowercased.
              Guessing this ordering instead of reading it produced confident
              garbled audio that was not recognisably the language - the model
              happily renders whatever symbols it is handed.

    "mms"     x / x_length / noise_scale / length_scale / noise_scale_w -> y
              Symbol table comes from a "char id" tokens.txt, blank is 0.

    Both interleave the blank between every symbol; VITS is trained that way
    and omitting it garbles the output.
    """

    def __init__(self, model_dir: str | Path, cfg: TTSConfig):
        import json

        import onnxruntime as ort

        self.cfg = cfg
        d = Path(model_dir)
        onnx = d / "model.onnx"
        if not onnx.exists():
            raise FileNotFoundError(
                f"Missing {onnx}. Run: python scripts/fetch_models.py"
            )

        self.session = ort.InferenceSession(
            str(onnx), providers=["CPUExecutionProvider"])
        names = {i.name for i in self.session.get_inputs()}
        self.style = "coqui" if "input" in names else "mms"

        coqui_meta = d / "coqui_tokens.json"
        if coqui_meta.exists():
            meta = json.loads(coqui_meta.read_text(encoding="utf-8"))
            self.vocab = {c: i for i, c in enumerate(meta["vocab"])}
            self.blank = meta["blank_id"]
            self.sample_rate = meta["sample_rate"]
        else:
            self.vocab = self._read_tokens(d / "tokens.txt")
            self.blank = 0
            self.sample_rate = 16000

        # Amharic's MMS-TTS checkpoint was trained on romanised text (its
        # vocabulary is 28 plain Latin letters, no Ge'ez) - see
        # scripts/export_mms_amharic.py, which writes this flag file.
        self._uroman = None
        if (d / "romanize.flag").exists():
            import uroman
            self._uroman = uroman.Uroman()

    @staticmethod
    def _read_tokens(path: Path) -> dict:
        vocab: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ")
            # Some rows map several characters to one id (case variants); the
            # first wins, which matches how the exports were built.
            if len(parts) == 2 and parts[0] != "":
                vocab.setdefault(parts[0], int(parts[1]))
        return vocab

    def _encode(self, text: str) -> list:
        if self._uroman is not None:
            text = self._uroman.romanize_string(text)
        ids = [self.vocab[c] for c in text.lower() if c in self.vocab]
        seq = [self.blank] * (len(ids) * 2 + 1)
        seq[1::2] = ids
        return seq

    def synthesize(self, text: str) -> Speech:
        started = time.perf_counter()
        text = (text or "").strip()
        seq = self._encode(text) if text else []
        # A sequence of nothing but blanks has no content to voice.
        if len(seq) <= 1:
            return Speech(np.zeros(0, dtype=np.float32), self.sample_rate, 0.0)

        x = np.array([seq], dtype=np.int64)
        if self.style == "coqui":
            out = self.session.run(["output"], {
                "input": x,
                "input_lengths": np.array([x.shape[1]], dtype=np.int64),
                "scales": np.array([0.667, self.cfg.length_scale, 0.8],
                                   dtype=np.float32),
            })[0]
        else:
            out = self.session.run(["y"], {
                "x": x,
                "x_length": np.array([x.shape[1]], dtype=np.int64),
                "noise_scale": np.array([0.667], dtype=np.float32),
                "length_scale": np.array([self.cfg.length_scale],
                                         dtype=np.float32),
                "noise_scale_w": np.array([0.8], dtype=np.float32),
            })[0]

        audio = np.asarray(out, dtype=np.float32).squeeze()
        # These voices come out far quieter than Piper's (peaks around 0.15
        # against 1.0), which on a phone in a hall is the difference between
        # audible and not. Normalise so every language lands at a similar level.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = audio / peak * 0.9 * self.cfg.volume
        return Speech(
            audio=audio.astype(np.float32, copy=False),
            sample_rate=self.sample_rate,
            compute_seconds=time.perf_counter() - started,
        )
