"""Local machine translation.

Engine: CTranslate2 running an int8-quantised OPUS-MT (Marian) model.

Why this choice for an 8 GB CPU-only laptop:
  * CTranslate2 is already installed as the faster-whisper backend, so
    translation adds no new inference runtime and no PyTorch.
  * OPUS-MT en-fr is a ~74M-parameter Transformer; int8 it is ~78 MB on disk
    and ~180 MB resident.
  * Typical latency on this class of CPU is well under 200 ms per sentence.

Phase 3 (Yoruba) plugs in here: download opus-mt-en-yo the same way and
instantiate a second MarianCT2Translator. Nothing else in the codebase changes.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import MTConfig

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TranslationResult:
    text: str
    compute_seconds: float


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> TranslationResult:
        ...


class MarianCT2Translator(Translator):
    """OPUS-MT / Marian model converted to CTranslate2."""

    def __init__(self, model_dir: str | Path, cfg: MTConfig):
        import ctranslate2
        import sentencepiece as spm

        self.cfg = cfg
        self.model_dir = Path(model_dir)

        src_spm = self.model_dir / "source.spm"
        tgt_spm = self.model_dir / "target.spm"
        if not src_spm.exists():
            raise FileNotFoundError(
                f"Missing {src_spm}. Run: python scripts/fetch_models.py"
            )

        self.sp_source = spm.SentencePieceProcessor(model_file=str(src_spm))
        self.sp_target = spm.SentencePieceProcessor(
            model_file=str(tgt_spm if tgt_spm.exists() else src_spm)
        )
        self.translator = ctranslate2.Translator(
            str(self.model_dir),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=cfg.cpu_threads,
        )

    def translate(self, text: str) -> TranslationResult:
        started = time.perf_counter()
        text = text.strip()
        if not text:
            return TranslationResult("", 0.0)

        # Marian was trained on single sentences; batching them is both more
        # accurate and cheaper than decoding one long sequence.
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        batch = [self.sp_source.encode(s, out_type=str) for s in sentences]

        results = self.translator.translate_batch(
            batch,
            beam_size=self.cfg.beam_size,
            max_decoding_length=self.cfg.max_decoding_length,
            replace_unknowns=True,
        )
        out = " ".join(
            self.sp_target.decode(r.hypotheses[0]) for r in results
        ).strip()
        return TranslationResult(out, time.perf_counter() - started)


# -------------------------------------------------------------- phrase overrides

_TRAILING_PUNCT = re.compile(r"([\.!?,;:]+)\s*$")


def load_phrase_overrides(path: Path) -> dict[str, str]:
    """Parse a `english = target` file. Case-insensitive on the English side.

    Empty file / missing file returns an empty dict, so this is safe to call
    unconditionally.
    """
    rules: dict[str, str] = {}
    if not path.exists():
        return rules
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        en, tgt = left.strip(), right.strip()
        if en and tgt:
            rules[en.lower()] = tgt
    return rules


class OverrideTranslator(Translator):
    """Wraps another `Translator` with a phrase table applied first.

    Matches per-sentence, not per-input. The input is split on sentence
    boundaries; each sentence whose core text (ignoring trailing punctuation)
    matches a rule is replaced verbatim; every other sentence goes to the
    underlying translator. The pieces are then reassembled in order.

    An earlier version rewrote matches as substrings inside the English input
    and then fed the mixed English+Yoruba string to the model, which produced
    garbage - Marian has never seen target-language characters as input.
    Sentence-level replacement avoids that.
    """

    def __init__(self, inner: Translator, rules: dict[str, str]):
        self.inner = inner
        self.rules = rules

    def _match(self, sentence: str) -> str | None:
        m = _TRAILING_PUNCT.search(sentence)
        core = _TRAILING_PUNCT.sub("", sentence).strip()
        tail = m.group(1) if m else ""
        hit = self.rules.get(core.lower())
        return (hit + tail) if hit is not None else None

    def _split(self, text: str) -> list[str]:
        """Sentence split, plus one extra: a comma right after a rule match at
        the start of a sentence is treated as a sentence boundary too. Handles
        input like "Hello, good morning. How are you..." where the greeting is
        stuck to the next clause by a comma the model would otherwise keep."""
        parts: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            while True:
                comma = sentence.find(",")
                if comma < 1:
                    break
                head = sentence[:comma].strip()
                if head.lower() not in self.rules:
                    break
                parts.append(head + ",")
                sentence = sentence[comma + 1:].strip()
            if sentence:
                parts.append(sentence)
        return parts

    def translate(self, text: str) -> TranslationResult:
        text = text.strip()
        if not text or not self.rules:
            return self.inner.translate(text)

        started = time.perf_counter()
        sentences = self._split(text)
        parts: list[str] = []
        to_translate: list[tuple[int, str]] = []
        for i, s in enumerate(sentences):
            hit = self._match(s)
            if hit is not None:
                parts.append(hit)
            else:
                parts.append("")             # placeholder, filled below
                to_translate.append((i, s))

        if to_translate:
            joined = " ".join(s for _, s in to_translate)
            mt = self.inner.translate(joined)
            # Split the model output back the same way we split the input.
            translated = [p for p in _SENTENCE_SPLIT.split(mt.text) if p.strip()]
            if len(translated) == len(to_translate):
                for (i, _), out in zip(to_translate, translated):
                    parts[i] = out
            else:
                # Sentence count didn't round-trip; safest is to fall back to
                # per-sentence translation for the unmatched ones.
                for i, s in to_translate:
                    parts[i] = self.inner.translate(s).text

        return TranslationResult(" ".join(parts).strip(),
                                 time.perf_counter() - started)
