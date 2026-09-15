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


class NllbTranslator(Translator):
    """One NLLB-200 model serving several target languages.

    Used for Igbo, Yoruba, Hausa and Swahili. OPUS-MT does have bilingual
    models for these, but they are low-resource pairs and a native speaker
    judged them worse than NLLB in a blind comparison; the European and Asian
    targets stay on OPUS-MT, which is both better there and ~13x faster.

    Unlike Marian, the target language is not baked into the model - it is a
    token. The source is the sentence followed by </s> and the SOURCE language
    tag, and the target is prefixed with the TARGET tag. A wrong arrangement
    here does not raise: it produces fluent-looking output in no particular
    language, so `SHARED` instances are validated at startup.

    Because one model serves every target, all of them can be decoded in a
    SINGLE batched call - measured at 2.3-2.8x faster than looping, which is
    what makes carrying a 1.3B model affordable.
    """

    SRC = "eng_Latn"

    def __init__(self, model_dir: str | Path, cfg: MTConfig, target_code: str):
        import ctranslate2
        import sentencepiece as spm

        self.cfg = cfg
        self.target_code = target_code       # e.g. "ibo_Latn"
        self.model_dir = Path(model_dir)
        spm_path = self.model_dir / "sentencepiece.bpe.model"
        if not spm_path.exists():
            raise FileNotFoundError(
                f"Missing {spm_path}. Run: python scripts/fetch_models.py"
            )
        self.sp = spm.SentencePieceProcessor(model_file=str(spm_path))
        self.translator = ctranslate2.Translator(
            str(self.model_dir),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=cfg.cpu_threads,
        )

    # -- shared-instance plumbing ---------------------------------------
    # Loading a 1.3B model once per language would cost gigabytes, so callers
    # build ONE and clone a cheap view per target.

    def for_target(self, target_code: str) -> "NllbTranslator":
        clone = object.__new__(NllbTranslator)
        clone.cfg = self.cfg
        clone.model_dir = self.model_dir
        clone.sp = self.sp
        clone.translator = self.translator
        clone.target_code = target_code
        # Lets the pipeline recognise that several targets ride on one model
        # and decode them together instead of one call each.
        clone.batch_base = self
        return clone

    batch_base = None

    def _encode(self, text: str) -> list:
        return self.sp.encode(text, out_type=str) + ["</s>", self.SRC]

    def _decode(self, tokens: list) -> str:
        if tokens and tokens[0].endswith("_Latn") or (
                tokens and tokens[0] == self.target_code):
            tokens = tokens[1:]
        return self.sp.decode(tokens)

    def translate(self, text: str) -> TranslationResult:
        started = time.perf_counter()
        text = text.strip()
        if not text:
            return TranslationResult("", 0.0)
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        batch = [self._encode(s) for s in sentences]
        results = self.translator.translate_batch(
            batch,
            target_prefix=[[self.target_code]] * len(batch),
            beam_size=self.cfg.beam_size,
            max_decoding_length=self.cfg.max_decoding_length,
        )
        out = " ".join(self._decode(r.hypotheses[0]) for r in results).strip()
        return TranslationResult(out, time.perf_counter() - started)

    def translate_many(self, text: str, targets: list) -> dict:
        """Every target in one batched pass. Returns {target_code: text}."""
        text = text.strip()
        if not text or not targets:
            return {}
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        batch, prefixes, owner = [], [], []
        for code in targets:
            for s in sentences:
                batch.append(self._encode(s))
                prefixes.append([code])
                owner.append(code)
        results = self.translator.translate_batch(
            batch,
            target_prefix=prefixes,
            beam_size=self.cfg.beam_size,
            max_decoding_length=self.cfg.max_decoding_length,
            max_batch_size=len(batch),
        )
        parts: dict = {}
        for code, r in zip(owner, results):
            parts.setdefault(code, []).append(self._decode(r.hypotheses[0]))
        return {c: " ".join(v).strip() for c, v in parts.items()}


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
