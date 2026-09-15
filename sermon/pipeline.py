"""Wiring: audio source -> transcriber -> {translator per target} -> sinks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .audio import Utterance
from .mt import Translator
from .stt import Transcriber


@dataclass
class PipelineResult:
    english: str
    translations: dict[str, str]       # {"fr": "...", "yo": "..."}
    audio_seconds: float
    stt_seconds: float
    mt_seconds: dict[str, float]       # per-target compute cost
    total_latency: float               # utterance end -> LAST translation ready
    forced: bool = False
    peak_db: float = -120.0            # input level, surfaces clipping live

    @property
    def french(self) -> str:
        """Convenience for callers that only care about the French output."""
        return self.translations.get("fr", "")

    @property
    def mt_total_seconds(self) -> float:
        return sum(self.mt_seconds.values())

    @property
    def rtf(self) -> float:
        return (self.stt_seconds + self.mt_total_seconds) / max(self.audio_seconds, 1e-6)

    @property
    def clipped(self) -> bool:
        """Clipping distorts the waveform before we ever see it, so it costs
        accuracy and cannot be fixed in software - only at the mic gain."""
        return self.peak_db >= -0.5


class Pipeline:
    """Single-threaded on purpose: one utterance at a time keeps peak RAM low
    and avoids oversubscribing a 2-core CPU. Backpressure is handled upstream
    by the bounded queue in MicrophoneSource.

    `translators` is an insertion-ordered dict of {target_code: Translator};
    all run sequentially on each utterance. Adding a third target is one line.
    """

    def __init__(self, transcriber: Transcriber, translators: dict[str, Translator],
                 sinks: list | None = None, sample_rate: int = 16000):
        if not translators:
            raise ValueError("Pipeline needs at least one translator")
        self.transcriber = transcriber
        self.translators = translators
        self.sinks = sinks if sinks is not None else []
        self.sample_rate = sample_rate
        self.processed = 0
        self.skipped_empty = 0
        self.clipped = 0

    def process(self, utt: Utterance) -> PipelineResult | None:
        stt = self.transcriber.transcribe(utt.audio, self.sample_rate)
        if not stt.text:
            self.skipped_empty += 1
            return None

        translations: dict[str, str] = {}
        mt_seconds: dict[str, float] = {}

        # Targets sharing one multilingual model (NLLB) decode together: the
        # model runs once over a batch of (sentence, target) pairs instead of
        # once per language. Measured at 2.3-2.8x faster than looping, which
        # is the difference between four African languages being affordable
        # and not. Bilingual OPUS-MT targets have nothing to share and run
        # individually as before.
        #
        # A target carrying a phrase-override table is deliberately NOT
        # batched: those rules substitute canned text for whole sentences
        # BEFORE translation, so the wrapper has to own the call. It loses the
        # batching win and keeps the overrides, which is the right trade.
        grouped: dict[int, list] = {}
        for code, tr in self.translators.items():
            base = getattr(tr, "batch_base", None)
            if base is not None and hasattr(base, "translate_many"):
                grouped.setdefault(id(base), []).append((code, tr, base))

        for members in grouped.values():
            base = members[0][2]
            tags = [tr.target_code for _, tr, _ in members]
            started = time.monotonic()
            try:
                out = base.translate_many(stt.text, tags)
            except Exception:
                out = {}
            elapsed = time.monotonic() - started
            share = elapsed / max(len(members), 1)
            for code, tr, _ in members:
                translations[code] = out.get(tr.target_code, "")
                mt_seconds[code] = share

        for code, tr in self.translators.items():
            if code in translations:
                continue
            r = tr.translate(stt.text)
            translations[code] = r.text
            mt_seconds[code] = r.compute_seconds

        result = PipelineResult(
            english=stt.text,
            translations=translations,
            audio_seconds=stt.audio_seconds,
            stt_seconds=stt.compute_seconds,
            mt_seconds=mt_seconds,
            total_latency=time.monotonic() - utt.captured_at,
            forced=utt.forced,
            peak_db=utt.peak_db,
        )
        self.processed += 1
        if result.clipped:
            self.clipped += 1
        for sink in self.sinks:
            sink.emit(result)
        return result

    def run(self, source) -> None:
        """Consume utterances until the source stops or Ctrl+C."""
        for utt in source.utterances():
            self.process(utt)
