"""Wiring: audio source -> transcriber -> {translator per target} -> sinks."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .audio import Utterance
from .mt import Translator
from .stt import Transcriber

# Bounds how many languages translate concurrently per utterance. Each
# individual translate() call already runs with intra_threads=1 on this class
# of machine (see mt_threads() in config.py), so a handful of them sharing the
# CPU's logical threads is a real win, not oversubscription stacked on
# oversubscription - see Pipeline's docstring for the measurement.
_FANOUT_WORKERS = max(2, min(4, os.cpu_count() or 2))


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
    """One utterance at a time, by design: that is what keeps peak RAM low and
    backpressure simple, and it is unaffected by the change below. Backpressure
    for utterances themselves is handled upstream by the bounded queue in
    MicrophoneSource.

    Translating into every configured target WITHIN one utterance used to be
    the single biggest cost in the whole pipeline once more than a few
    languages were configured: looping translate() one language at a time
    measured at 3.5-4+ seconds of pure translation time for a 14-language
    default (nine ~100-200ms OPUS-MT calls plus one ~2.5-3s batched NLLB
    call) - BEFORE speech recognition or synthesis. Worse, a language late in
    that loop could miss the listener page's 6s wait-for-server-audio window,
    so its phone gave up and spoke its own (often poor, for a low-resource
    language) fallback voice instead - "sometimes no audio" was this same
    queueing problem, not a separate bug.

    Each translate() call already runs with intra_threads=1 on a 2-core/
    4-thread machine (see mt_threads() in config.py), so running several
    concurrently is real parallelism, not threads fighting over one core the
    sequential version already had to itself.

    `translators` is an insertion-ordered dict of {target_code: Translator}.
    """

    def __init__(self, transcriber: Transcriber, translators: dict[str, Translator],
                 sinks: list | None = None, sample_rate: int = 16000, recorder=None):
        # May start empty and change while running: under on-demand loading
        # (sermon/ondemand.py) another thread adds and removes entries, so
        # process() works from a snapshot.
        self.transcriber = transcriber
        self.translators = translators
        self.sinks = sinks if sinks is not None else []
        self.sample_rate = sample_rate
        self.recorder = recorder
        self.processed = 0
        self.skipped_empty = 0
        self.clipped = 0
        self._pool = ThreadPoolExecutor(
            max_workers=_FANOUT_WORKERS, thread_name_prefix="mt-fanout")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def process(self, utt: Utterance) -> PipelineResult | None:
        stt = self.transcriber.transcribe(utt.audio, self.sample_rate)
        if not stt.text:
            self.skipped_empty += 1
            return None

        # Same gate as everything else downstream: an utterance that was
        # dropped as noise/silence is not recorded either, so the source file
        # and every per-language file stay aligned on "what was actually said".
        if self.recorder is not None:
            self.recorder.write_source(utt.audio, self.sample_rate)

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
        active = list(self.translators.items())
        grouped: dict[int, list] = {}
        grouped_codes: set[str] = set()
        for code, tr in active:
            base = getattr(tr, "batch_base", None)
            if base is not None and hasattr(base, "translate_many"):
                grouped.setdefault(id(base), []).append((code, tr, base))
                grouped_codes.add(code)

        def _run_group(members):
            base = members[0][2]
            tags = [tr.target_code for _, tr, _ in members]
            started = time.monotonic()
            try:
                out = base.translate_many(stt.text, tags)
            except Exception:
                out = {}
            elapsed = time.monotonic() - started
            share = elapsed / max(len(members), 1)
            return [(code, out.get(tr.target_code, ""), share) for code, tr, _ in members]

        def _run_single(code, tr):
            r = tr.translate(stt.text)
            return [(code, r.text, r.compute_seconds)]

        # Every group and every un-batched target translates concurrently -
        # this fan-out, not any single call, was the bulk of per-utterance
        # latency once more than a couple of languages were configured (see
        # the class docstring). Each future still only does CPU-bound work in
        # C++ (CTranslate2), so this is wall-clock parallelism, not just
        # interleaving - the GIL is released for the duration of the call.
        futures = [self._pool.submit(_run_group, members) for members in grouped.values()]
        futures += [self._pool.submit(_run_single, code, tr)
                    for code, tr in active if code not in grouped_codes]
        for fut in futures:
            for code, text, seconds in fut.result():
                translations[code] = text
                mt_seconds[code] = seconds

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
