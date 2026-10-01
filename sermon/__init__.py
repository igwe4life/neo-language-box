"""Local, offline sermon translation pipeline.

Phase 1 scope: microphone -> local STT -> local EN->FR translation -> console.

The package is deliberately split so that later phases can extend it without
rewriting anything:

    audio.py      audio sources          (phase 5: network / line-in sources)
    stt.py        speech recognition     (swappable Transcriber implementations)
    mt.py         machine translation    (phase 3: add Yoruba as another target)
    sinks.py      output destinations    (phase 2: TTS sink, phase 6: websocket)
    pipeline.py   wiring: source -> stt -> mt -> sinks
"""

import os

# CTranslate2 (Whisper, OPUS-MT, NLLB) and sherpa-onnx (Parakeet, Silero VAD)
# each bundle their own copy of Intel's OpenMP runtime. Loading both in one
# process - which --serve always does - makes the SECOND one to initialise
# raise "mkl_malloc: failed to allocate memory", a real crash with a
# misleading name that has nothing to do with how much RAM is free: it is a
# known duplicate-OpenMP-runtime conflict, and this is Intel's own documented
# workaround. Must be set before those libraries are imported anywhere, which
# is why it lives here rather than in run.py - every entry point imports a
# sermon submodule, which runs this first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
