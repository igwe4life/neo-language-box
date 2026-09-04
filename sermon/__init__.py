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

__version__ = "0.1.0"
