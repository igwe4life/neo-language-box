# PyInstaller spec for the desktop app bundle.
#
# Entry: desktop_app.py - starts the Python server on a background thread
# and opens a WebView2 window pointed at http://127.0.0.1:8000/operator.
# NO console window in the final .exe; users see a native app, not a
# terminal.
#
# Output: dist/NeoLanguageBox/
#     NeoLanguageBox.exe               (double-click launcher)
#     _internal/                       (Python + libraries)
#
# Models are copied in by the Inno Setup script (installer.iss), not by
# PyInstaller, so we can update the model set without rebuilding the .exe.

# ruff: noqa

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# ---------------------------------------------------------------- native deps
binaries, datas, hiddenimports = [], [], []

# ctranslate2 - shared library used by faster-whisper AND OPUS-MT
binaries += collect_dynamic_libs("ctranslate2")

# onnxruntime - Piper + sherpa-onnx + Silero VAD
binaries += collect_dynamic_libs("onnxruntime")
datas    += collect_data_files("onnxruntime")

# sherpa_onnx - Parakeet backend + Silero VAD wrapper
binaries += collect_dynamic_libs("sherpa_onnx")
try:
    binaries += collect_dynamic_libs("sherpa_onnx_core")
except Exception:
    pass
datas    += collect_data_files("sherpa_onnx")
hiddenimports += collect_submodules("sherpa_onnx")

# piper - espeak-ng data lives inside the package
datas += collect_data_files("piper", include_py_files=False)
hiddenimports += collect_submodules("piper")

# faster_whisper - tokenizer JSON etc.
datas += collect_data_files("faster_whisper", include_py_files=False)

# tokenizers - Rust extension
binaries += collect_dynamic_libs("tokenizers")

# sentencepiece
binaries += collect_dynamic_libs("sentencepiece")

# PortAudio via sounddevice + PyAudioWPatch (WASAPI loopback)
binaries += collect_dynamic_libs("_sounddevice_data")
datas    += collect_data_files("sounddevice", include_py_files=False)
try:
    import pyaudiowpatch, os
    root = os.path.dirname(pyaudiowpatch.__file__)
    for f in os.listdir(root):
        if f.endswith(".dll") or f.endswith(".pyd"):
            binaries.append((os.path.join(root, f), "pyaudiowpatch"))
except Exception:
    pass

# cryptography - self-signed cert generation
binaries += collect_dynamic_libs("cryptography")

# pywebview - runs the native window (WebView2 on Windows)
hiddenimports += collect_submodules("webview")
datas    += collect_data_files("webview", include_py_files=False)
# clr_loader is a pywebview dep for the Edge WebView2 backend
try:
    hiddenimports += collect_submodules("clr_loader")
    binaries += collect_dynamic_libs("clr_loader")
except Exception:
    pass

# segno - QR code SVG generation (pure python, no binaries)
hiddenimports += collect_submodules("segno")

# --------------------------------------------------------------- app resources
datas += [
    ("sermon/templates/listen.html",   "sermon/templates"),
    ("sermon/templates/speaker.html",  "sermon/templates"),
    ("sermon/templates/operator.html", "sermon/templates"),
    ("sermon/templates/qr.html",       "sermon/templates"),
    ("vocabulary.txt", "."),
    ("phrases.yo.txt", "."),
]


a = Analysis(
    ["desktop_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        "sermon.audio", "sermon.config", "sermon.console",
        "sermon.diagnostics", "sermon.enlex", "sermon.integrity", "sermon.mt",
        "sermon.pipeline", "sermon.server", "sermon.sinks",
        "sermon.stt", "sermon.telemetry", "sermon.tls", "sermon.tts",
        "run",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "notebook", "IPython", "tkinter",
        "torch",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeoLanguageBox",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX makes AV think we're malware
    console=False,           # windowed app, no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NeoLanguageBox",
)
