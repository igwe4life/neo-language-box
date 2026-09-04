# PyInstaller spec for the desktop-app server bundle.
#
# Output: dist/sermon/  (one-dir mode)
#   sermon.exe            <- the launcher (double-clickable, or spawned by Tauri)
#   _internal/            <- Python runtime + all libraries + resources
#     ...
#
# Models are NOT bundled here - they sit next to sermon.exe in the final
# installer (Tauri's resource system copies them). That keeps this build
# fast, and lets us update the model set without rebuilding the .exe.

# ruff: noqa

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# ---------------------------------------------------------------- native deps
# These packages ship native DLLs / data files that PyInstaller's auto-detection
# misses without help.

binaries = []
datas = []
hiddenimports = []

# ctranslate2 - shared library used by faster-whisper AND OPUS-MT
binaries += collect_dynamic_libs("ctranslate2")

# onnxruntime - shared library used by Piper AND sherpa-onnx (and Parakeet)
binaries += collect_dynamic_libs("onnxruntime")
datas    += collect_data_files("onnxruntime")

# sherpa_onnx - native runtime + Python bindings for Parakeet
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

# sounddevice - not needed at runtime for the desktop build (we use the
# browser mic), but the module is imported by sermon.audio. Include the
# PortAudio DLL so the import doesn't fail on machines without it.
binaries += collect_dynamic_libs("_sounddevice_data")
datas    += collect_data_files("sounddevice", include_py_files=False)

# cryptography - self-signed cert generation
binaries += collect_dynamic_libs("cryptography")

# --------------------------------------------------------------- app resources
# HTML pages that live inside sermon/templates/
datas += [("sermon/templates/listen.html",  "sermon/templates"),
          ("sermon/templates/speaker.html", "sermon/templates"),
          ("vocabulary.txt", "."),
          ("phrases.yo.txt", ".")]


a = Analysis(
    ["desktop_main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        "sermon.audio", "sermon.config", "sermon.console",
        "sermon.diagnostics", "sermon.enlex", "sermon.mt",
        "sermon.pipeline", "sermon.server", "sermon.sinks",
        "sermon.stt", "sermon.tls", "sermon.tts",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # trim things we know we don't need at runtime in the desktop app
        "matplotlib", "notebook", "IPython", "tkinter",
        "torch",     # not installed anyway; belt-and-braces
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
    name="sermon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX makes AV think we're malware
    console=True,            # keep a console window during dev; Tauri hides it in prod
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
    name="sermon",
)
