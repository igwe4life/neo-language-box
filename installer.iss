; Inno Setup script for Neo Language Box.
; Wraps the PyInstaller bundle (dist/NeoLanguageBox/) AND the language models
; into a single .exe installer.
;
; Build, in order:
;   python -m PyInstaller --clean --noconfirm desktop.spec
;   python scripts/build_models_ship.py --clean
;   tools\innosetup\ISCC.exe installer.iss
; Output:   dist/NeoLanguageBoxSetup.exe
;
; SIZE LIMIT - read before adding models
; --------------------------------------
; Inno Setup cannot put more than 4,200,000,000 bytes of compressed data into
; a single Setup.exe. The bundled models alone are ~4.14 GB, so they MUST be
; compressed to fit - stored uncompressed, the installer was 47 MB over. With
; compression the Sept 2026 build (14 languages) is 3.62 GB, leaving ~575 MB:
; room for three or four more languages at ~140 MB each. Check the size after
; adding any. Past the limit compilation fails, and the choices are
; DiskSpanning=yes (Setup.exe plus .bin slices that must travel together) or
; dropping the fallback speech engines (parakeet-110m, whisper base.en: 337 MB).

#if !FileExists(AddBackslash(SourcePath) + "models-ship\models\models-manifest.json")
  #error models-ship\models is missing. Run: python scripts/build_models_ship.py --clean
#endif
;
; Design decisions
; ----------------
; * Per-user install by default (DefaultDirName -> LocalAppData). No admin
;   elevation required to install or uninstall, which matches the "sideload
;   unsigned" model.
; * Desktop + Start Menu shortcuts, both point at the .exe.
; * SetupIconFile omitted deliberately; installer uses a generic icon.

#define AppName        "Neo Language Box - Igbo Test"
#define AppShortName   "NeoLanguageBoxIgboTest"
; Kept in sync by hand with sermon/config.py's APP_VERSION - Inno Setup
; cannot read that file at build time. Bump both together, and update
; telemetry-server/manifest.json's app_version too so update pings fire.
#define AppVersion     "0.8.0"
#define AppPublisher   "Neo Language Box"
#define AppExeName     "NeoLanguageBox.exe"

[Setup]
; Unchanged across the rename from "Sermon Translation Speaker". Windows keys
; an installed program on this GUID, so keeping it makes this installer an
; UPGRADE of an existing install rather than a second app beside it.
; IGBO TEST BUILD: a DIFFERENT AppId on purpose - this is a separate app
; installed beside Neo Language Box, and must never upgrade or replace it.
AppId={{5D2B7C1E-8A43-4F6E-B9D1-3C7A2E9F4B60}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL=https://example.local/
DefaultDirName={localappdata}\{#AppShortName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=NeoLanguageBoxIgboTestSetup
OutputDir=dist
; Compression is required for size (see the limit above), but these models are
; already quantised and only shrink 6-20% at any setting, so /max buys almost
; nothing over the default level while costing a great deal of build time.
; Non-solid keeps memory use sane across 4 GB of input.
Compression=lzma2
SolidCompression=no
; 20 languages pushed raw models past ~5 GB; measured LZMA2 ratio on these
; already-quantised model.bin files is ~0.91 (barely compresses), so a
; single Setup.exe would exceed Inno's 4.2 GB hard limit - this is the
; DiskSpanning path the comment at the top of this file anticipated.
; Setup.exe plus the .bin slices must all be kept together and shipped as
; a set; the installer itself asks for the next one as it needs it.
; IGBO TEST BUILD: one language fits Inno's single-file limit - one .exe.
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=
; On a machine that already has an old install, this keeps using that folder
; (SermonTranslation, then NeoSermon) so a models folder already copied there
; stays valid. A fresh install gets the new NeoLanguageBox folder.
UsePreviousAppDir=no
CloseApplications=yes
RestartApplications=no
; Keep the "Ready to install" page: with ~4 GB to unpack, it is worth one
; confirmation before the long copy starts.
DisableReadyPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

[Files]
; The PyInstaller-produced folder, verbatim.
Source: "dist\NeoLanguageBox\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; The language models: only what the app loads, as assembled by
; scripts/build_models_ship.py, including models-manifest.json so the app's
; first-launch integrity check covers the installed copy.
Source: "models-ship\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
; Where each model came from and its licence. Two are non-commercial, and this
; installer redistributes them, so the notice travels with it.
Source: "models-ship\README.txt"; DestDir: "{app}"; DestName: "MODELS-README.txt"; Flags: ignoreversion
; Application vocab files (already inside the bundle's _internal/, but
; also placed at the app root so `SERMON_VOCAB_FILE` overrides work).
Source: "vocabulary.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "phrases.yo.txt";  DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Registry]
; IGBO TEST BUILD: the usage counter starts at every login. Its counted state
; lives under the same key and is deliberately NOT removed on uninstall, so
; uninstalling and reinstalling does not reset it.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "NeoLanguageBoxIgboTestCounter"; ValueData: """{app}\{#AppExeName}"" --counter"; Flags: uninsdeletevalue

[Run]
; Start the usage counter straight after installation.
Filename: "{app}\{#AppExeName}"; Parameters: "--counter"; Flags: nowait runhidden
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop only THIS app's counter - the main Neo Language Box shares the exe
; name, so a plain taskkill /IM would close it too.
Filename: "powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -Command ""Get-Process NeoLanguageBox -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{app}\*' }} | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "StopIgboTestCounter"

[InstallDelete]
; The executable was renamed with the app, twice now. An upgrade over an
; older install would otherwise leave the old exe beside the new one - still
; launchable, and running stale code.
Type: files; Name: "{app}\SermonTranslationSpeaker.exe"
Type: files; Name: "{app}\NeoSermon.exe"

[UninstallDelete]
; Runtime files the app creates alongside itself. sermon/tls.py writes the
; self-signed certificate two levels above itself, which in the installed app
; is _internal\certs - cleaning only {app}\certs left the key and certificate
; behind after uninstall.
Type: filesandordirs; Name: "{app}\_internal\certs"
Type: filesandordirs; Name: "{app}\certs"
Type: files; Name: "{app}\models-verified.json"
; sermon/telemetry.py's install id and opt-out flag, same two-levels-up
; reasoning as the certs above.
Type: files; Name: "{app}\_internal\telemetry.json"
; The bundled models are installer-owned files, so uninstalling removes them.
