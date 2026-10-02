; Inno Setup script for Neo Sermon.
; Wraps the PyInstaller bundle (dist/NeoSermon/) AND the language models into
; a single .exe installer.
;
; Build, in order:
;   python -m PyInstaller --clean --noconfirm desktop.spec
;   python scripts/build_models_ship.py --clean
;   tools\innosetup\ISCC.exe installer.iss
; Output:   dist/NeoSermonSetup.exe
;
; SIZE LIMIT - read before adding models
; --------------------------------------
; Inno Setup cannot put more than 4,200,000,000 bytes of compressed data into
; a SINGLE Setup.exe. The Sept 2026 build (14 languages) was already at
; 3.62 GB compressed with ~575 MB of headroom left. The Oct 2026 wave (11 more
; languages, 25 total) pushed well past that, so DiskSpanning is now ON:
; Setup.exe plus numbered disk1.bin, disk2.bin, ... slices (DiskSliceSize=max
; keeps the slice count as low as possible). All of them must ship and be
; extracted together - see the [Run]/output note below. If a future language
; wave makes even this unwieldy, the remaining lever is dropping the fallback
; speech engines (parakeet-110m, whisper base.en: 337 MB).

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

#define AppName        "Neo Sermon"
#define AppShortName   "NeoSermon"
#define AppVersion     "0.8.0"
#define AppPublisher   "Neo Sermon"
#define AppExeName     "NeoSermon.exe"

[Setup]
; DiskSpanning: see the SIZE LIMIT note at the top of this file. "max" keeps
; the slice count as low as possible rather than aiming for literal floppies.
DiskSpanning=yes
DiskSliceSize=max
; Unchanged across the rename from "Sermon Translation Speaker". Windows keys
; an installed program on this GUID, so keeping it makes this installer an
; UPGRADE of an existing install rather than a second app beside it.
AppId={{A9C8E7B2-4F3D-4A8C-9E12-8F5B2C4D7E19}
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
OutputBaseFilename=NeoSermonSetup
OutputDir=dist
; Compression is required for size (see the limit above), but these models are
; already quantised and only shrink 6-20% at any setting, so /max buys almost
; nothing over the default level while costing a great deal of build time.
; Non-solid keeps memory use sane across 4 GB of input.
Compression=lzma2
SolidCompression=no
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=
; On a machine that already has the old install, this keeps using that folder
; (SermonTranslation) so a models folder already copied there stays valid.
; A fresh install gets the new NeoSermon folder.
UsePreviousAppDir=yes
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
Source: "dist\NeoSermon\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
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

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; The executable was renamed with the app. An upgrade over an older install
; would otherwise leave the old SermonTranslationSpeaker.exe beside the new
; one - still launchable, and running stale code.
Type: files; Name: "{app}\SermonTranslationSpeaker.exe"

[UninstallDelete]
; Runtime files the app creates alongside itself. sermon/tls.py writes the
; self-signed certificate two levels above itself, which in the installed app
; is _internal\certs - cleaning only {app}\certs left the key and certificate
; behind after uninstall.
Type: filesandordirs; Name: "{app}\_internal\certs"
Type: filesandordirs; Name: "{app}\certs"
Type: files; Name: "{app}\models-verified.json"
; The bundled models are installer-owned files, so uninstalling removes them.
