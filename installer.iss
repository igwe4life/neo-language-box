; Inno Setup script for Neo Sermon.
; Wraps the PyInstaller bundle (dist/NeoSermon/) into a single .exe
; installer. The language models are NOT included - see [Files].
;
; Compile:  tools\innosetup\ISCC.exe installer.iss
; Output:   dist/NeoSermonSetup.exe
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
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=
; On a machine that already has the old install, this keeps using that folder
; (SermonTranslation) so a models folder already copied there stays valid.
; A fresh install gets the new NeoSermon folder.
UsePreviousAppDir=yes
CloseApplications=yes
RestartApplications=no
; The compressed payload is large (~1 GB); disable the 3-second on-screen
; "Ready to install" gate so users are not left staring at a spinner.
DisableReadyPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

[Files]
; The PyInstaller-produced folder, verbatim.
Source: "dist\NeoSermon\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Models are NOT bundled. With fourteen languages they come to ~5.8 GB, which
; makes the installer unusable to move around - and they change far less often
; than the code, so re-shipping them with every build wastes hours of copying.
; They are copied to {app}\models once per machine instead; desktop_app.py
; checks for them at startup and says exactly what is missing.
; Application vocab files (already inside the bundle's _internal/, but
; also placed at the app root so `SERMON_VOCAB_FILE` overrides work).
Source: "vocabulary.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "phrases.yo.txt";  DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Created empty so the destination for the model folder is obvious, and so
; the uninstaller knows about it.
Name: "{app}\models"

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
; Runtime folders the app creates alongside itself
Type: filesandordirs; Name: "{app}\certs"
Type: files; Name: "{app}\models-verified.json"
; Do NOT delete models/ - the user might reuse them for a reinstall
