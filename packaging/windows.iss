; Install for the current user, including Python and all native libraries.
[Setup]
AppId={{85F79449-B3F8-4829-BDB0-9FBF88A7F624}
AppName=RAW Studio
AppVersion=3.0
AppPublisher=Krumqnkata
AppPublisherURL=https://github.com/Krumqnkata/python-raw-processor
DefaultDirName={localappdata}\Programs\RAW Studio
DefaultGroupName=RAW Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=RAWStudio-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\RAWStudio.exe
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\RAWStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RAW Studio"; Filename: "{app}\RAWStudio.exe"
Name: "{userdesktop}\RAW Studio"; Filename: "{app}\RAWStudio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RAWStudio.exe"; Description: "Open RAW Studio"; Flags: nowait postinstall skipifsilent
