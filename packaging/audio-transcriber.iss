#ifndef AppName
  #define AppName "Audio Transcriber"
#endif
#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#ifndef BundleCompression
  #define BundleCompression "lzma2"
#endif
#ifndef AppId
  #define AppId "{{3E0FB1F0-E39B-47B2-86F8-A99EB0495727}"
#endif
#ifndef DesktopDir
  #define DesktopDir "{userdesktop}"
#endif
#ifndef BundleDir
  #define BundleDir "..\dist\Audio Transcriber"
#endif
#if !FileExists(BundleDir + "\Audio Transcriber.exe") || !FileExists(BundleDir + "\audio-transcriber-worker.exe")
  #error Build the complete PyInstaller bundle before compiling the installer.
#endif

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=derek-l8
AppPublisherURL=https://github.com/derek-l8/audio-transcriber
AppSupportURL=https://github.com/derek-l8/audio-transcriber/issues
AppUpdatesURL=https://github.com/derek-l8/audio-transcriber/releases
AppMutex=Local\AudioTranscriberRunning-v1
CloseApplications=no
RestartApplications=no
LicenseFile=..\LICENSE
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
OutputDir=..\dist\installer
OutputBaseFilename=audio-transcriber-{#AppVersion}-windows-x64
UninstallDisplayIcon={app}\Audio Transcriber.exe
Compression={#BundleCompression}
SolidCompression=yes

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: startup; Description: "Launch Audio Transcriber when I sign in"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{#DesktopDir}\{#AppName}"; Filename: "{app}\Audio Transcriber.exe"; Tasks: desktopicon
Name: "{group}\{#AppName}"; Filename: "{app}\Audio Transcriber.exe"
Name: "{group}\Getting started"; Filename: "{app}\_internal\help\GettingStarted.html"
Name: "{userstartup}\{#AppName}"; Filename: "{app}\Audio Transcriber.exe"; Tasks: startup

; Replace executables and shortcuts belonging to the previous product name.
#if AppName == "Audio Transcriber"
[InstallDelete]
Type: files; Name: "{app}\NPU Scribe.exe"
Type: files; Name: "{app}\npu-scribe-worker.exe"
Type: files; Name: "{userprograms}\NPU Scribe\NPU Scribe.lnk"
Type: files; Name: "{userprograms}\NPU Scribe\Getting started.lnk"
Type: files; Name: "{userstartup}\NPU Scribe.lnk"
#endif

[Run]
Filename: "{app}\Audio Transcriber.exe"; Description: "Launch Audio Transcriber"; Flags: postinstall nowait skipifsilent

; Uninstall removes tracked installation files only. Do not recursively delete
; {app}: a user may have placed a library or other personal files there.
