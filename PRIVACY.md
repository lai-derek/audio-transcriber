# Privacy and local data

Transcription, AI cleanup, dictation, playback, editing, and export operate locally.
The application has no telemetry, analytics, crash upload, or cloud speech API.
Default tests block network sockets, including tests of ordinary CLI commands.

## Library and recordings

The default library is `%LOCALAPPDATA%\audio-transcriber` on Windows. Override it with
`--data-dir` or `AUDIO_TRANSCRIBER_DATA_DIR`. Models can use a separate `--model-root`.
The app copies imported recordings into the library and retains transcripts,
manual-edit revisions, AI cleanup history, settings, checkpoints, and exports there.
Cleanup compilation caches are stored under `cache/cleanup` in the library.
Selected models remain loaded in memory while the app is open, including when
minimized. Exiting stops the model worker and releases that memory.

Choose a folder outside the repository. The app does not prevent an override
from pointing into Git or a synced directory, and it does not yet warn about
OneDrive redirection. Files in a synced library can be uploaded by the sync
service independently of Audio Transcriber. Local files use ordinary filesystem
permissions; the application does not encrypt them.

## Dictation

The microphone is opened only when recording starts through its button or an
enabled shortcut, and stops when you stop, release a hold shortcut, cancel, or
reach the two-minute limit. Closing the dictation window disables its shortcut.
The app does not listen while idle.

Dictation recordings and worker sessions are kept under `dictation/jobs` and
`dictation/engine`; raw and cleaned recovery copies are in `dictation/history`.
Temporary native PCM is removed after successful conversion to WAV. Failed or
cancelled capture can leave its PCM for recovery. These files are retained until
you remove them with the app and worker stopped. There is no automatic retention
limit or deletion UI yet.

Windows UI Automation checks the focused field's process, editability, password
flag, and field identifier. It does not read that app's text. Automatic insertion
sends Unicode input without using the clipboard. **Copy text** writes the selected
recovered text to the clipboard when you request it. Dictation diagnostics may
contain transcript text and local paths; review them before sharing.

## Downloads and update checks

- **Manage models** and `models download` request approved files from `huggingface.co`. The HTTP
  client can follow redirects to the hosting service's download/CDN endpoints.
  Installed files must match the manifest's checksums and sizes.
- `evaluate` reads a supplied manifest and fetches its declared audio using
  HTTPS or a local `file:` URL. HTTPS evaluation sources are not restricted to
  Hugging Face. Review the manifest and source before running it.

- Installed apps check GitHub for new releases at startup, at most once a day.
  Disable automatic checks in **Help → Check for updates**. Clicking **Update**
  downloads an installer from GitHub; no recordings or transcript content is sent.
  Downloaded installers and update logs are cached separately from your library.

These actions request downloads; they do not intentionally upload your lecture
recordings or transcripts. Remote services receive the normal request metadata,
including the requesting IP address. Ordinary transcription and cleanup do not download
missing models automatically.

## Sharing diagnostics and deleting data

Console messages and operating-system errors can contain local paths or source
filenames. A failed desktop start saves `startup-error.log` in the library.
Review these diagnostics before sharing. Host validation tools sanitize their
published reports, but inspect any report for private paths, audio, transcripts,
and credentials before copying it into Git or an issue.

Right-click a lecture and choose **Delete** to move its recording, transcripts,
edits, and internal exports into **File → Trash**. It can be restored for 30 days;
expired items are removed when the app starts. **Empty Trash** permanently
removes the deleted lectures immediately. Exported copies elsewhere remain separate.

Close the desktop and ensure its worker has stopped before manually managing or
backing up library files. Uninstalling keeps the library, including recordings,
models, dictation history, and Trash.
