# Architecture

Audio Transcriber is one Python 3.14 application with two front ends: a PySide6
desktop app and a command-line interface. Both use the same processing pipeline
and library format. Speech recognition runs locally through sherpa-onnx (the
default Parakeet model) or OpenVINO GenAI (Whisper models); AI cleanup, notes, and
summaries use a local Qwen2.5 7B model through OpenVINO GenAI.

A single Python application keeps one runtime and one test suite. C# or Tauri front
ends with a Python inference service were considered during planning, but would add
a second runtime and communication between processes; C++/Qt would need more native
code. Measured model, device, and default choices are in
[tested decisions](docs/TESTED_DECISIONS.md).

## From recording to transcript

```
import WAV/MP3/M4A/MP4
  └─ validate the file, copy it into the lecture folder, record its SHA-256
decode
  ├─ mono 16 kHz PCM WAV (checked from its header) is read directly
  └─ anything else, including stereo or 44.1/48 kHz WAV, goes through FFmpeg
     (bundled in installers; on PATH or chosen for source installs) with a fixed
     argument list, no shell, and a private temporary folder
transcribe in cuts (only one cut in memory at a time)
  ├─ Parakeet: 28-second cuts placed at pauses, CPU, two threads
  └─ Whisper: fixed 30-second cuts on CPU, GPU, or NPU
  after each cut: convert to absolute times, validate, save a checkpoint
finish
  ├─ Original (raw) transcript, written once and never changed
  ├─ Balanced: deterministic rule-based tidy of Original
  └─ optional AI steps: cleanup → notes → summary
```

Speech starts on CPU. **Automatic** device choice benchmarks devices on synthetic
audio and tries the next working device if one fails; a manual choice never
switches. Each cut records the device that produced it.

## Versions of a transcript

Every lecture keeps separate versions, so a later step never overwrites an
earlier one:

- **Original** (raw) and **Balanced** are written once.
- **Edited** holds typed corrections to Original as numbered revisions with the
  source hash; a stale editor cannot overwrite a newer revision.
- **Transcript** (AI cleanup), **Notes**, and **Summary** each keep the latest
  complete result plus a history of earlier ones. Results for each cleanup level
  and notes style are saved, so switching between them is instant.
- Typing in Transcript, Notes, or Summary saves the user's own copy, tied to the
  AI result it was typed on.
- Deleting a lecture moves its whole folder to the Trash for 30 days.

## AI text steps

Each step sends the text in bounded blocks to the local text model and checks
the result before keeping it.

- **Cleanup** (Light, Medium, Heavy; lecture or dictation mode) edits wording.
  Checks for changed numbers, negations, hedges, and large deletions keep the
  source text for a failing block. AUTO tries the GPU, then the CPU.
- **Notes** (Mixed or Mostly structured) condense each block into topics and
  points. A line giving a number the recording never said, in digits or words,
  is marked "⚠ check this number". Mostly prose needs no model.
- **Summary** summarizes parts of the cleanup (or Original before cleanup exists)
  and merges them in rounds into one overview in the chosen layout.

Requests within a job are sent together; on the GPU the model also drafts tokens
from the input itself (prompt lookup). Checks catch concrete changes such as
numbers, not every change in meaning, so AI text still needs review.

## Desktop process model

The desktop window starts one **model process** (`resident.py`, driven by
`desktop_worker.py`). It loads the selected speech and text models when the app
opens and keeps them loaded while the app runs or is minimized; jobs and dictation
run in it one at a time. Closing the app stops the process and releases its
memory. If the window process dies, the model process sees its input close and
exits. Jobs check a stop file between cuts or text blocks, so **Pause** stops
safely and **Resume** continues from the last checkpoint.

Live dictation (`desktop_dictation.py`) records the microphone with Qt, converts
the recording to mono 16 kHz WAV, and runs speech and cleanup in the same model
process. The text is saved to the dictation history before Windows Unicode input
inserts it (`windows_dictation.py`); a changed focus, password field, or failed
insertion leaves the saved copy to copy manually. No keyboard hook is installed.

First-run setup (`desktop_setup.py`) downloads the models. Model downloads
(`acquisition.py`) use a pinned manifest, an HTTPS host allowlist, staging, and
per-file size and SHA-256 checks before installing. In-app updates
(`updates.py`, `update_helper.py`) download a GitHub release installer, verify
its SHA-256, wait for current work, and run a separate updater that reinstalls
and reopens the app.

## Recovery

- Interrupted transcription resumes only on request, and only if the source
  fingerprint, model identity, decoder, and cut settings all match the checkpoint.
- An interruption before the first checkpoint, between cuts, or after the last cut
  resumes from the latest saved state; resumed runs give the same Original text as
  uninterrupted ones in the exercised checks.
- Cancelled or failed AI steps leave the previous complete result in place.
- All metadata is written to a temporary file and renamed into place.

## Modules

Code is in [`src/audio_transcriber`](src/audio_transcriber).

| Area | Modules |
|---|---|
| Records and storage | `models.py` (records), `storage.py` (library, atomic writes, path checks), `editing.py` (Edited revisions), `versions.py` (saved AI results, user copies), `trash.py` |
| Speech | `media.py` (decoding), `chunking.py`, `checkpoint.py`, `engines.py` (sherpa-onnx and OpenVINO engines), `devices.py` (benchmarks), `pipeline.py` (`BatchRunner`) |
| Text | `cleanup.py` (Balanced rules), `dictionary.py`, `ai_cleanup.py` (text model, prompts, checks), `formatting.py` (notes), `summary.py` |
| Output | `export.py` (text, Markdown, SRT, JSON) |
| Models and updates | `acquisition.py`, `updates.py`, `update_helper.py` |
| Desktop | `desktop.py` (launcher), `desktop_ui.py` (main window), `desktop_player.py`, `desktop_editing.py`, `desktop_versions.py`, `desktop_trash.py`, `desktop_setup.py`, `desktop_worker.py` |
| Model process | `resident.py`, `cli.py` (commands, also run inside the model process) |
| Dictation | `desktop_dictation.py`, `dictation.py`, `windows_dictation.py`, `insertion.py`, `config.py` |
| Evaluation | `evaluate.py`, `metrics.py` |

To follow a feature into its tests, search both folders, for example
`rg 'save_transcript' src tests`.

## Library layout

```
<library>/lectures/<id>/
    session.json               status and diagnostics
    source/original.<ext>      private copy of the import, with source.json (hash)
    checkpoint.json            progress while transcribing
    raw-transcript.json        Original, written once
    balanced-transcript.json   Balanced, written once
    edits/                     Edited revisions
    ai-transcript.json         latest cleanup, history in ai-cleanup-history/
    formatted-transcript.json  latest notes, history in formatting-history/
    summary-transcript.json    latest summary, history in summary-history/
    *-user.json                the user's typed copies
    exports/                   exports, never silently overwritten
<library>/dictation/           dictation recordings, jobs, and history
<library>/trash/               deleted lectures, kept for 30 days
<library>/models/<model-id>/   verified model installs
<library>/cache/cleanup/       compiled text-model caches
```

The default library is `%LOCALAPPDATA%\audio-transcriber` (from `platformdirs`);
`--data-dir` or `AUDIO_TRANSCRIBER_DATA_DIR` overrides it. See [Privacy](PRIVACY.md).

## Packaging

PyInstaller builds the desktop app, the model-process worker, and the updater into
one folder, and Inno Setup installs it per user without administrator rights. The
installer bundles Python and FFmpeg; models download on first launch. See the
[packaging guide](packaging/README.md).
