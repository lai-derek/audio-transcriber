# Command-line guide

Run these PowerShell commands from the source folder after the
[Windows setup](USER_GUIDE.md#install-on-windows). Replace example input paths and
session IDs with your own. In a new PowerShell window, set the variables again.
Global `--data-dir` and `--model-root` options go before the subcommand.

## Transcription and export

These examples use `parakeet-tdt-0.6b-v2-int8`, the default speech model. Download
it in **Manage models** first, or set `$Model` to the model you installed.
The examples use the environment under the source folder. If you run from
another directory, use the full path to its Python executable followed by
`-m audio_transcriber.cli`. Activating the environment is optional.

A mono 16 kHz WAV is read directly. MP3, M4A, MP4, and any other WAV, such as
stereo, 44.1 or 48 kHz, 24-bit, or floating point, are converted with FFmpeg.
Windows installer builds include it; source installations use one on PATH (see
[setup](USER_GUIDE.md#install-on-windows)) or one passed with `--ffmpeg`:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Model = 'parakeet-tdt-0.6b-v2-int8'
$InputFile = 'C:\Lectures\lecture.wav'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" transcribe "$InputFile" --model "$Model" --device CPU --cleanup off
if ($LASTEXITCODE -ne 0) { throw 'Transcription did not finish; check the message above.' }
```

The output contains `session: ...` and `status: ready` on completion. Copy the
session identifier into `$Session` below. Choose `raw` or `balanced`; `edited`
requires saved corrections and `ai` requires completed cleanup. Saved `formatted`
and `summary` versions support text, Markdown, and JSON, without SRT subtitles:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Session = 'SESSION_ID_FROM_OUTPUT'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" export "$Session" --format srt --layer balanced
if ($LASTEXITCODE -ne 0) { throw 'Export failed; check the message above.' }
```

The export is stored under `$Data\lectures\$Session\exports`. Existing exports
are refused unless you explicitly add `--overwrite`. Available formats are
`json`, `markdown`, `text`, and `srt`.

To use an FFmpeg executable that is not on PATH, add `--ffmpeg` to the run:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Model = 'parakeet-tdt-0.6b-v2-int8'
$FFmpeg = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
$InputFile = 'C:\Lectures\lecture.mp4'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" transcribe "$InputFile" --model "$Model" --device CPU --ffmpeg "$FFmpeg" --cleanup off
```

The library keeps the file exactly as imported; only a temporary converted copy
is transcribed.

### Pause and resume from the CLI

Add `--stop-file PATH` when starting a run. Creating that file requests a safe
pause; use a second PowerShell window to create it. Remove the marker before
resuming. Exit code 130 indicates a safe pause, 1 an operational error, and 2
invalid arguments.

Resume a paused/failed session with the **same model** and library. For compressed
media, supply its FFmpeg executable again. The saved chunk plan is reused:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Session = 'SESSION_ID_FROM_OUTPUT'
$Model = 'parakeet-tdt-0.6b-v2-int8' # Replace with the model originally used.
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" resume "$Session" --model "$Model" --device CPU --cleanup off
```

If the desktop was forcibly terminated, confirm its worker has stopped before
using CLI resume on a session still marked `processing`. Resume refuses changed
preserved audio or mismatched model identity. Completed sessions cannot resume;
start a new import to transcribe again.


## AI cleanup

Download the cleanup model into the same library/model root:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" models download qwen2.5-7b-instruct-int4-ov
if ($LASTEXITCODE -ne 0) { throw 'Cleanup model download failed; stop here.' }
```

CLI `transcribe` and `resume` default to Medium cleanup and Mixed
formatting. Use `--cleanup light` or `--cleanup heavy` to change the level, `--formatting prose`
for paragraphs, or `--cleanup off` for unedited text. Dictation mode defaults
to Medium cleanup and Mostly prose. Explicit options override these defaults.
Cleanup defaults to AUTO (GPU, then CPU). Use `--cleanup-device CPU`, `GPU`, or `NPU` to require that device.
Cleanup failure returns exit 1 while leaving a ready, exportable transcription.

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" transcribe 'lecture.wav' --device CPU --cleanup light
```

To clean an existing session and export it:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Session = 'SESSION_ID_FROM_OUTPUT'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" cleanup "$Session" --device AUTO --mode lecture --style light
if ($LASTEXITCODE -ne 0) { throw 'Cleanup did not finish; check the message above.' }
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" export "$Session" --layer ai --format markdown
```

For an existing UTF-8 text file, run from the source folder and replace the
input path. This uses the same downloaded model and library:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" cleanup-text 'draft.txt' --output 'cleaned.txt' --mode dictation --style light --device CPU
```

Use `--format json` for settings, source hash, and per-block warnings. Existing
outputs require `--overwrite`, and the input file cannot be the output. Global
`--data-dir` / `--model-root` options go before the subcommand. No transcription
session or audio decoder is required for text cleanup.


## Study notes

CLI example after cleanup:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Session = 'SESSION_ID_FROM_OUTPUT'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" format "$Session" --source ai --style mixed
if ($LASTEXITCODE -ne 0) { throw 'Formatting did not finish; check the message above.' }
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" export "$Session" --layer formatted --format markdown
```

Use `--source raw --style prose` without the text model. Add `--formatting prose`,
`mixed`, or `structured` to `transcribe`, `resume`, or `cleanup` to run the stage
automatically; a chained job reuses its loaded model. Notes condense about 3,000
source characters at a time; a line giving a number the recording never said ends
with "⚠ check this number".

Export Formatted as text, Markdown, or JSON; use Raw/Edited for SRT subtitles.
JSON includes the source layer and hash, settings, and per-block warnings.
Completed versions are retained in `formatting-history`; cancelled or failed jobs
keep the previous complete version.


## Summaries

CLI equivalent (`cleanup` and `transcribe` also accept `--summary on` or `off`):

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Session = 'SESSION_ID_FROM_OUTPUT'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" summarize "$Session"
if ($LASTEXITCODE -ne 0) { throw 'Summary did not finish; check the message above.' }
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" export "$Session" --layer summary --format markdown
```


## Alternate model folder

Both launchers accept `--model-root`. That folder contains model-ID subfolders,
not a single model's files. Use the same location for download and launch:

```powershell
$Data = Join-Path $env:LOCALAPPDATA 'audio-transcriber'
$Models = 'D:\Audio Transcriber Models'
.\.venv\Scripts\python.exe -m audio_transcriber.cli --data-dir "$Data" --model-root "$Models" models download parakeet-tdt-0.6b-v2-int8
if ($LASTEXITCODE -ne 0) { throw 'Model download failed; stop here.' }
.\.venv\Scripts\python.exe -m audio_transcriber.desktop --data-dir "$Data" --model-root "$Models"
```

The desktop saves a model folder or FFmpeg program in the library only when you
choose one; otherwise it uses the library's `models` folder and discovers FFmpeg
on PATH (or uses the bundled decoder in an installer build).
If a saved location no longer exists, the app says so and uses the
default. Command-line paths override saved settings.
