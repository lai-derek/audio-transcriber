# Development

How the code is organized is in [Architecture](ARCHITECTURE.md); measured
choices are in [tested decisions](docs/TESTED_DECISIONS.md).

## Setup

Use 64-bit Python 3.14 (the pinned PySide6, OpenVINO, and sherpa-onnx builds do
not support 3.15 yet). From the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,desktop,inference]'
```

The `desktop` extra supplies Qt (including microphone capture), `inference`
supplies OpenVINO, OpenVINO GenAI, and sherpa-onnx, and `dev` supplies the test,
lint, and build tools. On Linux, use `python3 -m venv .venv` and `.venv/bin/python`.

## Checks

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest -m 'not live' --basetemp .pytest-dev-tests
```

The default suite uses synthetic media, a mock speech engine, and no network or
downloaded models; it blocks sockets. Use a new `--basetemp` folder if a previous
run's files are still in use. For coverage, add
`--cov=audio_transcriber --cov-report=term`.

The tests cover storage, chunking, checkpoints and resume, devices, the CLI,
exports, editing and revisions, AI cleanup checks, notes, summaries, saved
versions, the Trash, the model process, updates, model setup, and packaging
archives. `tests/test_desktop.py` and `tests/test_desktop_setup.py` drive the Qt
window offscreen with real child processes and need the `desktop` extra.

A live test downloads a pinned model and needs network access:

```powershell
$env:AUDIO_TRANSCRIBER_RUN_LIVE = '1'
.\.venv\Scripts\python.exe -m pytest -m live
Remove-Item Env:AUDIO_TRANSCRIBER_RUN_LIVE
```

Remove the variable afterwards; it also turns off the socket block for other tests.
Hardware, installer, and long-recording checks on Windows are in
[host validation](host-validation/README.md).

## Package checks

```powershell
.\.venv\Scripts\python.exe -m build --outdir .scratch/dist
.\.venv\Scripts\python.exe scripts/check_distributions.py .scratch/dist
```

The checker reads the wheel and source archive without extracting them, confirms
the required files, and rejects runtime data, environments, and local assistant
files. Use an empty output folder each time, since it expects exactly one of each.
CI then installs the wheel into a fresh environment and runs
`scripts/smoke_installed.py`, which checks both launchers, pause and resume, and
exports with the mock engine.

The Windows installer is a separate build; see the [packaging guide](packaging/README.md).

## CI

[CI](.github/workflows/ci.yml) runs the formatting, lint, type, and test checks on
Windows and Linux with Python 3.14 (the Qt window tests run on Windows only), then
builds and checks the packages. The workflow has read-only permissions and pins
its actions by commit. Dependabot proposes weekly dependency and action updates.

## Private data

Tests use synthetic media. Keep recordings, transcripts, models, logs, and build
output out of Git; `.scratch/` is ignored for local runs. Private evaluation
outputs belong outside the repository.

## Known gaps

- Real-device results come from one Windows x64 computer.
- Not yet checked: the installer wizard visually, a second computer, real
  microphone dictation across many apps, and recovery of unsaved typing after a
  forced exit.
- Not implemented: a tray icon, audible dictation cues, and a support bundle.
- AI cleanup, notes, and summaries are checked for concrete changes such as
  numbers, not for every change in meaning.
