# Using Audio Transcriber

For the public source, start with [Install on Windows](#install-on-windows).
If you have a tester installer, follow [Using the Windows installer](#using-the-windows-installer).
Then [import and review a recording](#desktop-workflow), or use
[live dictation](#live-dictation). Processing works offline after model download.

## Install on Windows

Use [64-bit Python 3.14](https://www.python.org/downloads/windows/) and PowerShell.
Select **Python 3.14 → Windows installer (64-bit)** on that page and run the installer.
Check availability with `py -3.14 --version` before continuing. Windows x64 is the
locally tested target.

1. Get the source using **Code > Download ZIP** on GitHub, or clone it with Git.
   Extract or clone into a normal local folder outside OneDrive or other cloud
   sync. Open PowerShell in the folder containing `pyproject.toml`.

   If the project is already cloned, open that folder and continue at step 2.
   For a new Git checkout, this creates the folder under your local Projects directory:

   ```powershell
   & {
       $Projects = Join-Path $env:USERPROFILE 'Projects'
       New-Item -ItemType Directory -Force -Path $Projects -ErrorAction Stop | Out-Null
       Set-Location $Projects -ErrorAction Stop
       git clone https://github.com/lai-derek/audio-transcriber.git audio-transcriber
       if ($LASTEXITCODE -ne 0) { throw 'Clone failed; stop here.' }
       Set-Location audio-transcriber -ErrorAction Stop
   }
   ```

2. Install the engine in its own Python environment:

   ```powershell
   & {
       py -3.14 -m venv .venv
       if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed; stop here.' }
       .\.venv\Scripts\python.exe -m pip install '.[desktop,inference]'
       if ($LASTEXITCODE -ne 0) { throw 'Installation failed; stop here.' }
   }
   ```

3. Install FFmpeg, which the app needs to read MP3, M4A, and MP4 files and WAV
   files that are not already mono 16 kHz, such as stereo recordings from phones.
   Windows Package Manager puts FFmpeg on PATH, where the app finds it:

   ```powershell
   winget install --exact --id Gyan.FFmpeg
   ```

   Then close PowerShell and open a new window in the project folder so the new
   PATH applies. You can also download a build from the
   [FFmpeg site](https://www.ffmpeg.org/download.html) and choose its `ffmpeg.exe`
   later under **Settings → FFmpeg program**.

4. Open the desktop app:

   ```powershell
   .\.venv\Scripts\python.exe -m audio_transcriber.desktop
   ```

   First launch downloads the speech models (about 890 MB), then the AI cleanup
   model (about 4.5 GB); leave several more GB free for downloading and caches.
   **Manage models** shows progress, and you can stop setup and download
   individual models later. Transcription can start once a speech model is ready.

Run each step only after the previous one succeeds; no PowerShell activation
script is needed. The install command copies the program into `.venv`, so after
changing or updating the source files, repeat step 2.

## Using the Windows installer

Installer builds target Windows x64 and include Python, the desktop app, and
FFmpeg for compressed audio/video. A public binary has not been published yet;
the [source route](#install-on-windows) is available now. Once published,
download the Windows `.exe` from [GitHub Releases](https://github.com/lai-derek/audio-transcriber/releases).

1. Run the installer. It installs for your user without administrator access and
   offers a desktop shortcut. Launch at sign-in is optional and starts unchecked.
2. The app opens after installation and starts downloading every model: the
   three English speech models (Parakeet v2, the default, about 660 MB; then
   Whisper Small and Whisper Base, about 230 MB together), then the AI cleanup
   model (about 4.5 GB). **Manage models** shows each download's size and progress.
   You can close that window; downloads continue in the background. Leave several
   extra GiB of free disk space for download staging and runtime caches.
3. Once a speech model is ready, press **Import audio**, review the text, and
   choose **Export**. MP3, M4A, and MP4 decoding is included. When the cleanup
   model finishes, Medium cleanup and Mixed formatting turn on.

If setup is interrupted, it continues where it stopped the next time the app
opens. **Stop download** ends automatic setup; later, check models in **Manage
models** (main window or Settings menu) and click Download to continue. AI cleanup
needs several GiB of RAM while running; without its model, cleanup and formatting
stay **Off**. Downloads need internet access; processing works offline afterward.

If a downloaded model is incomplete or damaged, select it in **Manage models**
and click **Download** again. A replacement is downloaded and verified before the
old files are moved aside. Previous files stay in a `.previous-…` folder beside the
model; you can remove that backup in File Explorer after checking the repaired model.

The installer is unsigned. Windows may show an unknown-publisher warning.
Obtain installers from this project's releases. Your library normally stays in
`%LOCALAPPDATA%\audio-transcriber`; installation, updates, and uninstall keep it.
Existing libraries from before the rename are reused automatically.

## Desktop workflow

Open the installed desktop shortcut, or use the
[source setup](#install-on-windows).

Press **Import audio** and choose a WAV, MP3, M4A, or MP4 file. The speech model
and device are under **Settings → Speech model** and **Speech device**. Parakeet v2
is recommended and always runs on the CPU, so the device setting is unavailable
while it is selected. For Whisper Small and Whisper Base, CPU is initially
selected; Automatic benchmarks available devices and can try other successful
benchmark devices after inference fails; a manual choice stays on that device.
**Settings → Restore recommended settings** resets processing choices.
A mono 16 kHz WAV is read directly; other WAV recordings, such as stereo or
44.1/48 kHz, are converted with FFmpeg like MP3, M4A, and MP4 files, and the
imported file is kept unchanged. Installer builds include FFmpeg. Source installations use FFmpeg from PATH
([setup step 3](#install-on-windows)) or the executable chosen under
**Settings → FFmpeg program**, which takes precedence. **File → Open library folder** shows where recordings and
transcripts are stored.

**Pause** appears while a job runs and stops it after the current 30-second part
(or block of text, for AI steps). A lecture that stopped partway, by Pause, an
error, or the app closing, shows **Paused** or **Stopped** in the library, and
**Resume** appears when you select it. Transcription continues where it stopped,
using the lecture's saved model and chunk strategy; keep that model installed.
A paused AI step starts that step again; earlier versions are kept.

Only one desktop window can open a given library. Avoid using the CLI on the
same session while its worker is active.

### Review, correct, and export

A lecture opens on its most finished version. Tabs show only the versions it
has: **Original** (the speech model's output, with start times), **Edited** (your
corrections), **Transcript** (AI cleanup), **Notes** (formatted), and **Summary**.
Balanced, a rule-based tidy of Original, is still saved and can be exported from
the CLI. Search any version and play the preserved recording. On
Original or Edited, click a line to move the audio there; the line being played
is shaded. Switching tabs keeps the playback position. MP4 playback is audio
only. Hover over the lecture's title line to see which models and devices
produced it. **Copy all** copies the version shown (formatted when pasted into a
document), and **Export** saves it as text, Markdown, subtitles, or data;
subtitles need Original, Edited, or an unedited Transcript, which keep timing.

Type directly in any tab to correct it; edits save a moment after you stop typing.
On **Original**, keep one paragraph per time stamp: changed lines are saved as a
separate **Edited** version, and later edits continue there. **Revision history**
shows earlier edits. On **Notes**, **Summary**, and **Transcript**, your text is
saved as your own copy of that version and exported in its place; **Undo my
edits** shows the AI version again, and your copy stays in history. If you switch
to another cleanup level or note style, your copy is kept but hidden, because it
was typed on a different version.
Original and Balanced stay unchanged. Timestamps stay fixed for subtitles and
playback.
Right-click a line on Original or Edited to mark it unclear or clear; this is
saved like an edit, and the line's original model confidence is cleared. Empty
text retains its interval but omits that SRT cue.

Save failures keep the draft open, and stale editors cannot overwrite a newer
revision. Unsaved drafts are lost if the process is forcibly closed. JSON
exports include manual-edit attribution, revision ID, and source layer/hash.
Text, Markdown, and SRT exports contain the selected layer's text.

### Library and shortcuts

- Drag audio files onto the window, or pick several in **Import audio**; they
  import one after another.
- Right-click a lecture to rename it, show its recording in File Explorer, export
  it, or delete it (the **Delete** key also works).
- Deleted lectures go to **File → Trash**, where **Restore** puts them back. They
  are removed permanently after 30 days, or right away with **Empty Trash**. The
  app never moves lectures to the Trash on its own. A search box appears above the library once it holds more than
  eight lectures.
- **Ctrl+F** jumps to the search box; **Enter** or **F3** finds the next match.
- **Ctrl+Space** plays or pauses the recording.
- **Ctrl + mouse wheel**, or **Ctrl +** / **Ctrl -** (**Ctrl 0** resets), changes
  the text size.
- The window size, library width, text size, and the last lecture and tab you had
  open are restored next time.
- When a long job finishes while the window is minimized or in the background,
  the taskbar button flashes and Windows shows a notification.

## Live dictation

Launch the desktop, then open **Live dictation**. Download a speech model using
the installation steps above. For cleanup, also download the
[cleanup model](#local-ai-cleanup); Off needs only the speech model. FFmpeg is
not needed for microphone recordings.

1. Select your microphone and speech model. Speech runs on CPU (Parakeet v2 always
   does; the Whisper models can use another device); cleanup starts
   on AUTO (GPU, then CPU). With its model installed, cleanup starts at Medium;
   otherwise it starts Off. Dictation uses Mostly prose formatting and saves its own settings.
2. Choose **Toggle** or **Hold to talk**, enter a shortcut such as
   **Ctrl+Alt+Space**, and tick **Enable global shortcut**. If it is already in
   use, choose another. Disable it before changing its key or mode.
3. Focus an editable text field in another app. Press the shortcut, wait for
   **Recording**, then speak. The input meter should move. In Toggle mode, press
   again to stop; in Hold to talk, release the shortcut. Recording stops after
   two minutes at most.
4. Keep the same field focused while transcription and cleanup finish. The result
   is saved before insertion. **Recovered text** holds recent results;
   **Show unedited transcript** selects the original recognition output.
   **Copy text** deliberately replaces the clipboard when you need to paste manually.

**Record a copy** records from the window and saves text without automatic
insertion. **Cancel** stops capture or requests that the worker stop; model
compilation may need to finish first. Very quiet recordings are rejected. Check
Windows microphone permissions, mute, and input volume if the meter stays flat.

The shortcut works while the window is open, including when the main app is
minimized. Closing the dictation window disables it; enable it again after a
new app launch. There is no tray; the installer offers optional launch at sign-in.
Dictation and file processing run one at a time.

Text appears after you stop recording. Selected models load when the app opens
and stay loaded while it is open, including when minimized. Closing the app
releases their memory. Cleanup can take substantially longer than Off. Use Off for
faster results. This version has no live partial transcript. Insertion uses
Unicode keystrokes and leaves the clipboard unchanged. Password fields are
rejected; elevated apps and custom editors may reject insertion. Review the
result in the destination app and use the recovered copy if needed.

Recordings, raw text, cleaned text, and settings are kept under
`dictation` in the local library. They remain until you remove them; see
[Privacy](../PRIVACY.md). Measured dictation speed and memory are in
[tested decisions](TESTED_DECISIONS.md#live-dictation-latency-and-idle-memory).
Broader microphone and app checks remain open.

## Local AI cleanup

Cleanup uses an optional local text model. Download it through **Manage models**.

The verified download is about 4.5 GB; leave room for staging, runtime caches,
and model memory. No model weights are committed or bundled with the program.
After downloading, ordinary cleanup uses local files without an API key or
network connection.

In the desktop, choose **Off**, **Light**, **Medium**, or **Heavy** under
**Settings → Cleanup for new imports**. Cleanup
starts Off when its model is missing. With the model installed, new libraries
start with Medium cleanup and Mixed formatting. A completed import or
resume automatically runs the selected cleanup; Off skips cleanup. If cleanup fails, the
transcript stays ready and can be exported. Download the model and use **Clean up** and pick
a level, or **Redo all** (bottom of either menu) to make the shown cleanup, notes,
and summary again; that does not change the setting for new imports. After Redo
all, the background versions below are remade too.

After an imported lecture is cleaned up, the app prepares more versions in the
background so switching to them is instant. Choose which under
**Settings → Prepare in the background**:

- **Other note styles** (on by default): the cleanup's notes in each style.
- **Other summary styles** (on by default): the cleanup's summary in each style.
- **Other cleanup levels** (off by default): the other cleanup levels too, each
  with the note and summary styles chosen above. This keeps the AI model busy
  much longer.

Turn all three off to prepare nothing extra. Anything you start yourself,
including dictation, runs first: background work pauses after its current block
of text and continues later. It also runs on battery. Cleanup has its own device
setting under **Settings → AI device**.
Cleanup is saved as the **Transcript** tab. When formatting is enabled, the
desktop then opens **Notes**; switch to Transcript to review the cleanup alone.
Cleanup always reads Original, even if another tab is visible. Export Notes for
the chosen layout or Transcript for cleanup alone. Search and playback remain
available; clicking a line does not move the audio on AI versions because their
paragraphs are not aligned to words.

- **Lecture** prioritizes retaining detail, technical wording, corrections, and
  uncertainty. Words such as “period” are treated as content.
- **Dictation** attempts to remove false starts, resolve explicit self-corrections,
  and interpret spoken punctuation, “new paragraph,” and list commands. These
  edits can be imperfect; review wording that resembles a formatting command.
- **Off** skips AI cleanup; **Raw** always remains available as unedited text.
- **Light** removes hesitation sounds, stutters, repeated words, and filler such
  as "you know", and fixes grammar. Otherwise it keeps the speaker's wording.
- **Medium** also removes false starts and turns broken or run-on sentences into
  clear ones with little rewording.
- **Heavy** rewrites for easy reading: it removes all filler and repeats, states
  a repeated point once, and rephrases tangled sentences, keeping every point,
  example, name, number, and qualification. It is usually noticeably shorter.

Each cleaned block is checked against its source. A block that changes a number
or makes an excessive deletion or addition keeps its source text and is marked
for review. Changes to negation or hedges such as "I think" reject Light cleanup;
Medium and Heavy can retain those changes with a review warning.

Each completed cleanup keeps a snapshot in `ai-cleanup-history`; the latest
version appears in **Transcript**. JSON exports include settings and block warnings.
Raw, Balanced, Edited, and the recording stay available.

The model can change names, numbers, or meaning. Suspicious edits retain the
source text with an uncertainty flag. Review important results against Raw/audio.
AI subtitles use coarse source-block timing; use Raw or Edited for precise cues.

AUTO tries a GPU exposed by OpenVINO; if GPU loading or generation fails, it
retries the current block on CPU and stays there for the job. JSON records the
requested and successful devices, per-block devices, and any fallback reason.
Explicit CPU/GPU/NPU selections fail rather than switching; driver/model support varies. First use includes compilation and can be slow. Stop/Pause requests are checked between blocks and during token generation;
compilation must finish first. Cancelled/failed runs retain the previous complete
AI version and do not resume partial output: rerun cleanup to start again.

## Formatted study notes

Formatting is a separate layer after transcription and optional cleanup.
Choose **Mostly prose**, **Mixed**, **Mostly structured**, or **Off** under
**Settings → Formatting for new imports**. The desktop starts with Mixed. With cleanup enabled, formatting
reads the new AI version; with cleanup Off, it reads Raw.

- **Mostly prose** groups the text into paragraphs without loading an AI model.
- **Mixed** condenses the text into short paragraphs under a heading for each
  topic, with bullets where the speaker listed things.
- **Mostly structured** condenses it into an outline: a heading for each topic,
  one short bullet per point, and indented bullets for key details.
- **Off** skips this stage. Raw and AI remain available independently.

Notes keep each point and its key details and drop filler, repetition, and
asides; they are usually about a third as long as the AI version in Mostly
structured. They are in the model's own words, so they can misstate who said or
wants what, or what was meant; check anything important against the AI version
or the recording. A line giving a number the recording never said, in digits or
in words, ends with "⚠ check this number". If the model's answer is empty or runs
away, that part keeps the complete source text instead.

For a completed session, show Original or Transcript and choose
**Format**, then a style. The result appears in **Notes** with
its own history.

Notes export as text, Markdown, or JSON. Use Original or Edited for subtitles.
Completed versions are retained; cancelled or failed jobs keep the previous complete version.

## Summaries

After cleanup, each imported lecture also gets a **Summary**: an overview of what
the lecture covers, then the main ideas under headings. Notes go topic by topic
through the whole lecture; summaries are shorter and cover only the main ideas. A summary reads the
AI version (Raw if cleanup has not run), and leaves both intact. Every cleanup
started from the window writes a summary, whatever the recording type. The style
chosen in **Format** (Mostly prose, Mixed, Mostly structured) applies to both Notes
and Summary; a style not made yet is written when you pick it, notes first.

The **Lecture | Dictation** switch next to **Import audio** decides how recordings
are cleaned. Lecture keeps everything that was said; Dictation keeps only the
correction when the speaker corrects themselves. The choice is remembered for new
imports, and flipping it re-cleans the selected lecture in that mode (instantly
if it was cleaned that way before). The other mode is never prepared in the
background.

Long lectures are summarized in parts of about 5,000 characters, then the part
summaries are merged into one. Numbers in the summary are checked against the
transcript; one that doesn't appear there marks the summary for review. A summary
is written in the model's own words, so check important points against the
transcript or audio.

Export Summary as text, Markdown, or JSON; summaries cannot become SRT subtitles.
Completed versions are retained in `summary-history`; failure or cancellation
keeps the prior version.

## Choose devices and manage resource use

Speech recognition and AI cleanup have separate device settings. Speech starts
on CPU; cleanup starts on AUTO (GPU, then CPU). These policies were chosen from tests on one computer. Resource use varies with
hardware and other applications. Saved choices
are restored, including explicit devices. See the compact
[tests and decisions](TESTED_DECISIONS.md) record for the comparisons.

- **Speech:** Parakeet v2 is the default and was the most accurate and fastest
  speech model in our tests; it always uses the CPU (two threads). Whisper Small
  is the more accurate of the two Whisper alternatives, Whisper Base the smaller
  download; both run on OpenVINO and start on CPU. Parakeet and the Whisper models
  are English-only. Use `models list` to find approved model IDs.
- **Cleanup on GPU:** fastest under moderate graphics activity in our one-host
  test, with less resident RAM than CPU or NPU cleanup.
- **Cleanup on NPU:** consumed the least CPU, but still needed several GiB of
  RAM and slowed a heavier graphics workload. It is not guaranteed to leave
  other applications unaffected.
- **Cleanup on CPU:** works without a compatible accelerator. The 7B cleanup
  model used more CPU and RAM in that comparison.

The [one-host resource comparison](../host-validation/evidence/2026-10-02/cleanup-resource-cost/README.md)
records actual speed, RAM, and graphics interference. Its short requests and
synthetic graphics workloads do not predict every app or device.

Check resource use with your normal applications open. The desktop app loads
the selected speech model, and the cleanup model when cleanup or AI formatting
is on, as soon as it opens, and loads a newly selected model when you change
the choice. Models stay loaded while the window is minimized, and closing the
app unloads all of them. The cleanup model holds several gigabytes of memory
the whole time the app is open; set Cleanup to Off and Formatting to Off or
Mostly prose to release it. A job started while a model is still loading waits
for it.
First use may compile the model before producing text. Stop/Pause can wait for
compilation to finish; otherwise it takes effect within a few seconds. On CPU and GPU,
the cleanup model works on all of a job's blocks at once, so progress moves in steps
as blocks finish. There is no automatic resource cap or idle scheduling;
stop or defer cleanup when you need the headroom. Audio device Auto benchmarking
does not measure cleanup or the effect on your other applications.

## Alternate model folder

Choose **Settings → Model folder** to store model downloads on another drive. Select a parent folder containing model-ID subfolders. If the folder becomes unavailable, the app reports this and uses the default. [CLI examples](CLI_GUIDE.md#alternate-model-folder) cover explicit paths.

## Update the installed app

Open **Help → Check for updates**. The app also checks at startup, at most once
per day; disable **Check for updates automatically** in that window if desired.
There are no automatic downloads or installations.

When a newer release is available, read its notes and choose **Update and reopen**.
The app downloads and checks the installer, waits for processing, dictation,
and model downloads to finish, then closes, installs, and reopens. Models,
recordings, settings, and transcripts stay in place. **Cancel** stops a pending
update and **Later** postpones it. No terminal commands are needed.

If a download fails, retry from the update window. If installation fails, its
dialog offers Retry and gives the installer/log folder for repair. You can also
close the app and run a newer installer manually. Update checks use GitHub and
can fail while offline without affecting transcription. Only stable Windows x64
releases are offered. In a source installation, the update button opens Releases;
use the [source update procedure](#update-an-existing-checkout) to update a checkout.

If you have a tester build from before the GitHub account became `lai-derek`,
close the app and run the current installer manually once instead: older updaters
can reject the new download address. Reinstalling keeps the library, models, and
settings.

## Update an existing checkout

Close the app first, after pausing or finishing any running job. Keep the library
separate from the source folder and back it up before an update.

For a **Git checkout**, run `git status --short` from the repository folder.
Preserve local edits before updating. Check `git remote -v` too. If this clone
still points to the former `derek-l8` account, update its origin:

```powershell
git remote set-url origin https://github.com/lai-derek/audio-transcriber.git
if ($LASTEXITCODE -ne 0) { throw 'Remote update failed; stop here.' }
```

Keep a fork's origin pointing to your fork; update its upstream remote instead.
When the checkout is clean and on your tracking branch, run:

```powershell
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Update failed; resolve the Git error before reinstalling.' }
```

If Git reports detached HEAD or divergent history, stop and resolve that state
before reinstalling; do not discard local work to bypass the error.

For a **ZIP installation**, download and extract the new source ZIP into a
new local folder. Keep the old folder if it contains your own code changes;
do not copy its virtual environment into the new folder.

In either case, install from the updated source folder:

```powershell
py -3.14 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed; stop here.' }
.\.venv\Scripts\python.exe -m pip install --upgrade '.[desktop,inference]'
if ($LASTEXITCODE -ne 0) { throw 'Reinstallation failed; stop here.' }
```

Reinstallation is required to refresh the installed program and dependencies.
Your existing models and sessions remain in the separate library; reopen with
the same library and model folder. [CLI options](CLI_GUIDE.md#alternate-model-folder) cover custom paths.

## CLI transcription and export

See the [command-line guide](CLI_GUIDE.md) for transcription, exports, FFmpeg setup, pause/resume, cleanup, and alternate model folders.

## Common problems

| Message or symptom | Check |
|---|---|
| `py -3.14` unavailable | Install 64-bit Python 3.14 or make that version available to the Python launcher. |
| Python module missing | Run from the source folder with its `.venv\Scripts\python.exe`, or use the full executable path. Install the extras shown in the README. |
| Model not installed | Open **Manage models** and download the speech or cleanup model into the selected model folder. |
| Resume model mismatch | Supply the original `--model`; the CLI default is `parakeet-tdt-0.6b-v2-int8`. |
| "requires FFmpeg" for an MP3, M4A, MP4, or WAV | Installer builds include it. For source installs, follow [setup step 3](#install-on-windows) or choose its executable under **Settings → FFmpeg program**. |
| GPU/NPU fails | Try CPU. An explicit device request will not silently switch. |
| Library already open | Close the other desktop window using that library. |
| Transcript misses words | Review the audio; model errors and chunk-boundary losses remain possible. |
| Cleanup is slow or uses too much RAM | Check [device/resource guidance](#choose-devices-and-manage-resource-use); compilation and a resident 7B model can be costly. |
| AI text contains unchanged or incorrect passages | Compare with Raw/audio. A rejected block retains its source; passing the checks does not guarantee correct edits. |

For installed-app updates, use [Check for updates](#update-the-installed-app).
For data handling and report sharing, see [Privacy](../PRIVACY.md). Advanced
evaluation and hardware checks are described in [Development](../DEVELOPMENT.md#checks)
and [Windows validation](../host-validation/README.md).
