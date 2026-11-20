# Audio Transcriber

Transcribe English audio and video files locally, clean up the text, and export
transcripts, study notes, or summaries. The Windows app includes playback,
corrections, search, and dictation with a configurable hotkey.

Processing runs offline after model download, with no subscription or API key.
This is an early Windows x64 version; AI cleanup and dictation are experimental.

## Setup

Windows x64 is the supported desktop target. No public installer is published
yet. To run the public source, follow the [Windows setup](docs/USER_GUIDE.md#install-on-windows)
with 64-bit Python 3.14. Building an [installer](packaging/README.md) is optional.

If you have a tester installer, run it and open the desktop shortcut. It includes
Python and FFmpeg; no developer tools or separate Python installation are needed.
On first launch the app downloads the English speech models (Parakeet v2, the
default, then Whisper Small and Whisper Base; about 890 MB together), then the AI
cleanup model (about 4.5 GB), and turns on cleanup when it arrives.
Transcription can start once a speech model is ready,
and everything works offline after the downloads.

Published installers will appear on [GitHub Releases](https://github.com/lai-derek/audio-transcriber/releases).

## Transcribe a file

1. Download the speech model (the app does this on first launch). Installers
   include FFmpeg for MP3, M4A, MP4, and stereo or high-rate WAV files; source
   installs add it in [setup](docs/USER_GUIDE.md#install-on-windows).
2. Press **Import audio**. Cleanup and formatting for new imports are under
   **Settings**.
3. Review the text. Each lecture opens on its most finished version; tabs switch
   between **Original** (speech recognition output), **Transcript** (AI cleanup), **Notes**,
   and **Summary**. Every version is saved separately.
4. Export text, Markdown, JSON, or SRT subtitles.

Choose **Lecture** or **Dictation** to control how self-corrections are cleaned.
**Clean up** changes the cleanup level; **Format** changes the layout of both
Notes and Summary. Saved versions switch immediately; missing versions are
prepared by the app. Type directly in a tab to correct it; edits save automatically.

Right-click a lecture to rename or delete it. Deleted lectures can be restored
from **File → Trash** for 30 days. See the
[desktop guide](docs/USER_GUIDE.md#desktop-workflow) for playback, corrections,
pause/resume, and background preparation, or the [CLI guide](docs/CLI_GUIDE.md)
for commands.

## Dictation

Open **Live dictation**, select a microphone and speech model, and enable the
shortcut. The default is **Ctrl+Alt+Space**; choose Toggle or Hold to talk.
Wait for **Recording** before speaking.

Focus a text field in another app. When recording stops, the app transcribes,
cleans up the text, saves a copy, and inserts it into that field. If insertion
fails, recover the text from the dictation window.
[Dictation setup and options](docs/USER_GUIDE.md#live-dictation).

## Cleanup and formatting

| Workflow | Default cleanup | Default formatting |
| --- | --- | --- |
| Imported files, cleanup model installed | Medium | Mixed |
| Dictation, cleanup model installed | Medium | Mostly prose |

Without the cleanup model, imported files start with cleanup and formatting
**Off**. Dictation starts with cleanup **Off** and Mostly prose formatting.
Download the model in **Manage models**, then select the settings you want.

Choose **Off**, **Light**, **Medium**, or **Heavy** cleanup. Light removes filler
and fixes grammar, Medium also removes false starts and smooths broken sentences,
and Heavy rewrites for easy reading while keeping every point. **Mixed** and
**Mostly structured** formatting condense the text into study notes under topic
headings, as short paragraphs or bullets; **Mostly prose** keeps plain paragraphs.
After cleanup, each lecture also gets a written **Summary** of its main ideas. Review important text: recognition,
cleanup, notes, and summaries can all make mistakes.

## Devices and performance

Speech recognition uses NVIDIA's Parakeet TDT 0.6B v2 through sherpa-onnx, which
always runs on the **CPU**. The Whisper Small and Base alternatives run on OpenVINO
and start on CPU, with GPU and NPU choices. Cleanup uses **AUTO**, which tries an
available OpenVINO GPU, then CPU, and also offers explicit CPU, GPU, and NPU choices.

The app loads selected models when it opens and keeps them in memory while
minimized. Closing the app releases them. AI cleanup needs several GiB of RAM;
**Settings → Prepare in the background** chooses what is made ahead of time.

The [tested decisions](docs/TESTED_DECISIONS.md) document compares speed, accuracy,
memory use, and graphics interference on the test computer, and explains the
chosen defaults. See [device settings](docs/USER_GUIDE.md#choose-devices-and-manage-resource-use)
for configuration and resource limits.

## Files and updates

Recordings, models, transcripts, and exports stay in your local library.
[Privacy and storage](PRIVACY.md) explains the locations.

Installed apps offer **Help → Check for updates**. Choose **Update and reopen**;
the app waits for current work to finish, installs, and reopens. The
[update guide](docs/USER_GUIDE.md#update-the-installed-app) covers this flow and
[source updates](docs/USER_GUIDE.md#update-an-existing-checkout). Updates keep the library and models. See
[common problems](docs/USER_GUIDE.md#common-problems) for model, FFmpeg, device,
and startup errors.

## Development

[Development](DEVELOPMENT.md) covers contributor setup and checks;
[Architecture](ARCHITECTURE.md) explains how the app is built;
[packaging](packaging/README.md) covers building the Windows installer and
preparing a release.

Code is [MIT licensed](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md)
for dependencies and models.
