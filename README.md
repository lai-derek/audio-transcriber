# Audio Transcriber

Transcribe English audio and video files locally, clean up the text, and export
transcripts or study notes. The Windows app includes playback, corrections,
search, and dictation with a configurable hotkey.

Processing runs offline after model download, with no subscription or API key.
This is an early Windows x64 version; AI cleanup and dictation are experimental.

## Setup

Windows x64 is the supported desktop target. No public installer is published
yet; use the [source setup](docs/USER_GUIDE.md#install-on-windows) with 64-bit
Python 3.14, or build the [Windows installer](packaging/README.md).

Installer builds include Python and FFmpeg. Open the installed app, choose
**Manage models**, and download an English speech model. AI cleanup is optional
and uses a separate download of about 4.5 GB. The setup window shows download
sizes and progress; transcription works offline once the speech model is ready.

Published installers will appear on [GitHub Releases](https://github.com/lai-derek/audio-transcriber/releases).

## Transcribe a file

1. Select your installed speech model. Installer builds include compressed-file support.
2. Choose cleanup and formatting settings, then press **Import lecture**.
3. Review the text. **Raw** keeps the original transcript; cleanup and formatting
   are saved separately. **Summarize selected** creates shorter study notes.
4. Export text, Markdown, JSON, or SRT subtitles.

**Clean selected** and **Format selected** rerun those stages. See the
[desktop guide](docs/USER_GUIDE.md#desktop-workflow) for editing and playback,
or the [CLI guide](docs/USER_GUIDE.md#cli-transcription-and-export) for commands.

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
| Imported files, cleanup model installed | Light | Mostly structured |
| Dictation, cleanup model installed | Medium | Mostly prose |

Without the cleanup model, imported files start with cleanup and formatting
**Off**. Dictation starts with cleanup **Off** and Mostly prose formatting.
Download the model in **Manage models**, then select the settings you want.

Choose **Off**, **Light**, or **Medium** cleanup. Formatting offers **Mostly
prose**, **Mixed**, and **Mostly structured** layouts. Summaries are a separate
option. Review important text: recognition and cleanup can make mistakes, and
summaries can omit detail.

## Devices and measured decisions

Transcription starts on **CPU**. Cleanup uses **AUTO**, which tries an available
OpenVINO GPU, then CPU. Both stages also offer explicit CPU, GPU, and NPU choices.

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

[Development](DEVELOPMENT.md) covers contributor setup;
[testing](TESTING.md) records the checks;
[packaging](packaging/README.md) covers building the Windows installer and
preparing a release.

Code is [MIT licensed](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md)
for dependencies and models.
