# Audio Transcriber 0.2.0 — Windows x64

This release adds graphical model setup, in-app updates, saved cleanup versions,
editable notes and summaries, and a recoverable Trash. Python and FFmpeg are
bundled, so installation and ordinary use require no terminal commands.

Download `audio-transcriber-0.2.0-windows-x64.exe`, run it, and open the app.
First launch downloads the English speech models, then the AI cleanup model
(about 4.5 GB). **Manage models** shows progress. Transcription can start once a
speech model is ready; Medium cleanup and Mixed formatting turn on after the
cleanup model arrives. Interrupted downloads resume on the next launch. You
can stop setup and later select individual models in **Manage models**.

Original recognition text stays available beside Transcript, Notes, and Summary.
Type to correct a version; exports use the selected version and its matching
edits. The Lecture / Dictation switch controls cleanup of self-corrections.
Models stay loaded while the app is open and unload when it exits. Choose which
versions are prepared in the background under **Settings → Prepare in the background**.

For updates, use **Help → Check for updates → Update and reopen**. Installation
waits for current work to finish and preserves the library. Automatic update
checks can be disabled; downloading and installation require your click.

Existing testers using a build from before the move to `lai-derek` should close
the app and run this installer manually once. Older updaters can reject the
renamed repository's asset URLs. Reinstallation keeps models and the library.

The installer is unsigned and Windows may show a publisher warning. This is an
early Windows x64 release for testers. Review important transcripts and AI text.

`SHA256SUMS.txt` contains attachment checksums. Third-party sources and rebuild
instructions accompany the installer in `audio-transcriber-0.2.0-library-sources.zip`.
