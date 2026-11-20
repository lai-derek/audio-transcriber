"""Exercise the Qt workflow with real child processes; Qt remains optional."""

from __future__ import annotations

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from audio_transcriber.desktop_editing import RevisionHistoryDialog, SegmentEditDialog  # noqa: E402
from audio_transcriber.desktop_player import LecturePlayer, clock  # noqa: E402
from audio_transcriber.desktop_setup import choice, select  # noqa: E402
from audio_transcriber.desktop_ui import LectureWindow  # noqa: E402
from audio_transcriber.editing import revision_history  # noqa: E402
from audio_transcriber.models import Segment  # noqa: E402
from audio_transcriber.storage import sha256_file  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    (tmp_path / "models").mkdir()
    result = LectureWindow(tmp_path / "library", tmp_path / "models")
    # The mock is a test fixture, never offered in the production model selector.
    result.model.addItem("Test fixture", "mock")
    result.model.setCurrentIndex(result.model.findData("mock"))
    select(result.cleanup_style, "off")
    result.format_style.setCurrentIndex(result.format_style.findData("off"))
    result.background_enabled = False  # Tests that need background versions turn it on.
    yield result
    if result.busy:
        result.pause()
        wait_finished(app, result)
    result.close()


def install_cleanup_placeholder(window):
    """Present the cleanup model's file names; contents are not valid model weights."""
    from audio_transcriber.acquisition import MANIFEST
    from audio_transcriber.ai_cleanup import DEFAULT_MODEL

    folder = window.model_root / DEFAULT_MODEL
    folder.mkdir(parents=True, exist_ok=True)
    for entry in MANIFEST[DEFAULT_MODEL].files:
        (folder / entry.name).touch()
    window._buttons()
    return folder


def use_fixture_model(monkeypatch, window, tmp_path, fixture):
    """Run the app's model process with `fixture` (a class named Fixture) as the text model."""
    import sys

    worker = tmp_path / "fixture-worker.py"
    worker.write_text(
        "import sys\nimport audio_transcriber.cli as cli\n"
        + fixture
        + "\ncli.make_model = lambda *args: Fixture()\n"
        + "raise SystemExit(cli.main(sys.argv[1:]))\n"
    )
    monkeypatch.setattr(
        "audio_transcriber.desktop_worker.worker_command",
        lambda args: (sys.executable, [str(worker), *args]),
    )
    window.worker.stop()  # The next request starts the process with the fixture.


def wait_finished(app, window):
    deadline = time.monotonic() + 30
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not window.busy, window.output


def wait_seekable(app, player):
    deadline = time.monotonic() + 10
    while not player.player.isSeekable() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert player.player.isSeekable(), player.message.text()


def test_player_local_media_seek_clamp_and_no_reload(app, three_second_wav):
    player = LecturePlayer()
    player.audio.setMuted(True)
    player.load(three_second_wav)
    wait_seekable(app, player)
    assert 2900 <= player.player.duration() <= 3100
    player.seek(1.2)
    assert player.player.position() == 1200
    player.load(three_second_wav)
    assert player.player.position() == 1200
    player.seek(-1)
    assert player.player.position() == 0
    player.seek(100)
    assert player.player.position() == player.player.duration()
    player.load(None)
    assert not player.play_button.isEnabled()
    assert clock(3_661_999) == "01:01:01"
    player.close()


def test_slider_keyboard_and_accessibility_value_seek(app, three_second_wav):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    player = LecturePlayer()
    player.audio.setMuted(True)
    player.load(three_second_wav)
    wait_seekable(app, player)
    # Accessibility providers change the slider value without dragging it.
    player.slider.setValue(1200)
    assert player.player.position() == 1200
    QTest.keyClick(player.slider, Qt.Key.Key_Right)
    assert player.player.position() == 1200 + player.slider.singleStep()
    player.close()


def test_transcript_seek_uses_preserved_source(app, window, three_second_wav, monkeypatch):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    wait_seekable(app, window.player)
    session = next(iter(window.sessions.values()))
    assert window.player.source == window.store.source_copy(session.id).resolve()
    recorded = []
    monkeypatch.setattr(window.player, "seek", recorded.append)
    window.seek_selected_segment()
    assert recorded == [window.segment_starts[0]]
    source = window.player.source
    window.layer.setCurrentText("Balanced")
    assert window.player.source == source


def test_import_search_layers_export_and_reopen(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    window.start_import(three_second_wav)
    assert window.busy
    assert not window.import_button.isEnabled()
    wait_finished(app, window)
    assert len(window.sessions) == 1
    session = next(iter(window.sessions.values()))
    assert session.status == "ready", window.status.text()
    assert window.transcript.toPlainText()
    window.layer.setCurrentText("Balanced")
    text = window.transcript.toPlainText()
    window.search.setText(text.split()[-1])
    window.find_text()
    assert window.transcript.textCursor().hasSelection()
    destination = tmp_path / "lecture.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    select(window.export_format, "json")
    window.choose_export()
    assert '"layer": "balanced"' in destination.read_text()
    reopened = LectureWindow(window.store.data_dir)
    assert len(reopened.sessions) == 1
    assert reopened.model_root == window.model_root
    reopened.library.setCurrentRow(0)
    assert reopened.transcript.toPlainText()
    reopened.close()


def test_pause_and_resume_in_child_process(app, window, three_second_wav):
    window.start_import(three_second_wav)
    window.pause()
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    assert session.status == "interrupted", window.status.text()
    assert window.store.checkpoint_path(session.id).exists()
    assert window.resume_button.isEnabled()
    assert not window.resume_button.isHidden()
    assert window.pause_button.isHidden()  # Nothing is running.
    assert window.status_label(session) == "Paused"
    assert window.progress.maximum() > 0
    window.resume_selected()
    wait_finished(app, window)
    session = window.store.load_session(session.id)
    assert session.status == "ready", window.status.text()
    assert not window.store.checkpoint_path(session.id).exists()
    assert window.transcript.toPlainText()


def test_close_waits_for_safe_pause(app, window, three_second_wav):
    window.show()
    window.start_import(three_second_wav)
    window.close()
    assert window.closing
    wait_finished(app, window)
    assert not window.isVisible()
    assert next(iter(window.sessions.values())).status == "interrupted"


def test_invalid_model_does_not_start(app, window, three_second_wav):
    window.model.setCurrentIndex(0)
    window.start_import(three_second_wav)
    assert not window.busy
    assert "No speech model is ready" in window.status.text()
    window.model.setCurrentIndex(1)
    window.start_import(three_second_wav)
    wait_finished(app, window)
    assert "failed" in window.status.text()
    assert not window.sessions
    assert window.import_button.isEnabled()


def test_bad_metadata_and_settings_are_visible(app, tmp_path):
    root = tmp_path / "library"
    folder = root / "lectures" / "broken"
    folder.mkdir(parents=True)
    (folder / "session.json").write_text("[]")
    (root / "desktop-settings.json").write_text("[]")
    result = LectureWindow(root)
    assert "Skipped 1" in result.status.text()
    assert not result.sessions
    assert (folder / "session.json").read_text() == "[]"
    result.close()


def test_missing_saved_locations_fall_back_and_defaults_are_not_saved(app, tmp_path):
    import json

    root = tmp_path / "library"
    root.mkdir()
    settings = root / "desktop-settings.json"
    settings.write_text(
        json.dumps(
            {
                "model_root": str(tmp_path / "other-pc" / "models"),
                "ffmpeg": str(tmp_path / "other-pc" / "ffmpeg.exe"),
                "device": "GPU",
            }
        )
    )
    result = LectureWindow(root)
    assert result.model_root == (root / "models").resolve()
    assert "saved model folder was not found" in result.status.text()
    assert "other-pc" not in result.ffmpeg
    assert choice(result.device) == "GPU"
    result._save_settings()
    saved = json.loads(settings.read_text())
    assert "model_root" not in saved and "ffmpeg" not in saved
    assert saved["device"] == "GPU"
    result.close()


def test_chosen_locations_are_saved(app, tmp_path, monkeypatch):
    import json

    decoder = tmp_path / "tools" / "ffmpeg.exe"
    decoder.parent.mkdir()
    decoder.touch()
    models = tmp_path / "shared-models"
    models.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(models))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(decoder), ""))
    result = LectureWindow(tmp_path / "library")
    result.choose_models()
    result.choose_ffmpeg()
    result._save_settings()
    result.close()
    saved = json.loads((tmp_path / "library" / "desktop-settings.json").read_text())
    assert saved["model_root"] == str(models.resolve())
    assert saved["ffmpeg"] == str(decoder)
    reopened = LectureWindow(tmp_path / "library")
    assert reopened.model_root == models.resolve()
    assert reopened.ffmpeg == str(decoder.resolve())
    reopened.close()


def test_gui_edit_save_reopen_history_and_export(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session_id = window.selected_id()
    raw = window.store.session_dir(session_id) / "raw-transcript.json"
    raw_hash = sha256_file(raw)

    def edit_dialog(dialog):
        dialog.text.setPlainText("Manual correction for the GUI test.")
        dialog.uncertain.setChecked(True)
        dialog.save()
        return dialog.result()

    monkeypatch.setattr(SegmentEditDialog, "exec", edit_dialog)
    window.edit_selected_segment()
    assert window.layer.currentText() == "Edited"
    assert "Edited by user" in window.details.text()
    assert "Manual correction" in window.transcript.toPlainText()
    assert "[unclear]" in window.transcript.toPlainText()
    assert sha256_file(raw) == raw_hash
    window.layer.setCurrentText("Raw")
    assert not window.edit_button.isEnabled()
    window.layer.setCurrentText("Edited")
    assert window.edit_button.isEnabled()
    history = RevisionHistoryDialog(revision_history(window.store, session_id))
    assert history.preview.isReadOnly()
    assert "Manual correction" in history.preview.toPlainText()
    history.close()
    reopened = LectureWindow(window.store.data_dir)
    reopened.library.setCurrentRow(0)
    reopened.layer.setCurrentText("Edited")
    assert "Manual correction" in reopened.transcript.toPlainText()
    destination = tmp_path / "edited.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    select(reopened.export_format, "json")
    reopened.choose_export()
    assert '"edit_provenance"' in destination.read_text()
    reopened.close()


def test_edit_dialog_keeps_draft_when_save_fails(app):
    def failed_commit(text, uncertain):
        raise PermissionError("synthetic write denial")

    dialog = SegmentEditDialog(Segment(0, 1, "original"), failed_commit)
    dialog.show()
    dialog.text.setPlainText("Keep this unsaved draft")
    dialog.save()
    assert dialog.isVisible()
    assert dialog.saved is None
    assert dialog.text.toPlainText() == "Keep this unsaved draft"
    assert "not saved" in dialog.error.text()
    dialog.reject()


def test_desktop_ai_cleanup_worker_review_export_and_reopen(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    from audio_transcriber.ai_cleanup import DEFAULT_MODEL

    assert window.model.findData(DEFAULT_MODEL) == -1
    assert window.cleanup_model.findData(DEFAULT_MODEL) >= 0
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    before = sha256_file(window.store.session_dir(session.id) / "raw-transcript.json")
    use_fixture_model(
        monkeypatch,
        window,
        tmp_path,
        """
class Fixture:
    def generate(self, text, mode, style):
        assert (mode, style) == ('dictation', 'medium')
        return text.capitalize()+'.'
""",
    )
    placeholder = install_cleanup_placeholder(window)
    window.cleanup_mode.setCurrentIndex(window.cleanup_mode.findData("dictation"))
    select(window.cleanup_style, "medium")
    select(window.cleanup_device, "NPU")
    window.start_cleanup()
    assert window.busy
    wait_finished(app, window)
    assert window.layer.currentText() == "AI cleanup"
    assert window.transcript.toPlainText() == "Synthetic transcript."
    assert not window.edit_button.isEnabled()
    assert not window.player.seek_button.isEnabled()
    assert window.export_button.isEnabled()
    assert before == sha256_file(window.store.session_dir(session.id) / "raw-transcript.json")
    destination = tmp_path / "ai-export.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    select(window.export_format, "json")
    window.choose_export()
    assert '"transformation"' in destination.read_text()
    import shutil

    shutil.rmtree(placeholder)
    reopened = LectureWindow(window.store.data_dir)
    assert reopened.cleanup_mode.currentData() == "dictation"
    # Missing cleanup weights disable AI on reopen without losing the saved choice.
    assert choice(reopened.cleanup_style) == "off"
    assert choice(reopened.cleanup_device) == "NPU"
    reopened.library.setCurrentRow(0)
    reopened.layer.setCurrentText("AI cleanup")
    assert reopened.transcript.toPlainText() == "Synthetic transcript."
    reopened.close()


def test_ai_controls_require_the_cleanup_model(app, window, three_second_wav):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    mixed = window.format_style.model().item(window.format_style.findData("mixed"))
    select(window.cleanup_style, "light")
    assert choice(window.cleanup_style) == "off"
    assert not window.cleanup_style.isEnabled()
    assert not window.summary_action.isEnabled()
    assert not window.cleanup_action.isEnabled()
    assert not mixed.isEnabled()
    assert window.cleanup_hint.isVisibleTo(window)
    install_cleanup_placeholder(window)
    assert window.cleanup_style.isEnabled()
    assert window.summary_action.isEnabled()
    assert mixed.isEnabled()
    assert not window.cleanup_hint.isVisibleTo(window)


def test_choosing_a_model_folder_updates_ai_controls(app, window, tmp_path, monkeypatch):
    other = LectureWindow(tmp_path / "other-library", tmp_path / "other-models")
    install_cleanup_placeholder(other)
    other.close()
    assert window.cleanup_hint.isVisibleTo(window)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path / "other-models")
    )
    window.choose_models()
    assert not window.cleanup_hint.isVisibleTo(window)
    assert window.cleanup_style.isEnabled()


def test_automatic_cleanup_failure_preserves_raw(app, window, three_second_wav):
    install_cleanup_placeholder(window)  # Present but unusable: the worker rejects it.
    select(window.cleanup_style, "light")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    assert session.status == "ready"
    assert window.status.text().startswith("Transcript remains ready. Cleanup failed (exit 1).")
    assert "damaged or incomplete" in window.status.text()
    assert "Manage models" in window.status.text()
    assert "\n" not in window.status.text()
    assert window.layer.currentText() == "Raw"
    assert window.export_button.isEnabled()


def test_layer_menu_offers_only_existing_versions(app, window, three_second_wav):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    items = window.layer.model()
    enabled = {
        window.layer.itemText(row): items.item(row).isEnabled()
        for row in range(window.layer.count())
    }
    assert enabled == {
        "Raw": True,
        "Balanced": True,
        "Edited": False,
        "AI cleanup": False,
        "Formatted": False,
        "Summary": False,
    }
    window.layer.setCurrentText("Edited")
    assert window.layer.currentText() == "Raw"
    assert "Could not read" not in window.transcript.toPlainText()
    assert "synthetic transcript" in window.transcript.toPlainText()


def test_cleanup_defaults_and_off_persist(app, tmp_path):
    window = LectureWindow(tmp_path / "library")
    assert choice(window.cleanup_style) == "off"
    assert choice(window.cleanup_device) == "AUTO"
    assert choice(window.device) == "CPU"
    assert window.format_style.currentData() == "off"
    window.open_dictation()
    assert choice(window.dictation_window.cleanup) == "off"
    window.dictation_window.close()
    select(window.cleanup_style, "off")
    window.format_style.setCurrentIndex(window.format_style.findData("off"))
    select(window.device, "GPU")
    window._save_settings()
    window.close()
    reopened = LectureWindow(tmp_path / "library")
    assert choice(reopened.cleanup_style) == "off"
    assert choice(reopened.device) == "GPU"
    assert reopened.format_style.currentData() == "off"
    reopened.close()


def test_desktop_summary_keeps_ai_and_disallows_subtitles(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    use_fixture_model(
        monkeypatch,
        window,
        tmp_path,
        """
class Fixture:
    def generate(self, text, mode, style):
        return '## Synthetic transcript\\n\\n- point'
""",
    )
    install_cleanup_placeholder(window)
    window.start_summary()
    wait_finished(app, window)
    assert window.layer.currentText() == "Summary"
    # Summary is shown as rendered Markdown, so heading marks are not visible.
    shown = window.transcript.toPlainText()
    assert "Synthetic transcript" in shown
    assert "point" in shown
    assert "##" not in shown
    assert window.summary_action.isEnabled()
    assert not window.edit_button.isEnabled()
    assert not window.player.seek_button.isEnabled()
    # Notes and summaries have no timing, so the Export menu offers no subtitles.
    assert not window.export_actions["srt"].isEnabled()
    assert window.export_actions["markdown"].isEnabled()
    assert window.export_button.isEnabled()
    assert window.store.load_transcript(session.id, "raw").text == "synthetic transcript"


def test_automatic_cleanup_success_after_import(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    use_fixture_model(
        monkeypatch,
        window,
        tmp_path,
        """
class Fixture:
    device='CPU'
    def generate(self, text, mode, style):
        return text.capitalize()+'.'
""",
    )
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "medium")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    assert window.layer.currentText() == "AI cleanup"
    assert window.transcript.toPlainText() == "Synthetic transcript."
    assert window.store.load_transcript(session.id, "ai").transformation["style"] == "medium"
    assert window.store.load_transcript(session.id, "raw").text == "synthetic transcript"
    # Turning cleanup off for new imports does not stop rerunning it on this lecture.
    select(window.cleanup_style, "off")
    assert window.cleanup_action.isEnabled()
    assert window.summary_action.isEnabled()


def test_cleanup_progress_names_each_stage_and_keeps_cleanup_when_summary_fails(
    app, window, three_second_wav
):
    from PySide6.QtCore import QProcess

    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    window.job_kind, window.active_id, window.stop_file = "cleanup", session.id, None
    window.cleanup_written, window.ai_stage = False, ""
    stages = []
    for line in (
        "cleanup_blocks: 1/4",
        "AI cleanup ready: x\nformat_blocks: 2/5",
        "cleanup_blocks: 1/3",
    ):
        window.output += line + "\n"
        window._read_output()
        stages.append(window.progress.format())
    assert stages == ["Cleaning up: 1 / 4", "Formatting: 2 / 5", "Writing summary: 1 / 3"]
    assert window.status.text().startswith("Writing summary…")
    window.output += "AI cleanup is ready. The summary failed: model stopped\n"
    window.format_style.setCurrentIndex(window.format_style.findData("off"))
    window._finished(1, QProcess.ExitStatus.NormalExit)
    assert window.status.text().startswith("AI cleanup is ready. The summary failed")
    assert "Write a summary" in window.status.text()


def test_job_stays_busy_until_completion_is_handled(app, window, three_second_wav):
    # Hold completion delivery to exercise a finished job before the UI updates.
    held = []
    window._job_done = lambda code, crashed: held.append((code, crashed))
    try:
        window.start_import(three_second_wav)
        deadline = time.monotonic() + 30
        while not held and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert held == [(0, False)], window.output
        assert window.busy
        assert not window.import_button.isEnabled()
    finally:
        del window._job_done
        window._job_done(*(held[0] if held else (1, True)))
    assert not window.busy
    assert window.import_button.isEnabled()
    assert window.layer.currentText() == "Raw"
    assert "synthetic transcript" in window.transcript.toPlainText()


def test_pause_at_speech_completion_does_not_start_cleanup(app, window, three_second_wav):
    from PySide6.QtCore import QProcess

    window.start_import(three_second_wav)
    wait_finished(app, window)
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "light")
    assert choice(window.cleanup_style) == "light"
    window.job_kind = "transcription"
    window.stop_file.touch()
    window._finished(0, QProcess.ExitStatus.NormalExit)
    assert not window.busy
    session = next(iter(window.sessions.values()))
    assert not (window.store.session_dir(session.id) / "ai-transcript.json").exists()


def test_automatic_prose_formatting_with_cleanup_off(app, window, three_second_wav):
    window.format_style.setCurrentIndex(window.format_style.findData("prose"))
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    assert window.layer.currentText() == "Formatted"
    assert window.transcript.toPlainText() == "synthetic transcript"
    assert (
        window.store.load_transcript(session.id, "formatted").transformation["source_layer"]
        == "raw"
    )
    assert not window.edit_button.isEnabled()
    assert not window.player.seek_button.isEnabled()
    # Notes and summaries have no timing, so the Export menu offers no subtitles.
    assert not window.export_actions["srt"].isEnabled()
    assert window.export_actions["markdown"].isEnabled()
    assert window.export_button.isEnabled()


def test_desktop_cleanup_then_format_and_rerun_keep_sources(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    use_fixture_model(
        monkeypatch,
        window,
        tmp_path,
        """
class Fixture:
    device='CPU'
    def generate(self, text, mode, style):
        if mode == 'format':
            return '- ' + text
        return text.capitalize()+'.'
""",
    )
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "light")
    window.format_style.setCurrentIndex(window.format_style.findData("mixed"))
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = next(iter(window.sessions.values()))
    output = window.store.load_transcript(session.id, "formatted")
    assert output.text == "- Synthetic transcript."
    assert output.transformation["source_layer"] == "ai"
    assert window.layer.currentText() == "Formatted"
    assert "Synthetic transcript." in window.transcript.toPlainText()
    directory = window.store.session_dir(session.id)
    hashes = {layer: sha256_file(directory / f"{layer}-transcript.json") for layer in ("raw", "ai")}
    window.layer.setCurrentText("AI cleanup")
    window.format_style.setCurrentIndex(window.format_style.findData("prose"))
    window.start_formatting()
    wait_finished(app, window)
    assert len(list((directory / "formatting-history").glob("*.json"))) == 2
    assert all(
        sha256_file(directory / f"{layer}-transcript.json") == digest
        for layer, digest in hashes.items()
    )


@pytest.fixture
def dictation(window):
    window.open_dictation()
    result = window.dictation_window
    result.model.setCurrentIndex(result.model.findData("mock"))
    select(result.cleanup, "off")
    yield result
    if result.busy:
        result.cancel()
        wait_finished(QApplication.instance(), result)
    result.close()


def prepare_dictation(dictation, wav):
    import uuid
    from datetime import UTC, datetime

    dictation.ident = uuid.uuid4().hex
    dictation.job = dictation.root / "jobs" / dictation.ident
    dictation.job.mkdir(parents=True)
    dictation.created_at = datetime.now(UTC).isoformat()
    dictation.cancelled = False
    dictation._start_worker(wav)


def test_dictation_pipeline_saves_history_before_insertion(
    app, window, dictation, three_second_wav
):
    from audio_transcriber.insertion import FocusTarget, InsertResult

    class Native:
        def insert(self, text, target, allowed):
            assert allowed()
            assert dictation.history.records()[0]["text"] == text
            return InsertResult(True, text)

    dictation.native = Native()
    dictation.intended = FocusTarget(1, 2, "Test", True)
    prepare_dictation(dictation, three_second_wav)
    assert window.busy
    assert not window.import_button.isEnabled()
    wait_finished(app, dictation)
    assert "input sent" in dictation.status.text()
    assert dictation.text.toPlainText()
    record = dictation.history.records()[0]
    assert record["raw"] == dictation.text.toPlainText()
    assert dictation.store.load_transcript(record["session_id"], "formatted").text == record["text"]
    assert not window.sessions  # Dictation has its own store, outside the file library.
    assert window.import_button.isEnabled()


def test_dictation_focus_change_recovers_without_insertion(app, dictation, three_second_wav):
    from audio_transcriber.insertion import FocusTarget, InsertResult

    class Native:
        def insert(self, text, target, allowed):
            return InsertResult(False, text, "Focus changed; text was saved instead.")

    dictation.native = Native()
    dictation.intended = FocusTarget(1, 2, "Test", True)
    prepare_dictation(dictation, three_second_wav)
    wait_finished(app, dictation)
    assert "Focus changed" in dictation.status.text()
    assert dictation.text.toPlainText() == dictation.history.records()[0]["text"]


def test_dictation_failed_cleanup_keeps_raw_and_skips_insertion(app, dictation, three_second_wav):
    class Native:
        def insert(self, *args):
            raise AssertionError("A failed job must not insert text")

    dictation.native = Native()
    install_cleanup_placeholder(dictation.owner)  # Selectable, but the worker rejects it.
    select(dictation.cleanup, "light")
    assert choice(dictation.cleanup) == "light"
    prepare_dictation(dictation, three_second_wav)
    wait_finished(app, dictation)
    assert "Processing failed" in dictation.status.text()
    record = dictation.history.records()[0]
    assert record["status"] == "failed"
    assert record["raw"] and record["text"] == record["raw"]


def test_dictation_history_write_failure_skips_insertion(
    app, dictation, three_second_wav, monkeypatch
):
    class Native:
        def insert(self, *args):
            raise AssertionError("Unsaved text must not be inserted")

    def fail(record):
        raise OSError("disk unavailable")

    monkeypatch.setattr(dictation.history, "save", fail)
    dictation.native = Native()
    prepare_dictation(dictation, three_second_wav)
    wait_finished(app, dictation)
    assert "insertion skipped" in dictation.status.text()
    assert dictation.text.toPlainText()
    assert dictation.copy.isEnabled()


def test_dictation_toggle_hold_and_repeat_behavior(dictation, monkeypatch):
    calls = []
    shortcut = type("Shortcut", (), {"enabled": True, "released": lambda self: True})()
    dictation.shortcut = shortcut
    monkeypatch.setattr(
        dictation, "start_recording", lambda insert: calls.append(("start", insert))
    )
    monkeypatch.setattr(dictation, "stop_recording", lambda: calls.append(("stop",)))
    dictation.activate_shortcut()
    dictation.state = "preparing"
    dictation.activate_shortcut()
    assert calls == [("start", True)]
    dictation.state = "recording"
    dictation.activate_shortcut()
    assert calls[-1] == ("stop",)
    calls.clear()
    dictation.mode.setCurrentText("Hold to talk")
    dictation.state = "idle"
    dictation.activate_shortcut()
    assert dictation.holding
    dictation.state = "recording"
    dictation.started = time.monotonic()
    dictation.activate_shortcut()  # Repeated key-down is ignored in hold mode.
    dictation._tick()  # Release ends it.
    assert calls == [("start", True), ("stop",)]
    dictation.state = "idle"
    dictation.shortcut = None


def test_dictation_recording_limit_and_cancel(dictation, monkeypatch):
    stopped = []
    monkeypatch.setattr(dictation.mic, "stop", lambda: stopped.append(True))
    dictation._state("recording")
    dictation.cancel()
    assert stopped and not dictation.busy
    assert "cancelled" in dictation.status.text()
    calls = []
    monkeypatch.setattr(dictation, "stop_recording", lambda: calls.append(True))
    dictation.state = "recording"
    dictation.started = time.monotonic() - 121
    dictation._tick()
    assert calls == [True]
    dictation.state = "idle"


def test_dictation_normalization_then_worker(app, dictation, three_second_wav):
    import uuid
    import wave
    from datetime import UTC, datetime

    dictation.ident = uuid.uuid4().hex
    dictation.created_at = datetime.now(UTC).isoformat()
    dictation.job = dictation.root / "jobs" / dictation.ident
    dictation.job.mkdir(parents=True)
    with wave.open(str(three_second_wav), "rb") as wav:
        (dictation.job / "capture.pcm").write_bytes(b"\x00\x10" * wav.getnframes())
    dictation.mic.stop = lambda: (16000, 1, "int16")
    dictation._state("recording")
    dictation.stop_recording()
    wait_finished(app, dictation)
    assert dictation.history.records()[0]["status"] == "ready"
    assert not (dictation.job / "capture.pcm").exists()
    assert (dictation.job / "recording.wav").is_file()


def test_dictation_settings_restore_without_enabling_shortcut(window, dictation):
    from audio_transcriber.desktop_dictation import DictationWindow

    dictation.hotkey_text.setText("Ctrl+Shift+F9")
    dictation.mode.setCurrentText("Hold to talk")
    select(dictation.device, "GPU")
    dictation._save_settings()
    restored = DictationWindow(window)
    assert restored.hotkey_text.text() == "Ctrl+Shift+F9"
    assert restored.mode.currentText() == "Hold to talk"
    assert choice(restored.device) == "GPU"
    assert choice(restored.cleanup) == "off"
    assert not restored.enable.isChecked()
    restored.close()


def test_dictation_close_cancels_worker_without_destroying_it(
    app, window, dictation, three_second_wav
):
    prepare_dictation(dictation, three_second_wav)
    window.close()
    assert dictation.closing and dictation.cancelled
    assert (dictation.job / "stop").is_file()
    wait_finished(app, dictation)
    app.processEvents()
    assert not window.busy
    assert dictation.history.records()[0]["status"] == "cancelled"


def test_dictation_worker_requires_result_identifier(dictation):
    import uuid

    from PySide6.QtCore import QProcess

    dictation.ident = uuid.uuid4().hex
    dictation._state("processing")
    dictation._finished(0, QProcess.ExitStatus.NormalExit)
    assert "no session identifier" in dictation.status.text()
    assert dictation.history.records()[0]["status"] == "failed"


def test_dictation_cancellation_while_preparing_does_not_capture(app, dictation, monkeypatch):
    from audio_transcriber.insertion import FocusTarget

    captured = []
    monkeypatch.setattr(dictation, "_begin_capture", lambda: captured.append(True))
    dictation._state("preparing")
    dictation._task(lambda: FocusTarget(1, 2, "Test", True), dictation._target_ready)
    dictation.cancel()
    wait_finished(app, dictation)
    assert not captured


def test_microphone_polling_reports_device_error(dictation):
    from PySide6.QtMultimedia import QtAudio

    class BrokenSource:
        def error(self):
            return QtAudio.Error.IOError

        def stop(self):
            pass

        def deleteLater(self):
            pass

    dictation.mic.source = BrokenSource()
    dictation._state("recording")
    dictation._tick()
    assert not dictation.busy
    assert "stopped unexpectedly" in dictation.status.text()


def test_native_shortcut_accepts_both_qt_windows_event_routes(app):
    import ctypes
    from ctypes import wintypes

    from PySide6.QtCore import QByteArray

    from audio_transcriber.desktop_dictation import Shortcut

    called = []
    shortcut = Shortcut(None, lambda: called.append(True))
    shortcut.enabled = True
    msg = wintypes.MSG()
    msg.message = 0x0312
    msg.wParam = shortcut.IDENT
    for route in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
        assert shortcut.nativeEventFilter(QByteArray(route), ctypes.addressof(msg)) == (True, 0)
        app.processEvents()
    assert called == [True, True]
    msg.wParam += 1
    assert shortcut.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg)) == (False, 0)


def test_dropdowns_mark_defaults_explain_choices_and_save_values(window):
    from PySide6.QtCore import Qt

    assert window.cleanup_style.itemText(window.cleanup_style.findData("medium")) == (
        "Medium (recommended)"
    )
    assert window.format_style.itemText(window.format_style.findData("mixed")) == (
        "Mixed (recommended)"
    )
    assert window.cleanup_device.itemText(0) == "Automatic (recommended)"
    for combo in (window.cleanup_style, window.format_style, window.layer, window.export_format):
        assert combo.toolTip()
        assert all(combo.itemData(row, Qt.ItemDataRole.ToolTipRole) for row in range(combo.count()))
    select(window.export_format, "srt")
    assert window.export_format.currentText() == "Subtitles (.srt)"
    assert choice(window.export_format) == "srt"


def test_settings_menu_mirrors_hidden_choices_and_restores_recommended(window):
    menu = next(m for m in window.choice_menus if m.title().startswith("AI device"))
    assert menu.title() == "AI device: Automatic"  # The current choice is in the name.
    next(a for a in menu.actions() if a.data() == "GPU").trigger()
    assert choice(window.cleanup_device) == "GPU"
    select(window.cleanup_device, "NPU")
    assert next(a for a in menu.actions() if a.isChecked()).data() == "NPU"
    window.restore_recommended()
    assert choice(window.cleanup_device) == "AUTO"
    assert choice(window.device) == "CPU"


def test_stopped_transcription_and_paused_ai_step_offer_resume(window):
    from audio_transcriber.models import Session

    stale = Session(id="stale", title="Stale", status="processing", source_audio="")
    assert window.resumable(stale)  # The app closed mid-transcription.
    assert window.status_label(stale) == "Stopped"
    done = Session(id="done", title="Done", status="ready", source_audio="")
    assert not window.resumable(done)
    window.paused_jobs["done"] = ["summarize", "done", "--model", "m"]
    assert window.resumable(done)
    assert window.status_label(done) == "Ready · AI step paused"


def test_transcript_times_are_short_and_versions_show_as_tabs(app, window, three_second_wav):
    from audio_transcriber.desktop_ui import timestamp

    assert (timestamp(5.92), timestamp(754), timestamp(3725)) == ("0:05", "12:34", "1:02:05")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    assert window.transcript.toPlainText().startswith("0:00   synthetic transcript")
    # Only Original exists, so there is nothing to switch between.
    tabs = [window.version_tabs.tabText(i) for i in range(window.version_tabs.count())]
    assert tabs == ["Original"]
    assert window.version_tabs.isHidden()
    assert "whisper" not in window.details.text().lower()
    assert window.progress.isHidden()


def test_ai_tools_rerun_with_the_chosen_level(app, window, three_second_wav, tmp_path, monkeypatch):
    window.start_import(three_second_wav)
    wait_finished(app, window)
    use_fixture_model(
        monkeypatch,
        window,
        tmp_path,
        """
class Fixture:
    def generate(self, text, mode, style):
        return text.capitalize()+'.'
""",
    )
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "light")
    heavy = next(a for a in window.cleanup_action.menu().actions() if a.text() == "Heavy")
    heavy.trigger()
    wait_finished(app, window)
    assert (
        window.store.load_transcript(window.selected_id(), "ai").transformation["style"] == "heavy"
    )
    assert choice(window.cleanup_style) == "light"  # The import setting is unchanged.
    tabs = [window.version_tabs.tabText(i) for i in range(window.version_tabs.count())]
    assert "Transcript" in tabs and "Original" in tabs


def test_getting_started_opens_in_an_app_window(window):
    from PySide6.QtWidgets import QTextBrowser

    from audio_transcriber.desktop_ui import getting_started_page

    page = getting_started_page()
    assert page is not None
    window.show_guide(page)
    viewer = window.guide_dialog.findChild(QTextBrowser)
    assert window.guide_dialog.isVisible()
    assert "Import audio" in viewer.toPlainText()
    window.guide_dialog.close()


NOTES_FIXTURE = """
class Fixture:
    device='CPU'
    def generate(self, text, mode, style):
        if mode == 'format':
            return '- ' + text
        return text.capitalize()+'.'
"""


def wait_background(app, window, seconds=120):
    deadline = time.monotonic() + seconds
    while (window.background or window.bg_job) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window.background and window.bg_job is None, window.bg_output


def test_background_prepares_every_version_and_picks_are_instant(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    from audio_transcriber.desktop_versions import current_info
    from audio_transcriber.versions import saved_results

    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    window.background_enabled = True
    window.background_actions["levels"].setChecked(True)  # Not on by default.
    select(window.cleanup_style, "medium")
    select(window.format_style, "mixed")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    wait_background(app, window)
    session_id = window.selected_id()
    levels = {r.style for r in saved_results(window.store, session_id, "ai")}
    assert levels == {"light", "medium", "heavy"}
    notes = saved_results(window.store, session_id, "formatted")
    assert {(r.source_revision_id, r.style) for r in notes} >= {
        (r.revision_id, style)
        for r in saved_results(window.store, session_id, "ai")
        for style in ("prose", "mixed", "structured")
    }
    window.pick_cleanup("heavy")
    assert not window.busy  # Shown from the saved result, no model run.
    ai = current_info(window.store, session_id, "ai")
    assert ai["style"] == "heavy"
    formatted = current_info(window.store, session_id, "formatted")
    assert (formatted["style"], formatted["source_revision_id"]) == ("mixed", ai["revision_id"])
    window.pick_format("structured")
    assert not window.busy
    assert current_info(window.store, session_id, "formatted")["style"] == "structured"
    assert window.layer.currentText() == "Formatted"
    # Every level also has its summary ready.
    summaries = {r.source_revision_id for r in saved_results(window.store, session_id, "summary")}
    assert summaries >= {r.revision_id for r in saved_results(window.store, session_id, "ai")}
    # Redo all remakes what is shown now, then plans the rest again (older ones are stale).
    from audio_transcriber.desktop_versions import background_plan

    window.background_enabled = False
    before = current_info(window.store, session_id, "ai")["revision_id"]
    window.redo_all()
    wait_finished(app, window)
    after = current_info(window.store, session_id, "ai")
    assert after["revision_id"] != before and after["style"] == "heavy"
    plan = background_plan(window.store, session_id, "m", "mixed", window.redo_since[session_id])
    levels_planned = sorted(job[job.index("--style") + 1] for job in plan if job[0] == "cleanup")
    assert levels_planned == ["light", "medium"]


def test_notes_lists_use_bullet_then_circle_then_square(window):
    from PySide6.QtGui import QTextListFormat

    window._show_markdown("## Topic\n\n- point\n  - detail\n    - finer\n")
    styles = []
    block = window.transcript.document().begin()
    while block.isValid():
        if block.textList() is not None:
            styles.append(block.textList().format().style())
        block = block.next()
    assert styles == [
        QTextListFormat.Style.ListDisc,
        QTextListFormat.Style.ListCircle,
        QTextListFormat.Style.ListSquare,
    ]


def test_typing_in_notes_saves_my_copy_and_undo_restores_the_ai_version(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    from PySide6.QtGui import QTextCursor

    from audio_transcriber.versions import load_user_text

    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "medium")
    select(window.format_style, "mixed")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    assert window.layer.currentText() == "Formatted"
    assert not window.transcript.isReadOnly()
    cursor = window.transcript.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText(" My own addition.")
    window._save_edits()
    copy = load_user_text(window.store, window.selected_id(), "formatted")
    assert copy is not None and "My own addition." in copy.text
    assert window.revert_button.isVisibleTo(window)
    window.rendered_key = None  # Force a reload from disk.
    window.show_selected()
    assert "My own addition." in window.transcript.toPlainText()
    window.revert_edits()
    assert "My own addition." not in window.transcript.toPlainText()
    assert load_user_text(window.store, window.selected_id(), "formatted") is None


def test_typing_in_original_saves_an_edited_version(app, window, three_second_wav):
    from PySide6.QtGui import QTextCursor

    window.start_import(three_second_wav)
    wait_finished(app, window)
    session_id = window.selected_id()
    assert window.layer.currentText() == "Raw"
    assert not window.transcript.isReadOnly()
    cursor = QTextCursor(window.transcript.document().findBlockByNumber(0))
    cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
    cursor.insertText(" fixed")
    window._save_edits()
    assert window.layer.currentText() == "Edited"
    edited = window.store.load_session(session_id).edited_transcript
    assert edited is not None
    assert "synthetic transcript fixed" in window.transcript.toPlainText()
    assert window.store.load_transcript(session_id, "raw").text == "synthetic transcript"


def test_library_lines_rename_search_and_remembered_place(
    app, window, three_second_wav, monkeypatch
):
    from PySide6.QtWidgets import QInputDialog

    assert window.library_stack.currentIndex() == 0  # Empty library: the import prompt.
    window.start_import(three_second_wav)
    wait_finished(app, window)
    assert window.library_stack.currentIndex() == 1
    line = window.library.item(0).text()
    assert ".wav" not in line.splitlines()[0]
    assert "Ready" in line and "0:03" in line
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Week 1 lecture", True))
    window.rename_lecture()
    assert window.library.item(0).text().startswith("Week 1 lecture")
    window._filter_library("nothing like this")
    assert window.library.item(0).isHidden()
    window._filter_library("week")
    assert not window.library.item(0).isHidden()
    window._save_settings()
    reopened = LectureWindow(window.store.data_dir)
    assert reopened.selected_id() == window.selected_id()
    reopened.close()


def test_clicking_a_line_seeks_and_unclear_marker_toggles(app, window, three_second_wav):
    from PySide6.QtGui import QTextCursor

    window.start_import(three_second_wav)
    wait_finished(app, window)
    seeks = []
    window.player.seek = seeks.append
    window.segment_starts = [0.0, 1.5]
    window.cursor_segment = 0
    cursor = window.transcript.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    window.transcript.setTextCursor(cursor)  # Block 0 only exists in this fixture.
    window._cursor_moved()
    assert seeks == []  # Still on the first line: no jump.
    window.cursor_segment = 1
    window.transcript.setTextCursor(QTextCursor(window.transcript.document().findBlockByNumber(0)))
    assert seeks == [0.0]
    window.toggle_unclear(0)
    assert window.layer.currentText() == "Edited"
    assert "[unclear]" in window.transcript.toPlainText()
    window.toggle_unclear(0)
    assert "[unclear]" not in window.transcript.toPlainText()


def test_text_size_copy_and_export_menu(app, window, three_second_wav):
    from PySide6.QtWidgets import QApplication

    window.set_text_size(40)
    assert window.text_size == 32
    window.set_text_size(16)
    window.start_import(three_second_wav)
    wait_finished(app, window)
    window.copy_all()
    assert "synthetic transcript" in QApplication.clipboard().text()
    assert set(window.export_actions) == {"text", "markdown", "srt", "json"}
    assert window.export_actions["srt"].isEnabled()  # Original keeps its timing.


def test_delete_moves_to_trash_and_restore_brings_it_back(
    app, window, three_second_wav, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    from audio_transcriber.desktop_trash import TrashDialog
    from audio_transcriber.trash import list_trash

    window.start_import(three_second_wav)
    wait_finished(app, window)
    session_id = window.selected_id()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    window.delete_lecture()
    assert session_id in window.sessions  # Cancel keeps it.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.delete_lecture()
    assert session_id not in window.sessions
    assert window.library_stack.currentIndex() == 0  # Library now empty.
    assert "Trash" in window.status.text()
    assert [entry.id for entry in list_trash(window.store)] == [session_id]
    dialog = TrashDialog(window.store, window._trash_changed, window)
    assert dialog.list.count() == 1 and dialog.restore_button.isEnabled()
    dialog.restore()
    assert session_id in window.sessions
    assert window.status.text().startswith("Restored")
    assert list_trash(window.store) == []
    dialog.close()


def test_background_choices_decide_what_is_prepared(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    # Default: the shown cleanup level's other note and summary styles, no other levels.
    assert {k: a.isChecked() for k, a in window.background_actions.items()} == {
        "notes": True,
        "summaries": True,
        "levels": False,
    }
    window.background_enabled = True
    for action in window.background_actions.values():
        action.setChecked(False)  # Settings → nothing.
    select(window.cleanup_style, "medium")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    assert not window.background and window.bg_job is None

    def planned():
        app.processEvents()
        jobs = [*window.background, *([window.bg_job] if window.bg_job else [])]
        return {arguments[0] for _, arguments in jobs}

    window.background_actions["summaries"].setChecked(True)
    assert planned() == {"summarize"}
    window.background_actions["notes"].setChecked(True)
    assert planned() <= {"format", "summarize"} and "format" in planned()
    window.background_actions["levels"].setChecked(True)
    assert "cleanup" in planned()
    for action in window.background_actions.values():
        action.setChecked(False)
    assert not window.background
    saved = json.loads(window.settings_path.read_text(encoding="utf-8"))
    assert [saved[f"background_{kind}"] for kind in ("notes", "summaries", "levels")] == [
        "off",
        "off",
        "off",
    ]


def test_earlier_background_off_setting_is_kept(app, tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "desktop-settings.json").write_text('{"background_versions": "off"}')
    window = LectureWindow(root)
    try:
        assert not any(a.isChecked() for a in window.background_actions.values())
    finally:
        window.close()


def test_replacing_the_whole_text_saves_like_typing(app, window, three_second_wav):
    # Accessibility tools and paste-over-all replace the text in one step; Qt then reports
    # the document as unmodified, so the app keeps its own record of unsaved typing.
    window.start_import(three_second_wav)
    wait_finished(app, window)
    window.transcript.setPlainText("0:00   replaced in one step")
    window._save_edits()
    assert window.layer.currentText() == "Edited"
    assert "replaced in one step" in window.transcript.toPlainText()


def test_transcript_does_not_inherit_the_notes_heading_style(app, window, three_second_wav):
    from PySide6.QtGui import QFont, QTextCursor, QTextDocument

    window.start_import(three_second_wav)
    wait_finished(app, window)
    session = window.sessions[window.selected_id()]
    window._render(
        session,
        "Formatted",
        lambda: window.transcript.document().setMarkdown(
            "## A heading\n\nbody", QTextDocument.MarkdownFeature.MarkdownNoHTML
        ),
    )  # The cursor now sits on the heading.
    window._render(session, "AI", lambda: window.transcript.setPlainText("plain words"))
    cursor = QTextCursor(window.transcript.document())
    cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
    assert cursor.charFormat().fontWeight() <= QFont.Weight.Normal
    assert cursor.blockFormat().headingLevel() == 0


def test_headings_are_modest_and_typing_after_a_time_is_not_grey(app, window, three_second_wav):
    from PySide6.QtGui import QTextCursor, QTextFormat

    window._show_markdown("# Top\n\n## Second\n\n### Third\n\nbody")
    document = window.transcript.document()
    sizes = []
    for number in (0, 1, 2):
        cursor = QTextCursor(document.findBlockByNumber(number))
        cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
        sizes.append(cursor.charFormat().intProperty(QTextFormat.Property.FontSizeAdjustment))
    assert sizes == [1, 1, 0]
    window.start_import(three_second_wav)
    wait_finished(app, window)
    stamp_colour = window.palette().color(window.palette().ColorRole.PlaceholderText)
    line = window.transcript.document().findBlockByNumber(0)
    cursor = QTextCursor(line)
    cursor.setPosition(line.position() + len("0:00") + 1)  # Just after the time.
    assert cursor.charFormat().foreground().color() != stamp_colour


def test_dictation_mode_lecture_still_gets_a_summary(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    select(window.cleanup_mode, "dictation")
    select(window.cleanup_style, "medium")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    directory = window.store.session_dir(window.selected_id())
    assert (directory / "summary-transcript.json").is_file()
    assert window.mode_buttons["dictation"].isChecked()  # The switch shows the mode.


def test_every_nested_list_gets_the_sub_bullet_style(window):
    from PySide6.QtGui import QTextListFormat

    # Two sections, several points each with details: every detail list is ○.
    window._show_markdown(
        "## One\n\n- a\n  - a1\n  - a2\n- b\n  - b1\n- c\n  - c1\n  - c2\n\n"
        "## Two\n\n- d\n  - d1\n- e\n  - e1\n"
    )
    seen = {}
    block = window.transcript.document().begin()
    while block.isValid():
        if block.textList() is not None:
            seen[block.text()] = block.textList().format().style()
        block = block.next()
    for text, style in seen.items():
        expected = (
            QTextListFormat.Style.ListCircle if len(text) == 2 else QTextListFormat.Style.ListDisc
        )
        assert style == expected, text


def test_tabs_order_and_format_applies_to_notes_and_summary(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    from audio_transcriber.desktop_versions import current_info

    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "medium")
    select(window.format_style, "mixed")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    tabs = [window.version_tabs.tabText(i) for i in range(window.version_tabs.count())]
    assert tabs == ["Original", "Transcript", "Notes", "Summary"]
    assert "Summary" not in [a.text() for a in window.format_button.menu().actions()]
    session_id = window.selected_id()
    window.layer.setCurrentText("Summary")
    window.pick_format("structured")  # Not saved yet: notes, then the summary, are made.
    wait_finished(app, window)
    assert current_info(window.store, session_id, "formatted")["style"] == "structured"
    assert current_info(window.store, session_id, "summary")["style"] == "structured"
    assert window.layer.currentText() == "Summary"  # Stayed on the tab being read.
    window.pick_format("mixed")  # Both saved from the import: shown without a job.
    assert not window.busy
    assert current_info(window.store, session_id, "summary")["style"] == "mixed"


def test_recording_type_switch_is_remembered_and_recleans_the_selected_lecture(
    app, window, three_second_wav, tmp_path, monkeypatch
):
    import json

    from audio_transcriber.desktop_versions import current_info

    use_fixture_model(monkeypatch, window, tmp_path, NOTES_FIXTURE)
    install_cleanup_placeholder(window)
    select(window.cleanup_style, "medium")
    select(window.format_style, "mixed")
    window.start_import(three_second_wav)
    wait_finished(app, window)
    session_id = window.selected_id()
    assert current_info(window.store, session_id, "ai")["mode"] == "lecture"
    window.mode_buttons["dictation"].click()
    wait_finished(app, window)
    assert current_info(window.store, session_id, "ai")["mode"] == "dictation"
    saved = json.loads(window.settings_path.read_text(encoding="utf-8"))
    assert saved["cleanup_mode"] == "dictation"
    window.mode_buttons["lecture"].click()  # Saved in Lecture mode already: no new job.
    assert not window.busy
    assert current_info(window.store, session_id, "ai")["mode"] == "lecture"
