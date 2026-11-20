from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from audio_transcriber.desktop_setup import choice  # noqa: E402
from audio_transcriber.desktop_ui import LectureWindow  # noqa: E402
from audio_transcriber.updates import Release  # noqa: E402

PARAKEET = "parakeet-tdt-0.6b-v2-int8"
SMALL = "whisper-small.en-int4-ov"
BASE = "whisper-base.en-int4-ov"
TINY = "whisper-tiny.en-int4-ov"


def install(root: Path, *model_ids: str) -> None:
    from audio_transcriber.acquisition import MANIFEST

    for model_id in model_ids:
        folder = root / model_id
        folder.mkdir(parents=True, exist_ok=True)
        for entry in MANIFEST[model_id].files:
            (folder / entry.name).touch()


def offered(combo) -> list[str]:
    return [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i)]


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = LectureWindow(tmp_path / "library")
    yield app, window
    window.close()


def wait(app, predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def fake_downloads(monkeypatch, calls, fail=None):
    """Install the manifest's file names (empty) instead of downloading."""
    from audio_transcriber.acquisition import MANIFEST, AcquisitionError

    def download(model_id, root, *, progress, cancelled, checking=None):
        calls.append(model_id)
        if model_id == fail:
            raise AcquisitionError("network down")
        folder = root / model_id
        folder.mkdir(parents=True, exist_ok=True)
        for entry in MANIFEST[model_id].files:
            (folder / entry.name).touch()
        progress(4, 4)
        return folder

    monkeypatch.setattr("audio_transcriber.desktop_setup.download_model", download)


def test_first_launch_downloads_everything_and_enables_ai(window, monkeypatch):
    import json

    from audio_transcriber.ai_cleanup import DEFAULT_MODEL
    from audio_transcriber.desktop_setup import MODEL_SETUP_VERSION

    app, owner = window
    calls = []
    fake_downloads(monkeypatch, calls)
    owner.first_launch()
    dialog = owner.setup_dialog
    assert dialog.isVisible()
    wait(app, lambda: dialog.task is None)
    # Speech first, recommended first, so transcription can start while the rest download.
    assert calls == [PARAKEET, SMALL, BASE, DEFAULT_MODEL]  # Never Whisper Tiny.
    assert owner.model.currentData() == PARAKEET
    assert choice(owner.cleanup_style) == "medium"
    assert owner.format_style.currentData() == "mixed"
    assert not dialog.checked()
    assert "Models ready" in dialog.status.text()
    saved = json.loads(owner.settings_path.read_text())
    assert saved["model_setup"] == MODEL_SETUP_VERSION
    assert saved["speech_default"] == PARAKEET
    dialog.close()
    calls.clear()
    owner.first_launch()  # Completed setup is not repeated.
    assert calls == []


def test_failed_queue_stops_and_resumes_on_next_launch(window, monkeypatch):
    app, owner = window
    calls = []
    fake_downloads(monkeypatch, calls, fail=SMALL)
    owner.first_launch()
    dialog = owner.setup_dialog
    wait(app, lambda: dialog.task is None)
    assert calls == [PARAKEET, SMALL]
    assert "Speech — Whisper Small: network down" in dialog.status.text()
    assert owner.model_installed(PARAKEET)
    # The finished model is unchecked; the failed and remaining ones stay queued.
    assert PARAKEET not in dialog.checked()
    assert dialog.checked()[0] == SMALL
    assert not owner.model_setup_done
    calls.clear()
    fake_downloads(monkeypatch, calls)
    owner.first_launch()
    wait(app, lambda: dialog.task is None)
    assert PARAKEET not in calls  # Installed models are not repeated.
    assert owner.model_setup_done


def test_downloads_do_not_block_transcription_controls(window):
    _, owner = window
    owner.manage_models()
    dialog = owner.setup_dialog

    class Running:
        pass

    dialog.task = Running()
    owner._buttons()
    assert owner.import_button.isEnabled()
    assert not owner.models_action.isEnabled()
    assert owner.busy and not owner.processing
    dialog.task = None


def test_stopping_setup_keeps_it_from_restarting(window, monkeypatch):
    _, owner = window
    owner.manage_models()
    dialog = owner.setup_dialog

    class Running:
        import threading

        cancelled = threading.Event()

    dialog.task = Running()
    dialog.cancel_download()
    assert Running.cancelled.is_set()
    assert owner.model_setup_done
    dialog.task = None


def test_update_waits_for_work_and_cancel_is_reversible(window, tmp_path):
    _, owner = window
    dialog = owner.updates
    dialog.release = Release("0.2.1", "", "", 1, "0" * 64)
    dialog.installer = tmp_path / "installer.exe"
    dialog.waiting = True
    owner._job_active = True
    dialog.install_when_idle()
    assert dialog.waiting
    assert not (tmp_path / "request.json").exists()
    dialog.cancel_update()
    owner._job_active = False
    assert not dialog.waiting
    assert dialog.check.isEnabled()
    assert dialog.update_button.isEnabled()


def test_later_postpones_a_pending_update(window, tmp_path):
    _, owner = window
    dialog = owner.updates
    dialog.release = Release("0.2.1", "", "", 1, "0" * 64)
    dialog.installer = tmp_path / "installer.exe"
    dialog.waiting = True
    dialog.poll.start()
    dialog.show()
    dialog.defer()
    assert not dialog.waiting
    assert not dialog.poll.isActive()
    assert not dialog.isVisible()


def test_update_handoff_preserves_library_and_runs_cached_helper(window, tmp_path, monkeypatch):
    import json
    import sys

    from PySide6.QtCore import QProcess

    _, owner = window
    install = tmp_path / "installed"
    install.mkdir()
    executable = install / "Audio Transcriber.exe"
    executable.touch()
    (install / "audio-transcriber-updater.exe").write_bytes(b"helper fixture")
    monkeypatch.setattr(sys, "executable", str(executable))
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []
    monkeypatch.setattr(QProcess, "startDetached", lambda *args: (calls.append(args) is None, 123))
    dialog = owner.updates
    dialog.release = Release("0.2.1", "", "", 1, "0" * 64)
    dialog.installer = cache / dialog.release.filename
    dialog.waiting = True
    dialog.install_when_idle()
    request = json.loads((cache / "request.json").read_text())
    assert request["arguments"][:2] == ["--data-dir", str(owner.store.data_dir)]
    assert request["install_dir"] == str(install)
    assert Path(calls[0][0]).parent == cache
    assert (cache / "audio-transcriber-updater.exe").read_bytes() == b"helper fixture"


def test_network_failure_is_shown_and_retry_enabled(window, monkeypatch):
    import sys

    app, owner = window
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def fail(_):
        raise OSError("offline")

    monkeypatch.setattr("audio_transcriber.desktop_setup.latest_release", fail)
    owner.updates.check_now(True)
    wait(app, lambda: owner.updates.task is None)
    assert "offline" in owner.updates.status.text()
    assert owner.updates.check.isEnabled()


def test_status_names_models_and_separates_checking(window):
    from audio_transcriber.ai_cleanup import DEFAULT_MODEL

    _, owner = window
    owner.manage_models()
    dialog = owner.setup_dialog
    dialog.started(1, 2, DEFAULT_MODEL)
    assert dialog.status.text() == "Downloading the AI cleanup model…"
    dialog.checking(True)
    assert dialog.status.text() == "Checking the files for the AI cleanup model…"
    dialog.checking(False)
    assert dialog.status.text() == "Downloading the AI cleanup model…"


def test_offers_parakeet_small_and_base_but_never_tiny(window, monkeypatch):
    from audio_transcriber.ai_cleanup import DEFAULT_MODEL

    app, owner = window
    assert offered(owner.model) == [PARAKEET, SMALL, BASE]
    assert owner.model.itemText(owner.model.findData(PARAKEET)) == "Parakeet v2 (recommended)"
    # Tiny installed from an older version is neither offered nor picked.
    install(owner.model_root, TINY, BASE)
    owner.manage_models()
    dialog = owner.setup_dialog
    listed = [dialog.models.item(i).data(256) for i in range(dialog.models.count())]
    assert listed == [PARAKEET, SMALL, BASE, DEFAULT_MODEL]
    assert dialog.models.item(0).text().startswith("Speech — Parakeet v2 (recommended)")
    calls = []
    fake_downloads(monkeypatch, calls)
    dialog.close()
    owner.first_launch()
    wait(app, lambda: dialog.task is None)
    assert TINY not in calls and BASE not in calls
    owner.open_dictation()
    assert offered(owner.dictation_window.model) == [PARAKEET, SMALL, BASE]
    owner.dictation_window.close()


def test_restore_recommended_selects_parakeet(window):
    _, owner = window
    install(owner.model_root, PARAKEET, BASE)
    owner.model.setCurrentIndex(owner.model.findData(BASE))
    owner.restore_recommended()
    assert owner.model.currentData() == PARAKEET


def test_device_choice_does_not_apply_to_parakeet(window):
    from audio_transcriber.desktop_setup import CPU_ONLY_HINT, DEVICE_HINT

    _, owner = window
    owner.model.setCurrentIndex(owner.model.findData(PARAKEET))
    assert not owner.device.isEnabled()
    assert owner.device.toolTip() == CPU_ONLY_HINT
    owner.model.setCurrentIndex(owner.model.findData(SMALL))
    assert owner.device.isEnabled()
    assert owner.device.toolTip() == DEVICE_HINT
    owner.open_dictation()
    dictation = owner.dictation_window
    dictation.model.setCurrentIndex(dictation.model.findData(PARAKEET))
    assert not dictation.device.isEnabled()
    dictation.model.setCurrentIndex(dictation.model.findData(BASE))
    assert dictation.device.isEnabled()
    dictation.close()


def write_settings(data: Path, lecture: dict, dictation: dict | None = None) -> None:
    import json

    data.mkdir(parents=True, exist_ok=True)
    (data / "desktop-settings.json").write_text(json.dumps(lecture), encoding="utf-8")
    if dictation is not None:
        (data / "dictation").mkdir(exist_ok=True)
        (data / "dictation" / "settings.json").write_text(json.dumps(dictation), encoding="utf-8")


@pytest.mark.parametrize("parakeet_installed", [False, True])
def test_existing_install_switches_to_parakeet_once(tmp_path, monkeypatch, parakeet_installed):
    import json

    from audio_transcriber.ai_cleanup import DEFAULT_MODEL
    from audio_transcriber.desktop_setup import MODEL_SETUP_VERSION

    app = QApplication.instance() or QApplication([])
    data = tmp_path / "library"
    # Settings from before Parakeet v2 was offered: setup "done", Whisper Base chosen.
    write_settings(
        data,
        {"model_setup": "done", "model": BASE, "device": "CPU"},
        {"model": BASE, "device": "CPU", "cleanup": "off"},
    )
    install(data / "models", TINY, BASE, DEFAULT_MODEL, *([PARAKEET] if parakeet_installed else []))
    owner = LectureWindow(data)
    try:
        assert not owner.model_setup_done
        assert owner.model.currentData() == BASE
        calls = []
        fake_downloads(monkeypatch, calls)
        owner.first_launch()
        wait(app, lambda: owner.setup_dialog is not None and owner.setup_dialog.task is None)
        # Only the newly offered models download.
        assert calls == ([SMALL] if parakeet_installed else [PARAKEET, SMALL])
        assert owner.model.currentData() == PARAKEET
        dictation = json.loads((data / "dictation" / "settings.json").read_text())
        assert dictation["model"] == PARAKEET
        assert dictation["cleanup"] == "off"  # Other dictation choices are kept.
        saved = json.loads(owner.settings_path.read_text())
        assert saved["model_setup"] == MODEL_SETUP_VERSION
        # The user's later choice stays.
        owner.model.setCurrentIndex(owner.model.findData(BASE))
        owner._save_settings()
    finally:
        if owner.setup_dialog is not None:
            owner.setup_dialog.close()
        owner.close()
    again = LectureWindow(data)
    try:
        calls.clear()
        again.first_launch()
        app.processEvents()
        assert calls == []
        assert again.model.currentData() == BASE
    finally:
        again.close()


def test_stopped_migration_does_not_switch_later(tmp_path):
    QApplication.instance() or QApplication([])
    data = tmp_path / "library"
    write_settings(data, {"model_setup": "done", "model": BASE})
    install(data / "models", BASE)
    owner = LectureWindow(data)
    try:
        owner.finish_model_setup()  # As Stop download does.
        install(owner.model_root, PARAKEET)
        owner.manage_models()
        owner.setup_dialog.installed(PARAKEET)  # A later manual download.
        assert owner.model.currentData() == BASE
    finally:
        owner.setup_dialog.close()
        owner.close()
