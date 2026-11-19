from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from audio_transcriber.desktop_ui import LectureWindow  # noqa: E402
from audio_transcriber.updates import Release  # noqa: E402


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


def test_first_launch_and_graphical_download_select_model(window, monkeypatch):
    app, owner = window
    calls = []

    def download(model_id, root, *, progress, cancelled):
        calls.append((model_id, root))
        progress(4, 4)
        return root / model_id

    monkeypatch.setattr("audio_transcriber.desktop_setup.download_model", download)
    owner.first_launch()
    dialog = owner.setup_dialog
    assert dialog.isVisible()
    assert dialog.model.currentData() == "whisper-base.en-int4-ov"
    assert owner.cleanup_style.currentText() == "off"
    dialog.start()
    wait(app, lambda: dialog.task is None)
    assert owner.model.currentData() == calls[0][0]
    assert "Model ready" in dialog.status.text()
    dialog.close()


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
