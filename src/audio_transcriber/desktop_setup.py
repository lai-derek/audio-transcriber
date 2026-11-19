"""Graphical model setup and user-initiated application updates."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_path
from PySide6.QtCore import QProcess, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from . import __version__
from .acquisition import MANIFEST, download_model
from .storage import atomic_json
from .updates import RELEASES_URL, Release, download_installer, latest_release

if TYPE_CHECKING:
    from .desktop_ui import LectureWindow


class DownloadTask(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(object, object)

    def __init__(self, action: Callable[[DownloadTask], Any], parent: QDialog) -> None:
        super().__init__(parent)
        self.action = action
        self.cancelled = threading.Event()

    def run(self) -> None:
        try:
            self.result.emit(self.action(self))
        except Exception as error:
            self.failed.emit(str(error))


class ModelDialog(QDialog):
    def __init__(self, owner: LectureWindow) -> None:
        super().__init__(owner)
        self.owner = owner
        self.task: DownloadTask | None = None
        self.setWindowTitle("Manage models")
        self.resize(520, 270)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Download a speech model to start transcribing. AI cleanup is optional. "
            "After download, models work offline."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.model = QComboBox()
        self.model.setAccessibleName("Model to download")
        preferred = ["whisper-base.en-int4-ov", "whisper-tiny.en-int4-ov"]
        for model_id in [*preferred, *(m for m in MANIFEST if m not in preferred)]:
            spec = MANIFEST[model_id]
            label = (
                "Speech — recommended for lectures"
                if model_id == preferred[0]
                else "Speech — smaller download"
                if model_id == preferred[1]
                else "AI cleanup (optional)"
            )
            self.model.addItem(f"{label}: {spec.download_bytes / 1024**2:,.0f} MB", model_id)
        layout.addWidget(self.model)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("Choose a model, then click Download.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.download = QPushButton("Download")
        self.download.clicked.connect(self.start)
        buttons.addWidget(self.download)
        self.cancel = QPushButton("Cancel download")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_download)
        buttons.addWidget(self.cancel)
        close = QPushButton("Done")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.model.currentIndexChanged.connect(self.describe)
        self.describe()

    def describe(self) -> None:
        spec = MANIFEST[str(self.model.currentData())]
        installed = (self.owner.model_root / spec.id).is_dir()
        self.detail.setText(
            f"{spec.id}\nLicense: {spec.license}. "
            + (
                "Download checks or repairs these files. Replaced files are kept in a "
                ".previous- folder. "
                if installed
                else ""
            )
            + (
                "Cleanup uses several GiB of memory."
                if spec.task == "text-cleanup"
                else "English speech recognition."
            )
        )

    def start(self) -> None:
        if self.task or self.owner.busy:
            self.status.setText("Wait until the current task finishes.")
            return
        model_id = str(self.model.currentData())
        self.status.setText("Downloading and verifying model…")
        self.progress.setValue(0)
        self.model.setEnabled(False)
        self.download.setEnabled(False)
        self.cancel.setEnabled(True)
        task = DownloadTask(
            lambda t: download_model(
                model_id,
                self.owner.model_root,
                progress=t.progress.emit,
                cancelled=t.cancelled.is_set,
            ),
            self,
        )
        self.task = task
        task.progress.connect(lambda n, total: self.progress.setValue(int(100 * n / total)))
        task.failed.connect(self.status.setText)
        task.result.connect(lambda _: self.installed(model_id))
        task.finished.connect(self.finished_task)
        task.start()
        self.owner._buttons()

    def installed(self, model_id: str) -> None:
        if MANIFEST[model_id].task == "speech":
            self.owner.model.setCurrentIndex(self.owner.model.findData(model_id))
        self.owner._save_settings()
        self.status.setText("Model ready. You can close this window and transcribe.")
        self.progress.setValue(100)
        self.describe()

    def finished_task(self) -> None:
        task, self.task = self.task, None
        if task:
            task.deleteLater()
        self.model.setEnabled(True)
        self.download.setEnabled(True)
        self.cancel.setEnabled(False)
        self.owner._buttons()

    def cancel_download(self) -> None:
        if self.task:
            self.task.cancelled.set()
            self.status.setText("Cancelling download…")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.task:
            self.cancel_download()
            event.ignore()
        else:
            event.accept()


class UpdateDialog(QDialog):
    def __init__(self, owner: LectureWindow) -> None:
        super().__init__(owner)
        self.owner = owner
        self.task: DownloadTask | None = None
        self.release: Release | None = None
        self.installer: Path | None = None
        self.pending_close = False
        self.waiting = False
        self.cache = user_cache_path("audio-transcriber", appauthor=False) / "updates"
        self.settings_path = owner.store.data_dir / "update-settings.json"
        self.settings: dict[str, Any] = {"automatic": True, "last_check": 0}
        try:
            value = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.settings["automatic"] = value.get("automatic") is not False
                checked = value.get("last_check", 0)
                self.settings["last_check"] = checked if type(checked) in (int, float) else 0
        except OSError, ValueError:
            pass
        self.setWindowTitle("Audio Transcriber updates")
        self.resize(520, 350)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Installed version: {__version__}"))
        self.automatic = QCheckBox("Check for updates automatically (once a day)")
        self.automatic.setChecked(bool(self.settings["automatic"]))
        self.automatic.toggled.connect(self.save_automatic)
        layout.addWidget(self.automatic)
        self.status = QLabel("Check for a newer version.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        layout.addWidget(self.notes)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        self.check = QPushButton("Check for updates")
        self.check.clicked.connect(lambda: self.check_now(True))
        row.addWidget(self.check)
        self.update_button = QPushButton("Update and reopen")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self.download)
        row.addWidget(self.update_button)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_update)
        row.addWidget(self.cancel)
        later = QPushButton("Later")
        later.clicked.connect(self.defer)
        row.addWidget(later)
        layout.addLayout(row)
        self.poll = QTimer(self)
        self.poll.setInterval(500)
        self.poll.timeout.connect(self.install_when_idle)

    def save_automatic(self, checked: bool) -> None:
        self.settings["automatic"] = checked
        self.save()

    def save(self) -> None:
        try:
            atomic_json(self.settings_path, self.settings)
        except OSError as error:
            self.status.setText(f"Could not save update preferences: {error}")

    def auto_check(self) -> None:
        if getattr(sys, "frozen", False) and self.settings["automatic"]:
            if time.time() - self.settings["last_check"] >= 86400:
                self.check_now(False)

    def check_now(self, show: bool) -> None:
        if self.task or self.waiting:
            return
        if not getattr(sys, "frozen", False):
            QDesktopServices.openUrl(QUrl(RELEASES_URL))
            return
        if show:
            self.show()
        self.release = None
        self.update_button.setEnabled(False)
        self.status.setText("Checking for updates…")
        self.settings["last_check"] = time.time()
        self.save()
        task = DownloadTask(lambda _: latest_release(__version__), self)
        task.result.connect(self.found)
        task.failed.connect(self.status.setText)
        self.start_task(task)

    def found(self, result: Release | None) -> None:
        if self.task and self.task.cancelled.is_set():
            return
        self.release = result
        if result is None:
            self.status.setText("No newer Windows release is available.")
            return
        self.status.setText(
            f"Version {result.version} is available ({result.size / 1024**2:,.0f} MB)."
        )
        self.notes.setPlainText(result.notes)
        self.update_button.setEnabled(True)
        self.show()

    def start_task(self, task: DownloadTask) -> None:
        self.task = task
        self.check.setEnabled(False)
        task.finished.connect(self.finished_task)
        task.start()

    def finished_task(self) -> None:
        task, self.task = self.task, None
        if task:
            task.deleteLater()
        self.check.setEnabled(not self.waiting)
        self.update_button.setEnabled(self.release is not None and not self.waiting)
        if self.pending_close:
            self.owner.close()

    def download(self) -> None:
        if self.task or self.release is None or self.waiting:
            return
        release = self.release
        self.status.setText("Downloading update…")
        self.update_button.setEnabled(False)
        self.progress.setValue(0)
        task = DownloadTask(
            lambda t: download_installer(
                release, self.cache / release.version, t.progress.emit, t.cancelled.is_set
            ),
            self,
        )
        task.progress.connect(lambda n, total: self.progress.setValue(int(100 * n / total)))
        task.failed.connect(self.status.setText)
        task.result.connect(self.downloaded)
        self.start_task(task)

    def downloaded(self, installer: Path) -> None:
        if self.task and self.task.cancelled.is_set():
            return
        self.installer = installer
        self.waiting = True
        self.status.setText("Update ready. Waiting for current work to finish before reopening…")
        self.poll.start()

    def cancel_update(self) -> None:
        if self.task:
            self.task.cancelled.set()
        self.waiting = False
        self.poll.stop()
        if self.task is None:
            self.check.setEnabled(True)
            self.update_button.setEnabled(self.release is not None)
        self.status.setText("Update cancelled. The installed app has not changed.")

    def defer(self) -> None:
        self.cancel_update()
        self.close()

    def install_when_idle(self) -> None:
        if QApplication.activeModalWidget() is not None:
            return
        if self.task or self.owner.busy or self.installer is None or self.release is None:
            return
        self.poll.stop()
        try:
            self.owner._save_settings()
            if self.owner.dictation_window is not None:
                self.owner.dictation_window.close()
            helper = Path(sys.executable).with_name("audio-transcriber-updater.exe")
            target = self.installer.parent / helper.name
            shutil.copy2(helper, target)
            arguments = [
                "--data-dir",
                str(self.owner.store.data_dir),
                "--model-root",
                str(self.owner.model_root),
            ]
            if self.owner.ffmpeg:
                arguments += ["--ffmpeg", self.owner.ffmpeg]
            request = target.parent / "request.json"
            atomic_json(
                request,
                {
                    "version": self.release.version,
                    "size": self.release.size,
                    "sha256": self.release.sha256,
                    "install_dir": str(Path(sys.executable).parent),
                    "arguments": arguments,
                },
            )
            started, _ = QProcess.startDetached(str(target), [str(request)], str(target.parent))
            if not started:
                raise OSError("The update helper could not start.")
            self.waiting = False
            self.owner.close()
        except OSError as error:
            self.waiting = False
            self.status.setText(f"Could not start installation: {error}. Click Update to retry.")
            self.check.setEnabled(True)
            self.update_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Closing this window never destroys a running download thread.
        event.accept()
