"""Graphical model setup and user-initiated application updates."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_path
from PySide6.QtCore import QProcess, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from . import __version__
from .acquisition import (
    DEFAULT_SPEECH_MODEL,
    MANIFEST,
    PARAKEET_V2,
    download_model,
    partial_download,
)
from .ai_cleanup import STYLES
from .storage import atomic_json
from .updates import RELEASES_URL, Release, download_installer, latest_release

if TYPE_CHECKING:
    from .desktop_ui import LectureWindow


class DownloadTask(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(object, object)
    step = Signal(object, object)  # (queue position, model ID) as each model starts
    downloaded = Signal(object)  # model ID once it is verified and installed
    checking = Signal(bool)  # True while a noticeable checksum pass runs

    def __init__(self, action: Callable[[DownloadTask], Any], parent: QDialog) -> None:
        super().__init__(parent)
        self.action = action
        self.cancelled = threading.Event()

    def run(self) -> None:
        try:
            self.result.emit(self.action(self))
        except Exception as error:
            self.failed.emit(str(error))


# Speech models the app offers, recommended first. Whisper Tiny stays in the manifest
# for tests and the command line but is not offered here.
SPEECH_MODELS = (DEFAULT_SPEECH_MODEL, "whisper-small.en-int4-ov", "whisper-base.en-int4-ov")
CLEANUP_MODELS = tuple(m for m, spec in MANIFEST.items() if spec.task == "text-cleanup")
OFFERED_MODELS = (*SPEECH_MODELS, *CLEANUP_MODELS)
# Saved as "model_setup" once automatic setup finishes. A new value makes the next launch
# download newly offered models; "done" was the value before Parakeet v2 was offered.
MODEL_SETUP_VERSION = "parakeet-v2"
# (dropdown label, hover explanation, status-sentence name) per offered speech model.
_SPEECH_TEXT = {
    PARAKEET_V2: (
        "Parakeet v2",
        "The most accurate and fastest. Always runs on the CPU.",
        "the Parakeet v2 speech model",
    ),
    "whisper-small.en-int4-ov": (
        "Whisper Small",
        "Accurate, but slower than Parakeet v2.",
        "the Whisper Small speech model",
    ),
    "whisper-base.en-int4-ov": (
        "Whisper Base",
        "Small download, but less accurate.",
        "the Whisper Base speech model",
    ),
}


def model_label(model_id: str) -> str:
    """Name in the Manage models list."""
    if model_id in _SPEECH_TEXT:
        label = _SPEECH_TEXT[model_id][0]
        return f"Speech — {label}" + (" (recommended)" if model_id == DEFAULT_SPEECH_MODEL else "")
    return "AI cleanup (optional)"


def model_choice(model_id: str) -> tuple[str, str, str]:
    """(value, dropdown label, hover explanation) for a model in a settings dropdown."""
    if model_id in _SPEECH_TEXT:
        label, explanation, _ = _SPEECH_TEXT[model_id]
        return model_id, label, explanation
    if model_id == "qwen2.5-7b-instruct-int4-ov":
        return model_id, "Qwen 2.5 (7B)", "The AI model for cleanup, formatting, and summaries."
    return model_id, model_id, ""


# Dropdown choices: (saved value, label, explanation shown when hovering the choice).
DEVICE_HINT = "Which part of the computer runs transcription."
CPU_ONLY_HINT = "Parakeet v2 always runs on the CPU, so this setting does not apply to it."
SPEECH_DEVICES = (
    ("CPU", "CPU", "Works on every computer."),
    ("GPU", "GPU", "The graphics chip. Can be faster if your computer has one."),
    ("NPU", "NPU", "The AI chip in some newer laptops."),
)
AI_DEVICES = (
    ("AUTO", "Automatic", "Uses the graphics chip if available, otherwise the CPU."),
    *SPEECH_DEVICES,
)
LEVEL_HELP = {
    "light": "Removes ums, repeats, and filler, and fixes punctuation.",
    "medium": "Also fixes false starts and run-on sentences.",
    "heavy": "Rewrites for easy reading. Keeps every point; usually shorter.",
}
CLEANUP_LEVELS = (
    ("off", "Off", "No AI cleanup. The transcript stays exactly as heard."),
    *((level, level.capitalize(), LEVEL_HELP.get(level, "")) for level in STYLES),
)


def add_choices(
    combo: QComboBox, choices: Iterable[tuple[str, str, str]], recommended: str = ""
) -> None:
    """Add labelled choices; code reads the saved value with choice(), never the label."""
    for value, label, explanation in choices:
        combo.addItem(f"{label} (recommended)" if value == recommended else label, value)
        if explanation:
            combo.setItemData(combo.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole)


def choice(combo: QComboBox) -> str:
    return str(combo.currentData() or "")


def select(combo: QComboBox, value: str) -> bool:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
    return index >= 0


def model_name(model_id: str) -> str:
    """Name for status sentences, e.g. "Downloading the AI cleanup model"."""
    if model_id in _SPEECH_TEXT:
        return _SPEECH_TEXT[model_id][2]
    return "the AI cleanup model"


def size_text(count: int) -> str:
    return f"{count / 1024**3:.1f} GB" if count >= 1024**3 else f"{count / 1024**2:,.0f} MB"


class ModelDialog(QDialog):
    """Checked models download one after another; missing models start checked."""

    def __init__(self, owner: LectureWindow) -> None:
        super().__init__(owner)
        self.owner = owner
        self.task: DownloadTask | None = None
        self.current = ""
        self.setWindowTitle("Manage models")
        self.resize(560, 340)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Checked models download one after another. A speech model is needed to "
            "transcribe; AI cleanup adds cleanup, summaries, and layouts. After download, "
            "models work offline."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.models = QListWidget()
        self.models.setAccessibleName("Models to download")
        for model_id in OFFERED_MODELS:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if owner.model_installed(model_id)
                else Qt.CheckState.Checked
            )
            self.models.addItem(item)
        self.models.setFixedHeight(
            self.models.sizeHintForRow(0) * self.models.count() + 2 * self.models.frameWidth() + 4
        )
        layout.addWidget(self.models)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.download = QPushButton("Download")
        self.download.clicked.connect(self.start)
        buttons.addWidget(self.download)
        self.cancel = QPushButton("Stop download")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_download)
        buttons.addWidget(self.cancel)
        close = QPushButton("Done")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.models.currentRowChanged.connect(self.describe)
        self.models.itemChanged.connect(self.selection_changed)
        self.models.setCurrentRow(0)
        self.refresh()

    def model_item(self, model_id: str) -> QListWidgetItem:
        for row in range(self.models.count()):
            item = self.models.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == model_id:
                return item
        raise KeyError(model_id)

    def checked(self) -> list[str]:
        return [
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in map(self.models.item, range(self.models.count()))
            if item.checkState() == Qt.CheckState.Checked
        ]

    def refresh(self, keep_status: bool = False) -> None:
        """Show each model's size and whether it is installed or partly downloaded."""
        self.models.blockSignals(True)
        for row in range(self.models.count()):
            item = self.models.item(row)
            model_id = str(item.data(Qt.ItemDataRole.UserRole))
            state = (
                " · Installed"
                if self.owner.model_installed(model_id)
                else " · Partly downloaded"
                if partial_download(self.owner.model_root, model_id)
                else ""
            )
            size = size_text(MANIFEST[model_id].download_bytes)
            item.setText(f"{model_label(model_id)} · {size}{state}")
        self.models.blockSignals(False)
        self.describe()
        if not keep_status:
            self.selection_changed()

    def selection_changed(self, *_: object) -> None:
        if self.task:
            return
        queue = self.checked()
        self.download.setEnabled(bool(queue))
        total = sum(MANIFEST[m].download_bytes for m in queue)
        self.status.setText(
            f"{len(queue)} selected · {size_text(total)}."
            if queue
            else "All set, or check a model to download or repair it."
        )

    def describe(self, *_: object) -> None:
        item = self.models.currentItem()
        if item is None:
            return
        spec = MANIFEST[str(item.data(Qt.ItemDataRole.UserRole))]
        self.detail.setText(
            f"{spec.id}\nLicense: {spec.license}. "
            + (
                "Download checks or repairs these files. Replaced files are kept in a "
                ".previous- folder. "
                if (self.owner.model_root / spec.id).is_dir()
                else ""
            )
            + (
                "An earlier download will continue where it stopped. "
                if partial_download(self.owner.model_root, spec.id)
                else ""
            )
            + (
                "Cleanup uses several GiB of memory."
                if spec.task == "text-cleanup"
                else "English speech recognition."
            )
        )

    def start(self) -> None:
        if self.task:
            return
        queue = self.checked()
        if not queue:
            self.status.setText("Check the models to download.")
            return
        root = self.owner.model_root

        def run(task: DownloadTask) -> list[str]:
            for index, model_id in enumerate(queue):
                task.step.emit(index, model_id)
                download_model(
                    model_id,
                    root,
                    progress=task.progress.emit,
                    cancelled=task.cancelled.is_set,
                    checking=task.checking.emit,
                )
                task.downloaded.emit(model_id)
            return queue

        self.models.setEnabled(False)
        self.download.setEnabled(False)
        self.cancel.setEnabled(True)
        task = DownloadTask(run, self)
        self.task = task
        task.step.connect(lambda index, model_id: self.started(index, len(queue), model_id))
        task.progress.connect(lambda n, total: self.progress.setValue(int(100 * n / total)))
        task.downloaded.connect(self.installed)
        task.checking.connect(self.checking)
        task.failed.connect(self.failed)
        task.result.connect(self.all_installed)
        task.finished.connect(self.finished_task)
        task.start()
        self.owner._buttons()

    def started(self, index: int, total: int, model_id: str) -> None:
        self.current = model_id
        self.progress.setValue(0)
        self.progress.setFormat(f"%p% · {index + 1} of {total}" if total > 1 else "%p%")
        self.status.setText(f"Downloading {model_name(model_id)}…")
        self.owner.status.setText(
            f"Downloading models ({index + 1} of {total}). You can keep working; "
            "Manage models shows progress."
        )

    def checking(self, active: bool) -> None:
        name = model_name(self.current)
        self.status.setText(f"Checking the files for {name}…" if active else f"Downloading {name}…")

    def installed(self, model_id: str) -> None:
        self.model_item(model_id).setCheckState(Qt.CheckState.Unchecked)
        if MANIFEST[model_id].task == "speech":
            selected = str(self.owner.model.currentData() or "")
            if not self.owner.model_installed(selected):
                self.owner.model.setCurrentIndex(self.owner.model.findData(model_id))
            self.owner.adopt_default_speech()
        else:
            self.owner.enable_ai_defaults()
        self.owner._save_settings()
        self.owner.preload_models()  # A newly installed selected model loads right away.
        self.progress.setValue(100)
        self.refresh(keep_status=True)

    def failed(self, message: str) -> None:
        self.status.setText(f"{model_label(self.current)}: {message}")
        self.owner.status.setText(f"Model download did not finish. {message}")

    def all_installed(self, done: list[str]) -> None:
        self.status.setText(
            ("Models" if len(done) > 1 else "Model")
            + " ready. You can close this window and transcribe."
        )
        self.owner.status.setText("Models ready.")
        self.owner.finish_model_setup()

    def finished_task(self) -> None:
        task, self.task = self.task, None
        if task:
            task.deleteLater()
        self.models.setEnabled(True)
        self.cancel.setEnabled(False)
        self.progress.setFormat("%p%")
        self.download.setEnabled(bool(self.checked()))
        self.refresh(keep_status=True)  # Keep the result or error visible.
        self.owner._buttons()

    def cancel_download(self) -> None:
        """Stop by choice: keep the progress, and do not restart setup automatically."""
        if self.task:
            self.task.cancelled.set()
            self.status.setText("Stopping download…")
            self.owner.finish_model_setup()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Closing only hides this window; downloads continue in the background.
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
            # Saved settings carry the model folder and any chosen decoder.
            arguments = ["--data-dir", str(self.owner.store.data_dir)]
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
