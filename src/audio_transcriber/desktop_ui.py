"""Imported lecture library; inference runs in a child CLI process."""

from __future__ import annotations

import json
import sys
import uuid
from bisect import bisect_right
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from PySide6.QtCore import (
    QByteArray,
    QMimeData,
    QPoint,
    QProcess,
    QSignalBlocker,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPalette,
    QShortcut,
    QStandardItemModel,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextListFormat,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyle,
    QSystemTrayIcon,
    QTabBar,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .acquisition import DEFAULT_SPEECH_MODEL, MANIFEST, cpu_only
from .ai_cleanup import DEFAULT_MODEL, CleanupError
from .config import PROCESSING_DEFAULTS
from .desktop_dictation import DictationWindow, set_saved_model
from .desktop_editing import RevisionHistoryDialog, SegmentEditDialog
from .desktop_player import LecturePlayer
from .desktop_setup import (
    AI_DEVICES,
    CLEANUP_LEVELS,
    CPU_ONLY_HINT,
    DEVICE_HINT,
    MODEL_SETUP_VERSION,
    OFFERED_MODELS,
    SPEECH_DEVICES,
    SPEECH_MODELS,
    ModelDialog,
    UpdateDialog,
    add_choices,
    choice,
    model_choice,
    select,
)
from .desktop_trash import TrashDialog
from .desktop_versions import (
    PREPARE,
    background_plan,
    current_info,
    find_cleanup,
    find_derived,
)
from .desktop_worker import ResidentWorker
from .editing import (
    EditRevision,
    current_revision,
    load_revision,
    revision_history,
    save_segment_edit,
)
from .export import ExportError, write_export
from .media import resolve_ffmpeg
from .models import Segment, Session
from .storage import SessionStore, atomic_json
from .trash import TRASH_DAYS, TrashError, empty_trash, trash_lecture
from .versions import (
    UserTextConflict,
    discard_user_text,
    load_user_text,
    save_user_text,
    saved_results,
    use_saved,
)


def getting_started_page() -> Path | None:
    """The bundled help page (installed app) or its source copy (development)."""
    bundle = getattr(sys, "_MEIPASS", None)
    candidates = [Path(bundle) / "help"] if bundle else []
    candidates.append(Path(__file__).resolve().parents[2] / "packaging")
    for folder in candidates:
        page = folder / "GettingStarted.html"
        if page.is_file():
            return page
    return None


# Version tabs in display order (internal layer name -> tab label). Balanced stays
# available to export code and the CLI but is not offered as a tab.
VERSION_TABS = {
    "Raw": "Original",
    "Edited": "Edited",
    "AI": "Transcript",
    "Formatted": "Notes",
    "Summary": "Summary",
}
# Versions with a whole-document user copy (layer -> versions.py name).
USER_COPIES = {"AI": "ai", "Formatted": "formatted", "Summary": "summary"}
BEST_VERSIONS = ("Formatted", "AI", "Edited", "Raw")
FORMAT_NAMES = {"prose": "Mostly prose", "mixed": "Mixed", "structured": "Mostly structured"}
SOURCE_NAMES = {"raw": "the original", "balanced": "the original", "ai": "the AI cleanup"}


MEDIA_SUFFIXES = (".wav", ".mp3", ".m4a", ".mp4")
DEFAULT_TEXT_SIZE = 14


def display_title(title: str) -> str:
    """Lecture names without the audio file's extension."""
    path = Path(title)
    return path.stem if path.suffix.casefold() in MEDIA_SUFFIXES else title


def short_date(created_at: str) -> str:
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError:
        return created_at[:10]
    year = f", {moment.year}" if moment.year != datetime.now().year else ""
    return f"{moment:%b} {moment.day}{year}"


def timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class TranscriptBox(QTextEdit):
    """The transcript text; Ctrl + mouse wheel asks for bigger or smaller text."""

    zoom = Signal(int)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)


class LectureWindow(QMainWindow):
    def __init__(
        self, data_dir: Path, model_root: Path | None = None, ffmpeg: str | None = None
    ) -> None:
        super().__init__()
        self.store = SessionStore(data_dir)
        self.settings_path = self.store.data_dir / "desktop-settings.json"
        settings: dict[str, str] = {}
        settings_error = ""
        if self.settings_path.exists():
            try:
                value = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if not isinstance(value, dict) or not all(
                    isinstance(v, str) for v in value.values()
                ):
                    raise ValueError("settings must contain text values")
                settings = value
            except (OSError, ValueError) as error:
                settings_error = f"Could not read settings: {error}"
        # Saved folders and programs may be gone (another PC, a removed tool);
        # fall back to the library's own models folder and the bundled decoder.
        self.default_model_root = (self.store.data_dir / "models").resolve()
        saved_root = settings.get("model_root", "")
        if saved_root and not Path(saved_root).expanduser().is_dir():
            settings_error = settings_error or (
                f"The saved model folder was not found ({saved_root}); using the default."
            )
            saved_root = ""
        self.model_root = (
            Path(model_root or saved_root or self.default_model_root).expanduser().resolve()
        )
        saved_ffmpeg = settings.get("ffmpeg", "")
        if saved_ffmpeg and not Path(saved_ffmpeg).expanduser().is_file():
            settings_error = settings_error or (
                f"The saved FFmpeg program was not found ({saved_ffmpeg}); using the default."
            )
            saved_ffmpeg = ""
        self.model_setup_done = settings.get("model_setup") == MODEL_SETUP_VERSION
        # Automatic setup switches to a newly offered default speech model once.
        self.default_speech_pending = (
            not self.model_setup_done and settings.get("speech_default") != DEFAULT_SPEECH_MODEL
        )
        # Only a user-chosen decoder is saved; discovery runs again on each launch.
        self.chosen_ffmpeg = ffmpeg or saved_ffmpeg
        self.ffmpeg = resolve_ffmpeg(self.chosen_ffmpeg or None) or ""
        if self.ffmpeg and Path(self.ffmpeg).expanduser().is_file():
            self.ffmpeg = str(Path(self.ffmpeg).expanduser().resolve())
        self.sessions: dict[str, Session] = {}
        self.active_id: str | None = None
        self.before_ids: set[str] = set()
        self.stop_file: Path | None = None
        self.output = ""
        self._job_active = False
        self.dictation_window: DictationWindow | None = None
        self.closing = False
        self.setup_dialog: ModelDialog | None = None
        self.job_kind = "transcription"
        self.cleanup_written = False
        self.ai_stage = ""
        self.job_arguments: list[str] = []
        # AI steps paused this session, by lecture, so Resume can start them again.
        self.paused_jobs: dict[str, list[str]] = {}
        self.shown_id = ""  # The lecture whose version was last chosen for display.
        self.job_reported = False  # A finished job's message is showing in the status line.
        self.guide_dialog: QDialog | None = None
        self.showing_copy = False  # The box shows the user's typed copy of an AI version.
        self.pending_trash = ""  # A lecture to delete once its background step stops.
        self.redo_since: dict[str, datetime] = {}  # Redo all: older versions are remade.
        # Jobs to run after the current one (notes, then summary) and the tab to keep.
        self.followups: list[tuple[str, list[str]]] = []
        self.keep_layer = ""
        self.text_size: int = DEFAULT_TEXT_SIZE
        # Versions prepared in the background: (lecture, worker arguments), oldest first.
        self.background: list[tuple[str, list[str]]] = []
        self.background_enabled = True  # Tests turn this off to keep runs predictable.
        # What to prepare (Settings → Prepare in the background). An "off" saved by the
        # earlier single switch keeps everything off.
        was_off = settings.get("background_versions") == "off"
        self.background_prepare = {
            kind: settings.get(f"background_{kind}", "on" if on and not was_off else "off") == "on"
            for kind, on in PREPARE.items()
        }
        self.bg_job: tuple[str, list[str]] | None = None
        self.bg_stop: Path | None = None
        self.bg_output = ""
        # One process keeps the selected models loaded while the app is open,
        # minimized included, and runs every job; closing the app stops it.
        self.worker = ResidentWorker(self)
        self.worker.status.connect(self._model_status)
        self.preload_timer = QTimer(self)
        self.preload_timer.setSingleShot(True)
        self.preload_timer.setInterval(300)  # One load after several quick changes.
        self.preload_timer.timeout.connect(self._preload_now)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.setWindowTitle("Audio Transcriber — Lecture library")
        self.resize(1100, 760)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 18)
        self.manage_models_button = QPushButton("Manage models…")
        self.manage_models_button.setToolTip("Download or repair the speech and AI models.")
        self.manage_models_button.clicked.connect(self.manage_models)
        controls = QHBoxLayout()
        # Rarely changed choices (models, devices, recording type) live in the Settings
        # menu; these dropdowns stay the source of truth but are not shown in the window.
        self.model = QComboBox()
        self.model.setAccessibleName("Speech model")
        self.model.setObjectName("speechModel")
        self.model.setToolTip("The model that turns speech into text.")
        self.model.addItem("Choose a speech model", "")
        add_choices(
            self.model,
            (model_choice(m) for m in SPEECH_MODELS),
            recommended=DEFAULT_SPEECH_MODEL,
        )
        self.model.setCurrentIndex(max(0, self.model.findData(settings.get("model", ""))))
        self.device = QComboBox()
        self.device.setAccessibleName("Transcription device")
        self.device.setObjectName("transcriptionDevice")
        self.device.setToolTip(DEVICE_HINT)
        add_choices(
            self.device,
            (
                *SPEECH_DEVICES,
                ("auto", "Automatic", "Tests each chip first and uses the fastest."),
            ),
            recommended="CPU",
        )
        select(self.device, settings.get("device", "CPU"))
        self.import_button = QPushButton("Import audio…")
        self.import_button.setToolTip("Transcribe an audio or video file (WAV, MP3, M4A, or MP4).")
        self.import_button.clicked.connect(self.choose_import)
        controls.addWidget(self.import_button)
        self.dictation_button = QPushButton("Live dictation…")
        self.dictation_button.setToolTip("Speak and have the text typed into any app.")
        self.dictation_button.clicked.connect(self.open_dictation)
        controls.addWidget(self.dictation_button)
        # Lecture | Dictation: remembered for new imports; also applies to the selected
        # lecture. Dictation keeps only the correction when the speaker corrects themselves.
        controls.addSpacing(16)
        self.mode_buttons: dict[str, QPushButton] = {}
        mode_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        for value, label, tip in (
            ("lecture", "Lecture", "Keeps everything that was said."),
            (
                "dictation",
                "Dictation",
                'When the speaker corrects themselves ("no, I mean…"), keeps only the correction.',
            ),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip + " Used for new imports and the selected lecture.")
            button.clicked.connect(lambda _=False, v=value: self.set_recording_type(v))
            mode_group.addButton(button)
            mode_row.addWidget(button)
            self.mode_buttons[value] = button
        controls.addLayout(mode_row)
        # Pause shows only while something runs; Resume only for a lecture that stopped.
        self.pause_button = QPushButton("Pause")
        self.pause_button.setToolTip("Stop after the current part. Resume continues from there.")
        self.pause_button.clicked.connect(self.pause)
        controls.addWidget(self.pause_button)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setToolTip("Continue the selected lecture from where it stopped.")
        self.resume_button.clicked.connect(self.resume_selected)
        controls.addWidget(self.resume_button)
        controls.addStretch(1)
        self.background_label = QLabel()
        self.background_label.setToolTip(
            "Other cleanup levels and note styles are prepared while the app is open, so "
            "picking one later is instant. Anything you start yourself runs first."
        )
        self.background_label.hide()
        controls.addWidget(self.background_label)
        controls.addWidget(self.manage_models_button)
        layout.addLayout(controls)
        self.cleanup_model = QComboBox()
        self.cleanup_model.setAccessibleName("Cleanup model")
        self.cleanup_model.setToolTip("The AI model for cleanup, formatting, and summaries.")
        add_choices(
            self.cleanup_model,
            (model_choice(m) for m, spec in MANIFEST.items() if spec.task == "text-cleanup"),
        )
        self.cleanup_model.setCurrentIndex(
            max(0, self.cleanup_model.findData(settings.get("cleanup_model", DEFAULT_MODEL)))
        )
        self.cleanup_mode = QComboBox()
        self.cleanup_mode.setAccessibleName("Cleanup mode")
        self.cleanup_mode.setToolTip("What kind of recording this is.")
        add_choices(
            self.cleanup_mode,
            (
                ("lecture", "Lecture", "Keeps everything that was said."),
                (
                    "dictation",
                    "Dictation",
                    'When you correct yourself ("no, I mean…"), keeps only the correction.',
                ),
            ),
        )
        select(self.cleanup_mode, settings.get("cleanup_mode", "lecture"))
        self.cleanup_mode.currentIndexChanged.connect(self._sync_mode_buttons)
        self._sync_mode_buttons()
        self.cleanup_style = QComboBox()
        self.cleanup_style.setAccessibleName("Cleanup style")
        self.cleanup_style.setToolTip(
            "How much the AI tidies the wording. Runs automatically after each import."
        )
        add_choices(
            self.cleanup_style, CLEANUP_LEVELS, recommended=PROCESSING_DEFAULTS["lecture"][0]
        )
        select(self.cleanup_style, settings.get("cleanup_style", PROCESSING_DEFAULTS["lecture"][0]))
        self.cleanup_device = QComboBox()
        self.cleanup_device.setAccessibleName("Cleanup device")
        self.cleanup_device.setToolTip("Which part of the computer runs the AI.")
        add_choices(self.cleanup_device, AI_DEVICES, recommended="AUTO")
        select(self.cleanup_device, settings.get("cleanup_device", "AUTO"))
        self.format_style = QComboBox()
        self.format_style.setAccessibleName("Formatting style")
        self.format_style.setToolTip(
            "How the Notes version is laid out. Runs automatically after each import."
        )
        add_choices(
            self.format_style,
            (
                ("off", "Off", "No Notes version."),
                ("prose", "Mostly prose", "Plain paragraphs. Quick, and needs no AI."),
                ("mixed", "Mixed", "Headings with short paragraphs, by the AI."),
                (
                    "structured",
                    "Mostly structured",
                    "Headings with bullet points, by the AI.",
                ),
            ),
            recommended=PROCESSING_DEFAULTS["lecture"][1],
        )
        select(self.format_style, settings.get("format_style", PROCESSING_DEFAULTS["lecture"][1]))
        self.cleanup_hint = QLabel(
            "AI cleanup, summaries, and Mixed or Structured formatting need the optional "
            "AI cleanup model. Download it from Manage models…"
        )
        self.cleanup_hint.setWordWrap(True)
        layout.addWidget(self.cleanup_hint)
        self.split = QSplitter()
        split = self.split
        library_panel = QWidget()
        library_layout = QVBoxLayout(library_panel)
        library_layout.setContentsMargins(0, 0, 0, 0)
        self.library_search = QLineEdit()
        self.library_search.setPlaceholderText("Search lectures")
        self.library_search.setAccessibleName("Search lectures")
        self.library_search.setClearButtonEnabled(True)
        self.library_search.textChanged.connect(self._filter_library)
        self.library_search.hide()  # Shown once the library is long enough to need it.
        library_layout.addWidget(self.library_search)
        self.library = QListWidget()
        self.library.setAccessibleName("Lecture library")
        self.library.setObjectName("lectureLibrary")
        self.library.setMinimumWidth(260)
        self.library.currentItemChanged.connect(self.show_selected)
        self.library.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.library.customContextMenuRequested.connect(self._library_menu)
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch(1)
        empty_text = QLabel("No lectures yet.\n\nImport audio, or drag an audio file here.")
        empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_text.setWordWrap(True)
        empty_text.setStyleSheet("font-size: 16px;")
        empty_layout.addWidget(empty_text)
        empty_import = QPushButton("Import audio…")
        empty_import.setAccessibleName("Import your first audio file")  # Unlike the top one.
        empty_import.clicked.connect(self.choose_import)
        empty_layout.addWidget(empty_import, 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addStretch(2)
        self.library_stack = QStackedWidget()
        self.library_stack.addWidget(empty)
        self.library_stack.addWidget(self.library)
        library_layout.addWidget(self.library_stack, 1)
        split.addWidget(library_panel)
        viewer = QWidget()
        right = QVBoxLayout(viewer)
        self.details = QLabel("Import audio or select a lecture from the library.")
        self.details.setWordWrap(True)
        right.addWidget(self.details)
        tools = QHBoxLayout()
        self.layer = QComboBox()
        self.layer.setAccessibleName("Transcript layer")
        self.layer.setObjectName("transcriptLayer")
        # Item data is the layer's internal name; "AI" alone reads like "Al" in Segoe UI.
        add_choices(
            self.layer,
            (
                ("Raw", "Raw", "Exactly what the speech model heard."),
                ("Balanced", "Balanced", "Raw, tidied by simple rules without AI."),
                ("Edited", "Edited", "Your own corrections."),
                ("AI", "AI cleanup", "The full text, tidied by the AI."),
                ("Formatted", "Formatted", "Condensed study notes of the whole lecture."),
                ("Summary", "Summary", "A short overview of the whole lecture."),
            ),
        )
        self.layer.setToolTip("Which version of the transcript to show and export.")
        self.layer.currentIndexChanged.connect(self.show_selected)
        # Tabs show only the versions this lecture has; the hidden dropdown stays in charge.
        self.version_tabs = QTabBar()
        self.version_tabs.setAccessibleName("Transcript version")
        self.version_tabs.setDocumentMode(True)
        self.version_tabs.setExpanding(False)
        self.version_tabs.currentChanged.connect(self._tab_chosen)
        self.search = QLineEdit()
        self.search.setAccessibleName("Find in transcript")
        self.search.setObjectName("transcriptSearch")
        self.search.setPlaceholderText("Find in transcript")
        self.search.setToolTip("Enter or F3 finds the next match; Ctrl+F jumps here.")
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self.find_text)
        tools.addWidget(self.search, 1)
        self.copy_button = QPushButton("Copy all")
        self.copy_button.setToolTip("Copy the version shown, ready to paste into notes or a doc.")
        self.copy_button.clicked.connect(self.copy_all)
        tools.addWidget(self.copy_button)
        self.export_format = QComboBox()
        self.export_format.setAccessibleName("Export format")
        self.export_format.setObjectName("exportFormat")
        self.export_format.setToolTip("File type for Export.")
        add_choices(
            self.export_format,
            (
                ("text", "Text (.txt)", "Plain text that opens anywhere."),
                ("markdown", "Markdown (.md)", "Keeps headings and bullets for note apps."),
                ("srt", "Subtitles (.srt)", "Timed captions for video players."),
                ("json", "Data (.json)", "Full detail for other programs."),
            ),
        )
        # The file types live in the Export menu; this hidden dropdown stays in charge.
        export_menu = QMenu(self)
        export_menu.setToolTipsVisible(True)
        self.export_actions: dict[str, QAction] = {}
        for row in range(self.export_format.count()):
            value = str(self.export_format.itemData(row))
            action = export_menu.addAction(self.export_format.itemText(row) + "…")
            action.setToolTip(
                str(self.export_format.itemData(row, Qt.ItemDataRole.ToolTipRole) or "")
            )
            action.triggered.connect(lambda _=False, v=value: self.export_as(v))
            self.export_actions[value] = action
        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("Save the version shown as a file.")
        self.export_button.setMenu(export_menu)
        tools.addWidget(self.export_button)
        right.addLayout(tools)
        self.player = LecturePlayer()
        self.player.seek_button.clicked.connect(self.seek_selected_segment)
        self.player.seek_button.hide()  # Clicking a line in Original or Edited seeks there.
        self.player.play_button.setToolTip("Play or pause the recording (Ctrl+Space).")
        self.player.player.positionChanged.connect(self._highlight_playing)
        self.segment_starts: list[float] = []
        self.cursor_segment = -1
        self.playing_segment = -1
        right.addWidget(self.player)
        edit_controls = QHBoxLayout()
        # Typing in the text replaces the old edit dialog; the method stays for the
        # one-segment flow (uncertainty flag) and its tests.
        self.edit_button = QPushButton("Edit selected segment…")
        self.edit_button.clicked.connect(self.edit_selected_segment)
        self.history_button = QPushButton("Revision history…")
        self.history_button.setToolTip("See your earlier saved edits.")
        self.history_button.clicked.connect(self.show_history)
        edit_controls.addWidget(self.history_button)
        # Cleanup and formatting run automatically after import; these rerun them on
        # the selected lecture with the choice picked here.
        # A level or style already made for this lecture is shown at once; Redo runs it
        # again. The checkmark marks the version on screen.
        cleanup_menu = QMenu("Clean up", self)
        cleanup_menu.setToolTipsVisible(True)
        self.cleanup_action = cleanup_menu.menuAction()
        self.cleanup_choices: dict[str, QAction] = {}
        for value, label, explanation in CLEANUP_LEVELS[1:]:
            action = cleanup_menu.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, v=value: self.pick_cleanup(v))
            action.setToolTip(explanation)
            self.cleanup_choices[value] = action
        cleanup_menu.addSeparator()
        # One Redo for everything, at the bottom of both menus.
        self.redo_actions = [cleanup_menu.addAction("Redo all", self.redo_all)]
        cleanup_menu.aboutToShow.connect(self._mark_choices)
        format_menu = QMenu("Format", self)
        format_menu.setToolTipsVisible(True)
        self.format_action = format_menu.menuAction()
        self.format_choices: dict[str, QAction] = {}
        for row in range(1, self.format_style.count()):
            value = str(self.format_style.itemData(row))
            action = format_menu.addAction(
                self.format_style.itemText(row).removesuffix(" (recommended)")
            )
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, v=value: self.pick_format(v))
            action.setToolTip(
                str(self.format_style.itemData(row, Qt.ItemDataRole.ToolTipRole) or "")
            )
            self.format_choices[value] = action
        # The style applies to both Notes and Summary, so there is no Summary item; this
        # action only tracks whether a summary can be written.
        self.summary_action = QAction("Summary", self)
        format_menu.addSeparator()
        self.redo_actions.append(format_menu.addAction("Redo all", self.redo_all))
        for redo in self.redo_actions:
            redo.setToolTip(
                "Make this lecture's cleanup, notes, and summary again; other levels "
                "and styles are then remade in the background."
            )
        format_menu.aboutToShow.connect(self._mark_choices)
        self.cleanup_button = QPushButton("Clean up")
        self.cleanup_button.setToolTip("Tidy this lecture's wording again at the level you pick.")
        self.cleanup_button.setMenu(cleanup_menu)
        edit_controls.addWidget(self.cleanup_button)
        self.format_button = QPushButton("Format")
        self.format_button.setToolTip("Lay out this lecture's notes and summary in a style.")
        self.format_button.setMenu(format_menu)
        edit_controls.addWidget(self.format_button)
        self.revert_button = QPushButton("Undo my edits")
        self.revert_button.setToolTip("Show the AI's version again. Your edits stay in history.")
        self.revert_button.clicked.connect(self.revert_edits)
        self.revert_button.hide()
        edit_controls.addWidget(self.revert_button)
        edit_controls.addStretch(1)
        right.addLayout(edit_controls)
        self.transcript = TranscriptBox()
        self.transcript.setAccessibleName("Transcript")
        self.transcript.setObjectName("transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("The transcript appears here once it is ready.")
        # Typing is saved shortly after it stops, and before switching away.
        self.rendering = False
        self.dirty = False  # Typing not yet saved.
        self.rendered: tuple[str, str] | None = None  # (lecture, version) in the box
        self.rendered_key: tuple[object, ...] | None = None
        self.rendered_user_rev: str | None = None
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(1500)
        self.save_timer.timeout.connect(self._save_edits)
        self.transcript.textChanged.connect(self._text_edited)
        self.transcript.cursorPositionChanged.connect(self._cursor_moved)
        self.transcript.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript.customContextMenuRequested.connect(self._transcript_menu)
        self.transcript.zoom.connect(lambda step: self.set_text_size(self.text_size + step))
        # Version tabs sit on the top edge of the text they switch.
        page = QVBoxLayout()
        page.setSpacing(0)
        tabs_row = QHBoxLayout()
        tabs_row.addWidget(self.version_tabs)
        tabs_row.addStretch(1)
        self.saved_label = QLabel("")
        self.saved_label.setStyleSheet("color: palette(placeholder-text);")
        tabs_row.addWidget(self.saved_label)
        page.addLayout(tabs_row)
        page.addWidget(self.transcript, 1)
        right.addLayout(page, 1)
        split.addWidget(viewer)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.hide()  # Shown only while a job runs.
        layout.addWidget(self.progress)
        self.status = QLabel(settings_error or "Ready.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.updates = UpdateDialog(self)
        file_menu = self.menuBar().addMenu("File")
        file_menu.setToolTipsVisible(True)
        library_action = file_menu.addAction("Open library folder", self.open_library_folder)
        library_action.setToolTip("Where your recordings and transcripts are stored.")
        self.show_recording_action = file_menu.addAction(
            "Show recording in folder", self.show_recording
        )
        self.show_recording_action.setToolTip("Open the folder with the selected lecture's audio.")
        trash = file_menu.addAction("Trash…", self.show_trash)
        trash.setToolTip(f"Restore deleted lectures within {TRASH_DAYS} days.")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.setToolTipsVisible(True)
        self.choice_menus = [
            self._choice_menu(settings_menu, "Cleanup for new imports", self.cleanup_style),
            self._choice_menu(settings_menu, "Formatting for new imports", self.format_style),
        ]
        prepare_menu = settings_menu.addMenu("Prepare in the background")
        prepare_menu.setToolTipsVisible(True)
        self.background_actions: dict[str, QAction] = {}
        for kind, label, tip in (
            (
                "notes",
                "Other note styles",
                "After cleanup, also make the cleanup's notes in each style (Mostly prose, "
                "Mixed, Mostly structured) so switching style is instant.",
            ),
            (
                "summaries",
                "Other summary styles",
                "After cleanup, also make the cleanup's summary in each style.",
            ),
            (
                "levels",
                "Other cleanup levels",
                "Also clean up at the other levels (Light, Medium, Heavy), with the note and "
                "summary styles chosen above. Keeps the AI model busy much longer.",
            ),
        ):
            action = prepare_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.background_prepare[kind])
            action.setToolTip(tip + " Runs on battery too.")
            action.toggled.connect(lambda on, kind=kind: self.set_background(kind, on))
            self.background_actions[kind] = action
        settings_menu.addSeparator()
        self.choice_menus += [
            self._choice_menu(settings_menu, "Speech model", self.model),
            self._choice_menu(settings_menu, "Speech device", self.device),
        ]
        settings_menu.addSeparator()
        self.choice_menus += [
            self._choice_menu(settings_menu, "AI model", self.cleanup_model),
            self._choice_menu(settings_menu, "AI device", self.cleanup_device),
        ]
        settings_menu.addSeparator()
        self.defaults_action = settings_menu.addAction(
            "Restore recommended settings", self.restore_recommended
        )
        self.defaults_action.setToolTip("Put every setting back to its recommended choice.")
        settings_menu.addSeparator()
        self.manage_models_action = settings_menu.addAction("Manage models…", self.manage_models)
        self.manage_models_action.setToolTip(self.manage_models_button.toolTip())
        self.models_action = settings_menu.addAction("Model folder…", self.choose_models)
        self.ffmpeg_action = settings_menu.addAction("FFmpeg program…", self.choose_ffmpeg)
        help_menu = self.menuBar().addMenu("Help")
        guide = getting_started_page()
        if guide is not None:
            help_menu.addAction("Getting started", lambda: self.show_guide(guide))
        help_menu.addAction("Check for updates…", self.show_updates)
        self.model.currentIndexChanged.connect(self._buttons)  # The device may not apply.
        self.cleanup_style.currentIndexChanged.connect(self._buttons)
        self.format_style.currentIndexChanged.connect(self._buttons)
        self.export_format.currentIndexChanged.connect(self._buttons)
        for selector in (
            self.model,
            self.device,
            self.cleanup_model,
            self.cleanup_device,
            self.cleanup_style,
            self.format_style,
        ):
            selector.currentIndexChanged.connect(self.preload_models)
        self.setCentralWidget(central)
        self.setStyleSheet(
            # Combo-box padding would clip item text: Qt sizes boxes without it.
            "QPushButton { padding: 7px 12px; } QLineEdit { padding: 5px; }"
        )
        self._folder_labels()
        for keys, slot in (
            ("Ctrl+F", self._focus_search),
            ("F3", self.find_text),
            ("Ctrl+Space", self.player.toggle),
            ("Ctrl+=", lambda: self.set_text_size(self.text_size + 1)),
            ("Ctrl++", lambda: self.set_text_size(self.text_size + 1)),
            ("Ctrl+-", lambda: self.set_text_size(self.text_size - 1)),
            ("Ctrl+0", lambda: self.set_text_size(DEFAULT_TEXT_SIZE)),
        ):
            QShortcut(QKeySequence(keys), self, slot)
        delete_key = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.library)
        delete_key.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_key.activated.connect(self.delete_lecture)
        self.setAcceptDrops(True)
        self.import_queue: list[Path] = []
        self.tray: QSystemTrayIcon | None = None
        # Size, layout, text size, and the last lecture and version come back next time.
        self.text_size = DEFAULT_TEXT_SIZE
        try:
            self.text_size = int(settings.get("text_size", DEFAULT_TEXT_SIZE))
            if settings.get("window_geometry"):
                self.restoreGeometry(QByteArray.fromHex(settings["window_geometry"].encode()))
            if settings.get("library_split"):
                self.split.restoreState(QByteArray.fromHex(settings["library_split"].encode()))
        except ValueError:
            pass  # A damaged layout setting just means the default layout.
        self.set_text_size(self.text_size)
        self.restore_lecture = settings.get("last_lecture", "")
        self.restore_version = settings.get("last_version", "")
        try:
            expired = empty_trash(self.store)
        except OSError, TrashError:
            expired = 0  # Tried again at the next start.
        self.refresh()
        if expired:
            self.status.setText(
                f"Removed {expired} lecture(s) that were in the Trash for over {TRASH_DAYS} days."
            )

    @property
    def processing(self) -> bool:
        """A transcription, AI job, or dictation is running; model downloads are separate."""
        return self._job_active or bool(self.dictation_window and self.dictation_window.busy)

    @property
    def busy(self) -> bool:
        return self.processing or bool(self.setup_dialog and self.setup_dialog.task)

    def model_installed(self, model_id: str) -> bool:
        spec = MANIFEST.get(model_id)
        return spec is not None and all(
            (self.model_root / spec.id / entry.name).is_file() for entry in spec.files
        )

    @property
    def cleanup_available(self) -> bool:
        return self.model_installed(str(self.cleanup_model.currentData()))

    def manage_models(self) -> None:
        if self.setup_dialog is None:
            self.setup_dialog = ModelDialog(self)
        self.setup_dialog.show()
        self.setup_dialog.raise_()

    def enable_ai_defaults(self) -> None:
        """After the cleanup model downloads, turn on the default cleanup and layout."""
        if not self.cleanup_available:
            return
        cleanup, layout = PROCESSING_DEFAULTS["lecture"]
        if choice(self.cleanup_style) == "off":
            select(self.cleanup_style, cleanup)
        if self.format_style.currentData() == "off":
            self.format_style.setCurrentIndex(self.format_style.findData(layout))
        dictation = self.dictation_window
        if dictation is not None and choice(dictation.cleanup) == "off":
            select(dictation.cleanup, PROCESSING_DEFAULTS["dictation"][0])
        self._buttons()

    def show_updates(self) -> None:
        self.updates.show()
        self.updates.check_now(True)

    def first_launch(self) -> None:
        self.adopt_default_speech()  # Already downloaded, for example by an earlier setup.
        installed = [m for m in SPEECH_MODELS if self.model_installed(m)]
        if installed and not self.model_installed(str(self.model.currentData() or "")):
            self.model.setCurrentIndex(self.model.findData(installed[0]))
        missing = [m for m in OFFERED_MODELS if not self.model_installed(m)]
        if missing and not self.model_setup_done:
            # Setup downloads everything until it finishes or the user stops it;
            # an interrupted setup continues on the next launch.
            self.manage_models()
            if self.setup_dialog is not None:
                self.setup_dialog.start()
        elif not installed and not self.sessions:
            self.manage_models()
        self.preload_models()
        self.updates.auto_check()

    def preload_models(self, *_: object) -> None:
        """Load the selected models in the background (after a short pause for more changes)."""
        if not self.closing:
            self.job_reported = False  # A new choice: its load result may be shown.
            self.preload_timer.start()

    def _preload_now(self) -> None:
        if self.closing:
            return
        selection: list[dict[str, str]] = []

        def add(kind: str, model: object, device: str, data_dir: Path) -> None:
            item = {
                "kind": kind,
                "model": str(model or ""),
                "device": device,
                "model_root": str(self.model_root),
                "data_dir": str(data_dir),
            }
            if self.model_installed(item["model"]) and item not in selection:
                selection.append(item)

        add("speech", self.model.currentData(), choice(self.device), self.store.data_dir)
        # The AI model is large; load it only when cleanup or AI formatting will use it.
        if choice(self.cleanup_style) != "off" or self.format_style.currentData() in (
            "mixed",
            "structured",
        ):
            add(
                "cleanup",
                self.cleanup_model.currentData(),
                choice(self.cleanup_device),
                self.store.data_dir,
            )
        dictation = self.dictation_window
        if dictation is not None and dictation.isVisible():
            add(
                "speech",
                dictation.model.currentData(),
                choice(dictation.device),
                dictation.store.data_dir,
            )
            if choice(dictation.cleanup) != "off":
                add(
                    "cleanup",
                    dictation.cleanup_model.currentData(),
                    choice(dictation.cleanup_device),
                    dictation.store.data_dir,
                )
        self.worker.load(selection)

    def _model_status(self, message: str) -> None:
        # A finished job's result stays on screen; a late background load message
        # (often about the same problem) must not replace it.
        if not self.processing and not self.job_reported:
            # "Loaded whisper-base… on CPU" is detail; failures keep their full text.
            self.status.setText("Ready." if message.startswith("Loaded ") else message)

    def adopt_default_speech(self) -> None:
        """During automatic setup, once: use the default speech model for lectures and dictation.

        Later choices are the user's; setup finishing (or being stopped) ends this.
        """
        if not self.default_speech_pending or not self.model_installed(DEFAULT_SPEECH_MODEL):
            return
        self.default_speech_pending = False
        select(self.model, DEFAULT_SPEECH_MODEL)
        dictation = self.dictation_window
        try:
            if dictation is not None:
                select(dictation.model, DEFAULT_SPEECH_MODEL)
                dictation._save_settings()
            else:
                set_saved_model(self.store.data_dir / "dictation", DEFAULT_SPEECH_MODEL)
            self._save_settings()
        except OSError as error:
            self.status.setText(f"Could not save settings: {error}")

    def finish_model_setup(self) -> None:
        self.model_setup_done = True
        self.default_speech_pending = False
        try:
            self._save_settings()
        except OSError as error:
            self.status.setText(f"Could not save settings: {error}")

    def open_dictation(self) -> None:
        if self.dictation_window is None:
            self.dictation_window = DictationWindow(self)
            self.dictation_window.changed.connect(self._dictation_changed)
        self.dictation_window.show()
        self.dictation_window.raise_()
        self.dictation_window.activateWindow()

    def _dictation_changed(self) -> None:
        self._buttons()
        QTimer.singleShot(0, self._run_background_next)  # Resume after dictation.
        if self.closing and not self.busy:
            QTimer.singleShot(0, self.close)

    def _folder_labels(self) -> None:
        self.models_action.setToolTip(
            f"Use models stored in another folder. Now: {self.model_root}"
        )
        decoder = self.ffmpeg or "not found"
        self.ffmpeg_action.setToolTip(
            f"The program that reads audio files. Not normally changed. Now: {decoder}"
        )

    def _choice_menu(self, parent: QMenu, title: str, combo: QComboBox) -> QMenu:
        """A Settings submenu for a dropdown kept off the window; the dropdown stays in charge."""
        menu = parent.addMenu(title)
        menu.setToolTipsVisible(True)
        group = QActionGroup(menu)
        for row in range(combo.count()):
            value = combo.itemData(row)
            if not value:
                continue  # The "Choose a speech model" placeholder is not a choice.
            action = menu.addAction(combo.itemText(row))
            action.setCheckable(True)
            action.setData(value)
            action.setToolTip(str(combo.itemData(row, Qt.ItemDataRole.ToolTipRole) or ""))
            action.triggered.connect(lambda _=False, v=value: select(combo, v))
            group.addAction(action)

        def show_current(*_: object) -> None:
            menu.menuAction().setToolTip(combo.toolTip())  # It can change, as for devices.
            label = combo.currentText().removesuffix(" (recommended)")
            menu.setTitle(f"{title}: {label}" if combo.currentData() else title)
            items = cast(QStandardItemModel, combo.model())
            for action in group.actions():
                action.setChecked(action.data() == combo.currentData())
                item = items.item(combo.findData(action.data()))
                action.setEnabled(combo.isEnabled() and item is not None and item.isEnabled())

        combo.currentIndexChanged.connect(show_current)
        menu.aboutToShow.connect(show_current)  # Some choices change without a signal.
        show_current()
        return menu

    def restore_recommended(self) -> None:
        cleanup, layout = PROCESSING_DEFAULTS["lecture"]
        for combo, value in (
            (self.device, "CPU"),
            (self.cleanup_model, DEFAULT_MODEL),
            (self.cleanup_device, "AUTO"),
            (self.cleanup_mode, "lecture"),
            (self.cleanup_style, cleanup),
            (self.format_style, layout),
        ):
            select(combo, value)
        if self.model_installed(DEFAULT_SPEECH_MODEL):
            select(self.model, DEFAULT_SPEECH_MODEL)
        for kind, on in PREPARE.items():
            self.background_actions[kind].setChecked(on)  # Saves and replans if changed.
        self._buttons()  # Turns AI choices back off if the AI model is not downloaded.
        self.status.setText("Recommended settings restored.")

    # -- library ---------------------------------------------------------------------

    def _filter_library(self, text: str = "") -> None:
        needle = text.casefold().strip()
        for row in range(self.library.count()):
            item = self.library.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def _lecture_busy(self, session_id: str) -> bool:
        return self._job_active and self.active_id == session_id

    def _library_menu(self, position: QPoint) -> None:
        item = self.library.itemAt(position)
        if item is None:
            return
        self.library.setCurrentItem(item)
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        rename = menu.addAction("Rename…", self.rename_lecture)
        rename.setEnabled(not self._lecture_busy(session_id))
        menu.addAction("Show recording in folder", self.show_recording)
        export = menu.addMenu("Export")
        for value, action in self.export_actions.items():
            entry = export.addAction(action.text())
            entry.setEnabled(action.isEnabled())
            entry.triggered.connect(lambda _=False, v=value: self.export_as(v))
        menu.addSeparator()
        delete = menu.addAction("Delete…", self.delete_lecture)
        delete.setEnabled(not self._lecture_busy(session_id))
        menu.exec(self.library.viewport().mapToGlobal(position))

    def delete_lecture(self) -> None:
        """Move the selected lecture to the Trash, after asking."""
        session = self.sessions.get(self.selected_id() or "")
        if session is None:
            return
        name = display_title(session.title)
        if self._lecture_busy(session.id):
            self.status.setText(f"Pause or wait for {name} to finish before deleting it.")
            return
        answer = QMessageBox.question(
            self,
            "Delete lecture",
            f"Move “{name}” to the Trash?\n\n"
            f"You can restore it from File → Trash for {TRASH_DAYS} days.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._save_edits()
        self.background = [job for job in self.background if job[0] != session.id]
        self._update_background_label()
        if self.bg_job is not None and self.bg_job[0] == session.id:
            # A background step is writing into this lecture; delete once it stops.
            self.pending_trash = session.id
            self.pause_background()
            self.status.setText(f"Deleting {name} once the current background step stops…")
            return
        self._trash_now(session.id)

    def _trash_now(self, session_id: str) -> None:
        self.pending_trash = ""
        session = self.sessions.get(session_id)
        name = display_title(session.title) if session else "the lecture"
        if self.selected_id() == session_id:
            self.player.load(None)  # The player keeps the audio file open.
        try:
            trash_lecture(self.store, session_id)
        except (OSError, TrashError) as error:
            self.status.setText(f"Could not delete {name}: {error}")
            self.show_selected()
            return
        self.sessions.pop(session_id, None)
        self.paused_jobs.pop(session_id, None)
        if self.active_id == session_id:
            self.active_id = None
        if self.shown_id == session_id:
            self.shown_id = ""
            self.rendered = None
            self.rendered_key = None
        self.refresh()
        self.status.setText(
            f"Moved {name} to the Trash. Restore it from File → Trash within {TRASH_DAYS} days."
        )

    def show_trash(self) -> None:
        TrashDialog(self.store, self._trash_changed, self).exec()

    def _trash_changed(self, message: str) -> None:
        self.refresh()
        self.status.setText(message)

    def rename_lecture(self) -> None:
        session = self.sessions.get(self.selected_id() or "")
        if session is None or self._lecture_busy(session.id):
            return
        name, accepted = QInputDialog.getText(
            self, "Rename lecture", "Name:", text=display_title(session.title)
        )
        name = name.strip()
        if not accepted or not name or name == display_title(session.title):
            return
        try:
            fresh = self.store.load_session(session.id)
            fresh.title = name
            self.store.save_session(fresh)
        except (OSError, ValueError, KeyError) as error:
            self.status.setText(f"Could not rename the lecture: {error}")
            return
        self.refresh()
        self.status.setText(f"Renamed to {name}.")

    def queue_imports(self, files: list[Path]) -> None:
        """Import files one after another; each starts when the previous one finishes."""
        self.import_queue += files
        if self.processing and self.import_queue:
            self.status.setText(
                f"{len(self.import_queue)} file(s) will import after the current job."
            )
        self._next_import()

    def _next_import(self) -> None:
        if self.processing or self.closing or not self.import_queue:
            return
        self.start_import(self.import_queue.pop(0))
        if self.import_queue and self._job_active:
            self.status.setText(
                f"{self.status.text()} {len(self.import_queue)} more waiting to import."
            )

    def _dropped_files(self, data: QMimeData) -> list[Path]:
        return [
            Path(url.toLocalFile())
            for url in data.urls()
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.casefold() in MEDIA_SUFFIXES
        ]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_files(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = self._dropped_files(event.mimeData())
        if files:
            event.acceptProposedAction()
            self.queue_imports(files)

    def _notify_finished(self) -> None:
        """Tell the user a job finished when the window is minimized or in the background."""
        if not self.isVisible() or (self.isActiveWindow() and not self.isMinimized()):
            return
        QApplication.alert(self)  # Flashes the taskbar button.
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        if self.tray is None:
            icon = self.windowIcon()
            if icon.isNull():
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
            self.tray = QSystemTrayIcon(icon, self)
            self.tray.messageClicked.connect(self._bring_to_front)
        self.tray.show()
        self.tray.showMessage(
            "Audio Transcriber", self.status.text() or "Finished.", self.tray.icon(), 8000
        )
        QTimer.singleShot(20_000, self.tray.hide)

    def _bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # -- transcript box -------------------------------------------------------------

    def _focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def set_text_size(self, size: int) -> None:
        self.text_size = max(10, min(32, size))
        self.transcript.setStyleSheet(f"QTextEdit {{ font-size: {self.text_size}px; }}")

    def copy_all(self) -> None:
        """Copy the version shown; documents get the formatting, plain editors the text.
        Qt's own copy builds the clipboard data (a Python-made QMimeData handed to the
        clipboard crashed the app at exit)."""
        cursor = self.transcript.textCursor()
        scroll = self.transcript.verticalScrollBar().value()
        self.rendering = True  # Selecting everything must not move the audio.
        try:
            self.transcript.selectAll()
            self.transcript.copy()
            self.transcript.setTextCursor(cursor)
        finally:
            self.rendering = False
        self.transcript.verticalScrollBar().setValue(scroll)
        self.status.setText("Copied. Paste it into your notes or a document.")

    def export_as(self, fmt: str) -> None:
        select(self.export_format, fmt)
        self.choose_export()

    def _cursor_moved(self) -> None:
        """Clicking (or moving to) a line in Original or Edited moves the audio there."""
        if self.rendering or not self.segment_starts:
            return
        index = self.transcript.textCursor().blockNumber() // 2
        if index != self.cursor_segment and 0 <= index < len(self.segment_starts):
            self.cursor_segment = index
            self.player.seek(self.segment_starts[index])

    def _highlight_playing(self, position: int) -> None:
        """Shade the line the audio is on."""
        index = bisect_right(self.segment_starts, position / 1000) - 1
        if not self.segment_starts:
            index = -1
        if index == self.playing_segment:
            return
        self.playing_segment = index
        if index < 0:
            self.transcript.setExtraSelections([])
            return
        shade = QTextCharFormat()
        colour = self.palette().color(QPalette.ColorRole.Highlight)
        colour.setAlpha(70)
        shade.setBackground(colour)
        shade.setProperty(QTextFormat.Property.FullWidthSelection, True)
        line = QTextEdit.ExtraSelection()
        line.cursor = QTextCursor(self.transcript.document().findBlockByNumber(2 * index))
        line.format = shade
        self.transcript.setExtraSelections([line])

    def _transcript_menu(self, position: QPoint) -> None:
        menu = self.transcript.createStandardContextMenu(position)
        session = self.sessions.get(self.selected_id() or "")
        if self.segment_starts and self._editable(session, self._layer()):
            index = self.transcript.cursorForPosition(position).blockNumber() // 2
            try:
                segments = self.store.load_transcript(
                    session.id if session else "", self._layer().lower()
                ).segments
                unclear = segments[index].uncertain
            except OSError, ValueError, KeyError, IndexError:
                unclear = None
            if unclear is not None:
                menu.addSeparator()
                toggle = menu.addAction("Mark line as clear" if unclear else "Mark line as unclear")
                toggle.triggered.connect(lambda: self.toggle_unclear(index))
        menu.exec(self.transcript.viewport().mapToGlobal(position))

    def toggle_unclear(self, index: int) -> None:
        """Flip a line's "[unclear]" marker; saved as an Edited version like typing."""
        self._save_edits()
        session_id = self.selected_id()
        if session_id is None:
            return
        layer = self._layer()
        try:
            segment = self.store.load_transcript(session_id, layer.lower()).segments[index]
            expected = current_revision(self.store, session_id)
            base = load_revision(self.store, session_id).base_layer if expected else layer.lower()
            save_segment_edit(
                self.store,
                session_id,
                index,
                segment.text,
                not segment.uncertain,
                base_layer=base,
                expected_revision=expected,
            )
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.status.setText(f"Could not change that line: {error}")
            return
        self.sessions[session_id] = self.store.load_session(session_id)
        self._show_layer("Edited")
        self.rendered_key = None
        self.show_selected()
        self.saved_label.setText("Saved")

    def show_guide(self, page: Path) -> None:
        """The bundled Getting started page, in an app window rather than a browser."""
        if self.guide_dialog is None:
            self.guide_dialog = QDialog(self)
            self.guide_dialog.setWindowTitle("Getting started")
            self.guide_dialog.resize(760, 680)
            layout = QVBoxLayout(self.guide_dialog)
            viewer = QTextBrowser()
            viewer.setOpenExternalLinks(True)  # The user-guide link opens on the web.
            viewer.setSource(QUrl.fromLocalFile(str(page)))
            layout.addWidget(viewer)
            close = QPushButton("Close")
            close.clicked.connect(self.guide_dialog.close)
            layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        self.guide_dialog.show()
        self.guide_dialog.raise_()

    def open_library_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.data_dir)))

    def show_recording(self) -> None:
        session_id = self.selected_id()
        if session_id is None:
            return
        try:
            recording = self.store.source_copy(session_id)
        except (OSError, ValueError, KeyError) as error:
            self.status.setText(f"Could not find this lecture's recording: {error}")
            return
        if sys.platform == "win32":
            QProcess.startDetached("explorer.exe", ["/select,", str(recording)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(recording.parent)))

    def _save_settings(self) -> None:
        locations = {}
        if self.model_root != self.default_model_root:
            locations["model_root"] = str(self.model_root)
        if self.chosen_ffmpeg:
            locations["ffmpeg"] = self.chosen_ffmpeg
        if self.model_setup_done:
            locations["model_setup"] = MODEL_SETUP_VERSION
        if not self.default_speech_pending:
            locations["speech_default"] = DEFAULT_SPEECH_MODEL
        atomic_json(
            self.settings_path,
            {
                **locations,
                "cleanup_model": str(self.cleanup_model.currentData()),
                "cleanup_mode": str(self.cleanup_mode.currentData()),
                "cleanup_style": choice(self.cleanup_style),
                "cleanup_device": choice(self.cleanup_device),
                "format_style": str(self.format_style.currentData()),
                "model": self.model.currentData(),
                "device": choice(self.device),
                "text_size": str(self.text_size),
                **{
                    f"background_{kind}": "on" if on else "off"
                    for kind, on in self.background_prepare.items()
                },
                "window_geometry": bytes(self.saveGeometry().toHex().data()).decode(),
                "library_split": bytes(self.split.saveState().toHex().data()).decode(),
                "last_lecture": self.selected_id() or "",
                "last_version": self._layer(),
            },
        )

    def choose_models(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder containing approved models", str(self.model_root)
        )
        if folder:
            self.model_root = Path(folder).resolve()
            self._folder_labels()
            self._buttons()
            self.preload_models()

    def choose_ffmpeg(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "Choose FFmpeg executable", self.ffmpeg, "Executable (*.exe);;All files (*)"
        )
        if file:
            self.chosen_ffmpeg = self.ffmpeg = file
            self._folder_labels()

    def choose_import(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import audio",
            "",
            "Media (*.wav *.mp3 *.m4a *.mp4);;All files (*)",
        )
        self.queue_imports([Path(file) for file in files])

    def start_import(self, source: Path) -> None:
        if self.processing:
            return
        model_id = self.model.currentData()
        if not model_id:
            self.status.setText(
                "No speech model is ready. Download one from Manage models, "
                "or pick one under Settings → Speech model."
            )
            return
        self.active_id = None
        self._start(["transcribe", str(source.expanduser().resolve()), "--model", model_id])

    def selected_id(self) -> str | None:
        item = self.library.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def status_label(self, session: Session) -> str:
        running = self._job_active and self.active_id == session.id
        if session.status == "processing":
            if not running:
                return "Stopped"
            percent = getattr(self, "job_percent", None)
            return "Transcribing…" if percent is None else f"Transcribing… {percent}%"
        if session.status == "ready" and running and self.job_kind != "transcription":
            return f"Ready · {self.ai_stage or 'AI step'}…"
        if session.status == "ready" and session.id in self.paused_jobs:
            return "Ready · AI step paused"
        return {"ready": "Ready", "interrupted": "Paused", "failed": "Stopped by an error"}.get(
            session.status, session.status.capitalize()
        )

    def resumable(self, session: Session | None) -> bool:
        """A lecture whose transcription or AI step stopped partway and can continue."""
        if session is None or self.processing:
            return False
        # "processing" with no job running means the app closed mid-transcription.
        return session.status in ("interrupted", "failed", "processing") or (
            session.id in self.paused_jobs
        )

    def resume_selected(self) -> None:
        session_id = self.selected_id()
        session = self.sessions.get(session_id or "")
        if session is None or not self.resumable(session):
            return
        if session.status == "ready":
            # A paused AI step starts that step again; earlier versions are kept.
            self.active_id = session.id
            self._start(self.paused_jobs.pop(session.id))
            return
        diag = session.diagnostics
        self.active_id = session.id
        self._start(
            [
                "resume",
                session.id,
                "--model",
                str(diag.get("model_id", "")),
                "--chunk-seconds",
                str(diag.get("chunk_seconds", 30)),
                "--overlap-seconds",
                str(diag.get("overlap_seconds", 0)),
                "--chunk-strategy",
                str(diag.get("chunk_strategy", "fixed")),
            ]
        )

    def start_cleanup(self, style: str = "") -> None:
        """Rerun cleanup on the selected lecture at `style` (default: the import setting)."""
        session_id = self.selected_id()
        if self.processing or not session_id or not self.cleanup_action.isEnabled():
            return
        self._run_cleanup(session_id, style)

    def _run_cleanup(self, session_id: str, style: str = "") -> None:
        if not style or style == "off":
            style = choice(self.cleanup_style)
        if style == "off":
            style = PROCESSING_DEFAULTS["lecture"][0]
        self.active_id = session_id
        self._start(
            [
                "cleanup",
                session_id,
                "--model",
                str(self.cleanup_model.currentData()),
                "--mode",
                str(self.cleanup_mode.currentData()),
                "--style",
                style,
                "--formatting",
                str(
                    current_info(self.store, session_id, "formatted").get("style")
                    or self.format_style.currentData()
                ),
                # Lectures always get a summary, whatever the recording type (the CLI
                # leaves it off for dictation-mode cleanups).
                "--summary",
                "on",
            ]
        )

    def start_formatting(self, style: str = "") -> None:
        """Re-lay out the version shown in `style` (default: the import setting)."""
        session_id = self.selected_id()
        if self.processing or not session_id or not self.format_action.isEnabled():
            return
        if style and not self.format_choices[style].isEnabled():
            return
        source = self._layer().lower()
        if source not in ("raw", "balanced", "ai"):
            source = (
                "ai"
                if (self.store.session_dir(session_id) / "ai-transcript.json").is_file()
                else "raw"
            )
        self._run_formatting(session_id, source, style)

    def _run_formatting(self, session_id: str, source: str, style: str = "") -> None:
        if not style or style == "off":
            style = str(self.format_style.currentData())
        if style == "off":
            style = "prose"
        self.active_id = session_id
        self._start(
            [
                "format",
                session_id,
                "--source",
                source,
                "--style",
                style,
                "--model",
                str(self.cleanup_model.currentData()),
            ]
        )

    def _ready_session(self) -> str | None:
        session = self.sessions.get(self.selected_id() or "")
        if session is None or session.status != "ready" or self.processing:
            return None
        return session.id

    def _mark_choices(self) -> None:
        """Check the level and style on screen."""
        session_id = self.selected_id() or ""
        level = current_info(self.store, session_id, "ai").get("style") if session_id else None
        style = (
            current_info(self.store, session_id, "formatted").get("style") if session_id else None
        )
        for value, action in self.cleanup_choices.items():
            action.setChecked(value == level)
        for value, action in self.format_choices.items():
            action.setChecked(value == style)
        for redo in self.redo_actions:
            redo.setEnabled(bool(level) and self.cleanup_action.isEnabled())

    def _use_saved_style(self, session_id: str, ai_revision: str | None, style: str) -> list[str]:
        """Make the saved notes and summary in `style` (made from that cleanup) current;
        returns the layers still missing ("formatted", "summary")."""
        missing = []
        for layer in ("formatted", "summary"):
            saved = find_derived(saved_results(self.store, session_id, layer), ai_revision, style)
            if saved is None:
                missing.append(layer)
            elif not saved.is_current:
                use_saved(self.store, session_id, layer, saved.revision_id)
        if "summary" in missing and not self.cleanup_available:
            missing.remove("summary")  # Summaries need the AI model.
        return missing

    def _make_missing(self, session_id: str, missing: list[str], style: str) -> None:
        """Make the notes and then the summary that are not saved yet, keeping the tab."""
        ai = bool(current_info(self.store, session_id, "ai"))
        model = str(self.cleanup_model.currentData())
        jobs = []
        if "formatted" in missing:
            source = "ai" if ai else "raw"
            jobs.append(
                ["format", session_id, "--source", source, "--style", style, "--model", model]
            )
        if "summary" in missing:
            jobs.append(["summarize", session_id, "--style", style, "--model", model])
        if not jobs:
            return
        self.keep_layer = self._layer()
        self.followups = [(session_id, job) for job in jobs[1:]]
        self.active_id = session_id
        self._start(jobs[0])

    def pick_cleanup(self, level: str) -> None:
        """Show `level`: the saved result if there is one, otherwise run it."""
        session_id = self._ready_session()
        if session_id is None or not self.cleanup_action.isEnabled():
            return
        self._save_edits()
        current = current_info(self.store, session_id, "ai")
        if current.get("style") == level:
            self._show_layer("AI")
            return
        mode = str(current.get("mode") or self.cleanup_mode.currentData())
        try:
            saved = find_cleanup(saved_results(self.store, session_id, "ai"), level, mode)
            if saved is None:
                self.start_cleanup(level)
                return
            use_saved(self.store, session_id, "ai", saved.revision_id)
            # Bring the notes and summary made from that cleanup with it, when saved.
            style = self._notes_style(session_id)
            missing = self._use_saved_style(session_id, saved.revision_id, style)
        except (CleanupError, OSError, ValueError) as error:
            self.status.setText(f"Could not show that cleanup: {error}")
            return
        self.show_selected()
        self.status.setText(f"Showing the saved {level.capitalize()} cleanup.")
        self._make_missing(session_id, missing, style)  # What this level still lacks.

    def redo_all(self) -> None:
        """Make the shown cleanup, notes, and summary again in one job; afterwards the
        background remakes the other levels and styles (older results count as missing)."""
        session_id = self._ready_session()
        if session_id is None or not self.cleanup_action.isEnabled():
            return
        self._save_edits()
        self.redo_since[session_id] = datetime.now(UTC)
        self.background = [job for job in self.background if job[0] != session_id]
        self._update_background_label()
        self.start_cleanup(str(current_info(self.store, session_id, "ai").get("style") or ""))

    def _notes_style(self, session_id: str) -> str:
        style = str(
            current_info(self.store, session_id, "formatted").get("style")
            or self.format_style.currentData()
        )
        return style if style in FORMAT_NAMES else "mixed"

    def pick_format(self, style: str) -> None:
        """Show Notes and Summary in `style`: saved ones from this cleanup, otherwise make
        the missing ones (notes first, then the summary)."""
        session_id = self._ready_session()
        if session_id is None or not self.format_choices[style].isEnabled():
            return
        self._save_edits()
        revision = current_info(self.store, session_id, "ai").get("revision_id")
        try:
            missing = self._use_saved_style(session_id, revision, style)
        except (CleanupError, OSError, ValueError) as error:
            self.status.setText(f"Could not show that style: {error}")
            return
        if self._layer() not in ("Formatted", "Summary"):
            self._show_layer("Formatted")
        self.show_selected()
        if missing:
            self._make_missing(session_id, missing, style)
        else:
            self.status.setText(f"Showing the saved {FORMAT_NAMES[style]} notes and summary.")

    def set_recording_type(self, mode: str) -> None:
        """The Lecture | Dictation switch: remembered for new imports, and applied to the
        selected lecture (re-cleaned, or shown from a saved result in that mode)."""
        changed = mode != choice(self.cleanup_mode)
        select(self.cleanup_mode, mode)
        self._sync_mode_buttons()
        if changed:
            try:
                self._save_settings()
            except OSError as error:
                self.status.setText(f"Could not save settings: {error}")
        session_id = self._ready_session()
        current = current_info(self.store, session_id, "ai") if session_id else {}
        if session_id is None or not current or current.get("mode") == mode:
            if changed:
                self.status.setText(f"New imports are cleaned as {mode.capitalize()}.")
            return
        if not self.cleanup_action.isEnabled():
            return
        self._save_edits()
        # Background versions in the old mode are no longer wanted for this lecture.
        self.background = [job for job in self.background if job[0] != session_id]
        self._update_background_label()
        level = str(current.get("style") or "")
        style = self._notes_style(session_id)
        try:
            saved = find_cleanup(saved_results(self.store, session_id, "ai"), level, mode)
            if saved is None:
                self.keep_layer = self._layer()
                self.start_cleanup(level)  # Cleans in the switch's mode, with notes + summary.
                return
            use_saved(self.store, session_id, "ai", saved.revision_id)
            missing = self._use_saved_style(session_id, saved.revision_id, style)
        except (CleanupError, OSError, ValueError) as error:
            self.status.setText(f"Could not switch this lecture: {error}")
            return
        self.show_selected()
        self.status.setText(f"Showing this lecture cleaned as {mode.capitalize()}.")
        self._make_missing(session_id, missing, style)

    def _sync_mode_buttons(self, *_: object) -> None:
        for value, button in self.mode_buttons.items():
            button.setChecked(value == choice(self.cleanup_mode))

    @property
    def background_setting(self) -> bool:
        """Whether Settings asks for anything to be prepared in the background."""
        return any(self.background_prepare.values())

    def _wanted(self, arguments: list[str]) -> bool:
        """Whether a planned background step is still among the chosen kinds."""
        kind = {"format": "notes", "summarize": "summaries", "cleanup": "levels"}
        return self.background_prepare.get(kind.get(arguments[0], ""), False)

    def schedule_background(self, session_id: str | None) -> None:
        """Queue the versions chosen under Settings that this lecture does not have yet."""
        if not session_id or self.closing or not self.cleanup_available:
            return
        if not self.background_enabled or not self.background_setting:
            return
        try:
            plan = background_plan(
                self.store,
                session_id,
                str(self.cleanup_model.currentData()),
                PROCESSING_DEFAULTS["lecture"][1],
                self.redo_since.get(session_id),
                notes=self.background_prepare["notes"],
                summaries=self.background_prepare["summaries"],
                levels=self.background_prepare["levels"],
            )
        except OSError, ValueError, KeyError, TypeError:
            return  # Background versions are a convenience; never block the window.
        for arguments in plan:
            job = (session_id, arguments)
            if job not in self.background and job != self.bg_job:
                self.background.append(job)
        self._update_background_label()
        QTimer.singleShot(0, self._run_background_next)

    def pause_background(self) -> None:
        """Let something the user started run first; the paused step runs again later."""
        if self.bg_stop is not None:
            try:
                self.bg_stop.touch()
            except OSError:
                pass

    def _run_background_next(self) -> None:
        if (
            self.closing
            or self._job_active
            or self.processing
            or self.bg_job is not None
            or not self.background
            or not self.background_setting
        ):
            return
        session_id, arguments = self.background.pop(0)
        if session_id not in self.sessions:
            QTimer.singleShot(0, self._run_background_next)
            return
        try:
            jobs = self.store.data_dir / "jobs"
            jobs.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        self.bg_job = (session_id, arguments)
        self.bg_stop = jobs / f"{uuid.uuid4().hex}.stop"
        self.bg_output = ""
        self._update_background_label()
        self.worker.run(self._command(arguments, self.bg_stop), self._bg_read, self._bg_done)

    def _bg_read(self, text: str = "") -> None:
        self.bg_output = (self.bg_output + text)[-8192:]

    def _bg_done(self, code: int, crashed: bool) -> None:
        job, stop = self.bg_job, self.bg_stop
        self.bg_job = self.bg_stop = None
        paused = stop is not None and stop.exists()
        if stop is not None:
            try:
                stop.unlink(missing_ok=True)
            except OSError:
                pass
        if job is not None:
            session_id, arguments = job
            if code == 0:
                if arguments[0] == "cleanup":
                    self.schedule_background(session_id)  # Note styles for the new level.
            elif paused and not crashed:
                if session_id != self.pending_trash and self._wanted(arguments):
                    self.background.insert(0, job)  # Something else ran first; try again.
            else:
                # Usually the AI model cannot run; stop rather than fail every step.
                self.background = [j for j in self.background if j[0] != session_id]
                lines = [line for line in self.bg_output.splitlines() if line.strip()]
                self.background_label.setToolTip(
                    "Preparing other versions stopped: "
                    + (lines[-1] if lines else self.worker.error_text)
                )
        self._update_background_label()
        if self.pending_trash:
            QTimer.singleShot(0, lambda sid=self.pending_trash: self._trash_now(sid))
        QTimer.singleShot(0, self._run_background_next)

    def _update_background_label(self) -> None:
        left = len(self.background) + (self.bg_job is not None)
        self.background_label.setText("" if not left else f"Preparing other versions… {left} left")
        self.background_label.setVisible(bool(left))

    def set_background(self, kind: str, enabled: bool) -> None:
        """One Settings choice for what to prepare in the background."""
        self.background_prepare[kind] = enabled
        try:
            self._save_settings()
        except OSError as error:
            self.status.setText(f"Could not save settings: {error}")
        # Plan again from the new choices; a running step no longer wanted stops after
        # its current block.
        self.background.clear()
        if self.bg_job is not None and not self._wanted(self.bg_job[1]):
            self.pause_background()
        self.schedule_background(self.selected_id())
        self._update_background_label()

    def start_summary(self, style: str = "") -> None:
        session_id = self.selected_id()
        if self.processing or not session_id or not self.summary_action.isEnabled():
            return
        self.active_id = session_id
        self._start(
            [
                "summarize",
                session_id,
                "--style",
                style or self._notes_style(session_id),
                "--model",
                str(self.cleanup_model.currentData()),
            ]
        )

    def _start(self, arguments: list[str]) -> None:
        try:
            self._save_settings()
            jobs = self.store.data_dir / "jobs"
            jobs.mkdir(parents=True, exist_ok=True)
            self.stop_file = jobs / f"{uuid.uuid4().hex}.stop"
        except OSError as error:
            self.status.setText(f"Could not save settings: {error}")
            return
        self.job_kind = (
            arguments[0] if arguments[0] in ("cleanup", "summarize", "format") else "transcription"
        )
        self.before_ids = set(self.sessions)
        self.job_arguments = arguments
        self.paused_jobs.pop(self.active_id or "", None)  # A new run replaces a paused one.
        self.output = ""
        self.cleanup_written = False
        self.ai_stage = ""
        self.pause_background()  # What the user started runs before background versions.
        command = self._command(arguments, self.stop_file)
        self.status.setText(
            (
                "Preparing the summary…"
                if self.job_kind == "summarize"
                else "Preparing transcript layout…"
                if self.job_kind == "format"
                else "Preparing local AI cleanup…"
            )
            + " Earlier versions stay available."
            if self.job_kind in ("cleanup", "summarize", "format")
            else "Preparing the audio… Pause stops after the current 30-second part."
        )
        self.progress.setRange(0, 0)
        self._job_active = True
        self.timer.start()
        self._buttons()
        self.worker.run(command, self._read_output, self._job_done)

    def _command(self, arguments: list[str], stop_file: Path) -> list[str]:
        ai_job = arguments[0] in ("cleanup", "summarize", "format")
        command = [
            "--data-dir",
            str(self.store.data_dir),
            "--model-root",
            str(self.model_root),
            *arguments,
            "--device",
            choice(self.cleanup_device) if ai_job else choice(self.device),
            "--stop-file",
            str(stop_file),
        ]
        if not ai_job:
            command += ["--cleanup", "off", "--formatting", "off"]
            if self.ffmpeg:
                command += ["--ffmpeg", self.ffmpeg]
        return command

    def pause(self) -> None:
        if self.busy and self.stop_file:
            try:
                self.stop_file.touch()
                self.status.setText(
                    "Pausing after the current part finishes…"
                    if self.job_kind == "transcription"
                    else "Pausing after the current block of text finishes…"
                )
                self._buttons()
            except OSError as error:
                self.status.setText(f"Could not request pause: {error}")

    def _read_output(self, text: str = "") -> None:
        self.output = (self.output + text)[-8192:]
        if self.job_kind in ("cleanup", "summarize", "format"):
            # A lecture cleanup runs cleanup, then formatting, then the summary in one
            # worker; summary progress reuses cleanup_blocks after "AI cleanup ready".
            if "AI cleanup ready: " in self.output:
                self.cleanup_written = True
            for line in reversed(self.output.splitlines()):
                if line.startswith(("cleanup_blocks: ", "format_blocks: ")):
                    try:
                        done, total = map(int, line.split(": ", 1)[1].split("/"))
                    except ValueError:
                        break
                    stage = (
                        "Formatting"
                        if line.startswith("format_blocks: ")
                        else "Writing summary"
                        if self.job_kind == "summarize" or self.cleanup_written
                        else "Cleaning up"
                    )
                    self.progress.setRange(0, max(1, total))
                    self.progress.setValue(done)
                    self.progress.setFormat(f"{stage}: {done} / {total}")
                    if stage != self.ai_stage and not (
                        self.stop_file is not None and self.stop_file.exists()
                    ):
                        self.ai_stage = stage
                        self.status.setText(f"{stage}… Earlier versions stay available.")
                    break

    def _job_done(self, code: int, crashed: bool) -> None:
        self._finished(
            code, QProcess.ExitStatus.CrashExit if crashed else QProcess.ExitStatus.NormalExit
        )

    def _finished(self, code: int, exit_status: QProcess.ExitStatus) -> None:
        self._job_active = False
        if code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self.followups.clear()
            self.keep_layer = ""
        self.job_reported = True
        self._read_output()
        self.timer.stop()
        cleanup_error = ""
        was_paused = self.stop_file is not None and self.stop_file.exists()
        if self.stop_file:
            try:
                self.stop_file.unlink(missing_ok=True)
            except OSError as error:
                cleanup_error = f" Could not remove the completed job's pause marker: {error}"
        self.refresh()
        if code == 0 and exit_status == QProcess.ExitStatus.NormalExit:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat("Complete")
            if self.job_kind in ("cleanup", "summarize", "format"):
                target_layer = self._show_ai_result()
                self.schedule_background(self.active_id)
                self.status.setText(
                    (
                        "Summary ready. Check it against the Original tab."
                        if self.job_kind == "summarize"
                        else "Notes ready. Check them against the Original tab."
                        if target_layer == "Formatted"
                        else "AI cleanup ready. Check it against the Original tab."
                    )
                    + (
                        " A written summary is in the Summary version."
                        if self.job_kind == "cleanup" and "Summary ready: " in self.output
                        else ""
                    )
                )
                if self.followups and not self.closing:
                    session_id, job = self.followups.pop(0)
                    self.active_id = session_id
                    self._start(job)
                else:
                    self.keep_layer = ""
            else:
                self._show_layer("Raw")
                if self.active_id:
                    for row in range(self.library.count()):
                        if self.library.item(row).data(Qt.ItemDataRole.UserRole) == self.active_id:
                            self.library.setCurrentRow(row)
                            break
                if not self.closing and not was_paused and choice(self.cleanup_style) != "off":
                    self._buttons()
                    if self.active_id:
                        self._run_cleanup(self.active_id)
                    if self.processing:
                        return
                if (
                    not self.closing
                    and not was_paused
                    and self.format_style.currentData() != "off"
                    and self.active_id
                ):
                    self._run_formatting(self.active_id, "raw")
                    if self.processing:
                        return
                self.status.setText("Transcript ready.")
        elif code == 130 and exit_status == QProcess.ExitStatus.NormalExit:
            if self.job_kind == "cleanup" and self.cleanup_written and self.active_id:
                self._show_ai_result()
                # Cleanup is saved; continue with the step that was interrupted.
                model = str(self.cleanup_model.currentData())
                self.paused_jobs[self.active_id] = (
                    ["summarize", self.active_id, "--model", model]
                    if self.ai_stage == "Writing summary"
                    else [
                        "format",
                        self.active_id,
                        "--source",
                        "ai",
                        "--style",
                        str(self.format_style.currentData()),
                        "--model",
                        model,
                    ]
                )
            elif self.job_kind in ("cleanup", "summarize", "format") and self.active_id:
                self.paused_jobs[self.active_id] = self.job_arguments
            self.status.setText(
                (
                    "Summary paused. "
                    if self.job_kind == "summarize" or self.ai_stage == "Writing summary"
                    else "Formatting paused. "
                    if self.job_kind == "format" or self.ai_stage == "Formatting"
                    else "AI cleanup paused. "
                )
                + "Press Resume to start that step again. Earlier versions are kept."
                if self.job_kind in ("cleanup", "summarize", "format")
                else "Transcription paused. Press Resume to continue where it stopped."
            )
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            # The worker's last line states the failure; earlier lines are progress.
            lines = [line.strip() for line in self.output.splitlines() if line.strip()]
            reason = lines[-1].removeprefix("error: ") if lines else self.worker.error_text
            if self.job_kind == "cleanup" and self.cleanup_written:
                # Cleanup was saved; a later step (formatting or the summary) failed.
                self._show_ai_result()
                self.status.setText(f"{reason} Format → Write a summary tries again.")
            else:
                self.status.setText(
                    (
                        "Transcript remains ready. "
                        if self.job_kind in ("cleanup", "summarize", "format")
                        else ""
                    )
                    + f"{self.job_kind.capitalize()} failed (exit {code}). {reason}"
                )
        self._buttons()
        if cleanup_error:
            self.status.setText(self.status.text() + cleanup_error)
        if not self._job_active and not self.closing:
            self._notify_finished()
            QTimer.singleShot(0, self._next_import)
            QTimer.singleShot(0, self._run_background_next)
        if self.closing:
            self.close()

    def _show_ai_result(self) -> str:
        """Show the version the finished AI job produced; returns its internal name."""
        if self.keep_layer:
            # A style or level pick: stay on the tab the user was reading.
            session = self.sessions.get(self.active_id or "")
            if session is not None and self._existing_layers(session).get(self.keep_layer):
                self._show_layer(self.keep_layer)
                self.show_selected()
                return self.keep_layer
        target_layer = "Summary" if self.job_kind == "summarize" else "AI"
        if self.job_kind == "format" or (
            self.job_kind == "cleanup"
            and self.format_style.currentData() != "off"
            and (
                self.store.session_dir(self.active_id or "") / "formatted-transcript.json"
            ).is_file()
        ):
            target_layer = "Formatted"
        self._show_layer(target_layer)
        self.show_selected()
        return target_layer

    def refresh(self) -> None:
        selected = self.selected_id()
        sessions: dict[str, Session] = {}
        errors = 0
        try:
            folders = list(self.store.lectures.iterdir()) if self.store.lectures.exists() else []
        except OSError as error:
            self.status.setText(f"Could not read the lecture library: {error}")
            self._buttons()
            return
        if folders:
            for folder in folders:
                if not folder.is_dir():
                    continue
                try:
                    session = self.store.load_session(folder.name)
                    if (
                        session.id != folder.name
                        or not isinstance(session.title, str)
                        or not isinstance(session.created_at, str)
                        or not isinstance(session.diagnostics, dict)
                    ):
                        raise ValueError("invalid session metadata")
                    sessions[session.id] = session
                except OSError, ValueError, TypeError, KeyError, AttributeError:
                    errors += 1
        self.sessions = sessions
        target = selected
        if self.active_id is None:
            created = set(sessions) - self.before_ids
            if len(created) == 1:
                self.active_id = next(iter(created))
                target = self.active_id
        if target is None and self.restore_lecture in sessions:
            target = self.restore_lecture  # The lecture open when the app last closed.
        self.restore_lecture = ""
        self.job_percent: int | None = None
        if self.active_id in sessions and self.job_kind == "transcription":
            checkpoint = self.store.checkpoint_path(str(self.active_id))
            if checkpoint.exists():
                try:
                    value = json.loads(checkpoint.read_text(encoding="utf-8"))
                    total = int(value["total_chunks"])
                    done = len(value["completed_chunks"])
                    self.progress.setRange(0, max(1, total))
                    self.progress.setValue(done)
                    self.progress.setFormat(f"{done} / {total} parts")
                    if self._job_active:
                        self.job_percent = round(100 * done / max(1, total))
                except OSError, ValueError, KeyError, TypeError:
                    pass  # Atomic replacements can briefly be unavailable on Windows.
        scroll = self.library.verticalScrollBar().value()
        self.library.blockSignals(True)
        self.library.clear()
        for session in sorted(sessions.values(), key=lambda s: s.created_at, reverse=True):
            facts = [self.status_label(session)]
            duration = session.diagnostics.get("source_duration_seconds")
            if isinstance(duration, (int, float)) and duration > 0:
                facts.append(timestamp(duration))
            facts.append(short_date(session.created_at))
            item = QListWidgetItem(f"{display_title(session.title)}\n{' · '.join(facts)}")
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.library.addItem(item)
            if session.id == target:
                self.library.setCurrentItem(item)
        self.library.blockSignals(False)
        # Rebuilding the list during a job must not jump it back to the top.
        self.library.verticalScrollBar().setValue(scroll)
        self.library_stack.setCurrentIndex(1 if sessions else 0)
        self.library_search.setVisible(len(sessions) > 8 or bool(self.library_search.text()))
        self._filter_library(self.library_search.text())
        self.show_selected()
        if errors and not self.processing:
            self.status.setText(
                f"Skipped {errors} unreadable session(s). Their files were left intact."
            )
        self._buttons()

    def _buttons(self) -> None:
        for action in (
            self.manage_models_action,
            self.models_action,
            self.ffmpeg_action,
            self.defaults_action,
            *(menu.menuAction() for menu in self.choice_menus),
        ):
            action.setEnabled(not self.processing)
        for control in (
            self.import_button,
            self.manage_models_button,
            self.model,
            self.device,
            self.cleanup_model,
            self.cleanup_mode,
            self.cleanup_style,
            self.cleanup_device,
            self.format_style,
        ):
            control.setEnabled(not self.processing)
        # Parakeet always runs on the CPU, so its device choice is shown but disabled.
        cpu_model = cpu_only(choice(self.model))
        self.device.setEnabled(not self.processing and not cpu_model)
        self.device.setToolTip(CPU_ONLY_HINT if cpu_model else DEVICE_HINT)
        # Downloads write into the model folder, so it cannot change during one.
        self.models_action.setEnabled(not self.busy)
        # AI stages need the separately downloaded text model; settle choices before use.
        cleanup_ready = self.cleanup_available
        self.cleanup_hint.setVisible(not cleanup_ready)
        self.cleanup_style.setEnabled(not self.processing and cleanup_ready)
        format_items = cast(QStandardItemModel, self.format_style.model())
        for value in ("mixed", "structured"):
            item = format_items.item(self.format_style.findData(value))
            item.setEnabled(cleanup_ready)
        if not cleanup_ready:
            turn_off = [(self.cleanup_style, self.cleanup_style.findData("off"))]
            if self.format_style.currentData() in ("mixed", "structured"):
                turn_off.append((self.format_style, self.format_style.findData("off")))
            for combo, off in turn_off:
                combo.blockSignals(True)
                combo.setCurrentIndex(off)
                combo.blockSignals(False)
        if self.dictation_window is not None:
            self.dictation_window._buttons()
        pausing = self.stop_file is not None and self.stop_file.exists()
        self.pause_button.setVisible(self._job_active)
        self.pause_button.setEnabled(
            self._job_active and self.stop_file is not None and not pausing
        )
        self.pause_button.setText("Pausing…" if self._job_active and pausing else "Pause")
        session = self.sessions.get(self.selected_id() or "")
        self.resume_button.setVisible(self.resumable(session))
        self.resume_button.setEnabled(self.resumable(session))
        self.show_recording_action.setEnabled(session is not None)
        completed = session is not None and session.status == "ready"
        edited = session is not None and session.edited_transcript is not None
        ai_ready = (
            session is not None
            and (self.store.session_dir(session.id) / "ai-transcript.json").is_file()
        )
        summary_ready = (
            session is not None
            and (self.store.session_dir(session.id) / "summary-transcript.json").is_file()
        )
        self.cleanup_action.setEnabled(not self.processing and completed and cleanup_ready)
        self.summary_action.setEnabled(not self.processing and completed and cleanup_ready)
        self.format_action.setEnabled(not self.processing and completed)
        for value, action in self.format_choices.items():
            action.setEnabled(value == "prose" or cleanup_ready)
        self.progress.setVisible(self._job_active)
        self.cleanup_button.setEnabled(self.cleanup_action.isEnabled())
        self.format_button.setEnabled(self.format_action.isEnabled())
        formatted_ready = (
            session is not None
            and (self.store.session_dir(session.id) / "formatted-transcript.json").is_file()
        )
        shown_exists = (
            completed
            and (self._layer() != "Edited" or edited)
            and (self._layer() != "AI" or ai_ready)
            and (self._layer() != "Formatted" or formatted_ready)
            and (self._layer() != "Summary" or summary_ready)
        )
        self.export_button.setEnabled(shown_exists)
        self.copy_button.setEnabled(shown_exists)
        for value, action in self.export_actions.items():
            # Notes, summaries, and typed copies have no timing for subtitles.
            no_timing = self._layer() in ("Formatted", "Summary") or self.showing_copy
            action.setEnabled(shown_exists and not (value == "srt" and no_timing))
        self.edit_button.setEnabled(
            not self.processing
            and completed
            and self._layer() not in ("AI", "Summary", "Formatted")
            and ((self._layer() == "Edited") == edited)
        )
        self.history_button.setEnabled(edited)
        self.history_button.setVisible(edited)

    def show_selected(self, *_: object) -> None:
        # Typing not yet saved belongs to the version still in the box; save it first.
        self._save_edits()
        session = self.sessions.get(self.selected_id() or "")
        if session is None:
            self.player.load(None)
            self.segment_starts = []
            self._render(None, "", lambda: self.transcript.clear())
            self.version_tabs.hide()
            self.revert_button.hide()
            self.details.setText("Import audio or select a lecture from the library.")
            self.details.setToolTip("")
            self.shown_id = ""
            self._buttons()
            return
        if session.id != self.shown_id:
            # A lecture coming into view (clicked, or picked at startup) opens on its most
            # finished version; later refreshes keep whatever the user or a job chose.
            self.shown_id = session.id
            if session.status == "ready":
                exists = self._existing_layers(session)
                version = (
                    self.restore_version
                    if exists.get(self.restore_version)
                    else self._best_layer(exists)
                )
                with QSignalBlocker(self.layer):
                    self._show_layer(version)
                QTimer.singleShot(0, lambda sid=session.id: self.schedule_background(sid))
            self.restore_version = ""
        diag = session.diagnostics
        try:
            self.player.load(self.store.source_copy(session.id))
        except OSError, ValueError, KeyError:
            self.player.load(None)
        self.segment_starts = []
        lines = [f"{display_title(session.title)} · {self.status_label(session)}"]
        # Model and device details are for the curious; they stay in the hover text.
        technical = [
            f"Speech model: {diag.get('model_id', 'unknown')} "
            f"on {diag.get('actual_device', 'pending')}"
        ]
        if self.resumable(session):
            lines.append("This lecture stopped partway. Press Resume to continue.")
        cleaned_as = current_info(self.store, session.id, "ai").get("mode")
        if cleaned_as and cleaned_as != choice(self.cleanup_mode):
            lines.append(
                f"Cleaned as {cleaned_as.capitalize()}. Click "
                f"{choice(self.cleanup_mode).capitalize()} above to clean it that way."
            )
        mine = False
        if session.status == "ready":
            self._offer_existing_layers(session)
            layer = self._layer()
            try:
                transcript = self.store.load_transcript(session.id, layer.lower())
                if layer in USER_COPIES:
                    text = "\n\n".join(s.text for s in transcript.segments)
                    metadata = transcript.transformation or {}
                    records = metadata.get("blocks", [])
                    retained = sum(bool(record.get("used_source")) for record in records)
                    provenance = transcript.provenance
                    if layer == "Formatted":
                        technical.append(
                            "Notes: "
                            + FORMAT_NAMES.get(str(metadata.get("style")), "unknown style")
                            + ", from "
                            + SOURCE_NAMES.get(str(metadata.get("source_layer")), "unknown")
                            + (
                                ", no AI"
                                if provenance.actual_device == "none"
                                else f", {provenance.model} on {provenance.actual_device}"
                            )
                        )
                    else:
                        technical.append(
                            ("Summary: " if layer == "Summary" else "AI cleanup: ")
                            + f"{provenance.model} on {provenance.actual_device}"
                        )
                    # The user's own copy is shown only over the AI version it was typed on.
                    copy = load_user_text(self.store, session.id, USER_COPIES[layer])
                    self.rendered_user_rev = copy.revision_id if copy else None
                    if copy is not None and copy.based_on_revision == metadata.get("revision_id"):
                        text, mine = copy.text, True
                        lines.append("Showing your edited copy.")
                    elif copy is not None:
                        lines.append(
                            "Your edits were made on a different version and are kept in "
                            "history. Typing here starts a new edited copy."
                        )
                    elif retained:
                        lines.append(
                            f"{retained} part(s) kept the original text after a check failed. "
                            "Review them against the Original tab."
                        )
                    if layer in ("Formatted", "Summary"):
                        self._render(session, layer, lambda: self._show_markdown(text))
                    else:
                        self._render(session, layer, lambda: self.transcript.setPlainText(text))
                else:
                    self._render(session, layer, lambda: self._show_segments(transcript.segments))
                    self.segment_starts = [s.start for s in transcript.segments]
                if layer == "Edited":
                    lines.append("Edited by user; the original is kept.")
            except (OSError, ValueError, TypeError, KeyError) as error:
                message = f"Could not read transcript: {error}"
                self._render(None, "", lambda: self.transcript.setPlainText(message))
        else:
            self._render(None, "", lambda: self.transcript.clear())
            self.version_tabs.hide()
        self.revert_button.setVisible(mine)
        self.showing_copy = mine
        self.details.setText("\n".join(lines))
        self.details.setToolTip("\n".join(technical))
        self._buttons()
        self.player.seek_button.setEnabled(bool(self.segment_starts))
        self._highlight_playing(self.player.player.position())

    def _editable(self, session: Session | None, layer: str) -> bool:
        if session is None or session.status != "ready":
            return False
        if layer == "Raw":
            # Once an Edited version exists, edits continue there; Original stays as heard.
            return session.edited_transcript is None
        return layer in ("Edited", *USER_COPIES)

    def _version_stamp(self, session: Session, layer: str) -> tuple[object, ...]:
        """Changes when the version's files change, so a refresh never resets typing."""
        directory = self.store.session_dir(session.id)
        names = [f"{layer.lower()}-transcript.json"]
        if layer in USER_COPIES:
            names.append(f"{USER_COPIES[layer]}-user.json")
        stamps: list[object] = [session.edited_transcript]
        for name in names:
            path = directory / name
            stamps.append(path.stat().st_mtime_ns if path.is_file() else 0)
        return tuple(stamps)

    def _render(self, session: Session | None, layer: str, fill: Callable[[], None]) -> None:
        """Replace the box's text unless it already shows this exact version."""
        key = (session.id, layer, self._version_stamp(session, layer)) if session else None
        if key is not None and key == self.rendered_key:
            return
        self.rendering = True
        try:
            # Plain text takes the formatting at the cursor; after Notes that is a heading,
            # which made the whole Transcript large and bold. Start from a blank box.
            self.transcript.clear()
            self.transcript.setCurrentCharFormat(QTextCharFormat())
            fill()
            # Replacing the document keeps the old scroll position; start at the top.
            self.transcript.moveCursor(QTextCursor.MoveOperation.Start)
            self.dirty = False
        finally:
            self.rendering = False
        self.rendered_key = key
        self.rendered = (session.id, layer) if session else None
        self.transcript.setReadOnly(not self._editable(session, layer))
        self.cursor_segment = 0
        self.playing_segment = -1
        self.transcript.setExtraSelections([])
        self.saved_label.setText("")

    def _text_edited(self) -> None:
        if not self.rendering and not self.transcript.isReadOnly():
            # Any change the app did not make itself is an edit. Qt's own "modified" flag
            # is not used: replacing the whole text in one step (accessibility tools) clears it
            # after this signal.
            self.dirty = True
            self.saved_label.setText("Saving…")
            self.save_timer.start()

    def _save_edits(self) -> None:
        """Save typing in the box as the user's copy (Notes, Summary, Transcript) or as
        Edited segments (Original); nothing to do when the text is unchanged."""
        self.save_timer.stop()
        document = self.transcript.document()
        if self.rendered is None or not self.dirty:
            return
        session_id, layer = self.rendered
        self.dirty = False
        try:
            if layer in USER_COPIES:
                text = (
                    document.toPlainText()
                    if layer == "AI"
                    else document.toMarkdown(QTextDocument.MarkdownFeature.MarkdownNoHTML)
                )
                copy = save_user_text(
                    self.store, session_id, USER_COPIES[layer], text, self.rendered_user_rev
                )
                self.rendered_user_rev = copy.revision_id
                self.revert_button.show()
                self.showing_copy = True
            else:
                if not self._save_segment_edits(session_id, layer):
                    return
            session = self.sessions.get(session_id)
            if session is not None and self.rendered == (session_id, layer):
                # The box already shows what was saved: no reload, cursor stays put.
                self.rendered_key = (session_id, layer, self._version_stamp(session, layer))
            self.saved_label.setText("Saved")
        except UserTextConflict:
            self.status.setText(
                "This version was changed elsewhere, so the latest typing was not saved. "
                "Select the lecture again to reload it."
            )
        except (CleanupError, OSError, ValueError, TypeError, KeyError) as error:
            self.dirty = True  # Try again on the next pause in typing.
            self.saved_label.setText("Not saved")
            self.status.setText(f"Could not save your edits: {error}")

    def _save_segment_edits(self, session_id: str, layer: str) -> bool:
        """Original and Edited keep one paragraph per time stamp; save each changed line."""
        transcript = self.store.load_transcript(session_id, layer.lower())
        segments = transcript.segments
        document = self.transcript.document()
        if document.blockCount() != max(1, 2 * len(segments) - 1):
            self.dirty = True
            self.status.setText(
                "Not saved yet: keep one paragraph per time stamp, with a blank line between."
            )
            return False
        changed: list[tuple[int, str, bool]] = []
        for index, segment in enumerate(segments):
            line = document.findBlockByNumber(2 * index).text()
            stamp = timestamp(segment.start) + "   "
            if not line.startswith(stamp):
                self.dirty = True
                self.status.setText(
                    "Not saved yet: keep each time stamp at the start of its paragraph."
                )
                return False
            text = " ".join(line[len(stamp) :].removeprefix("[unclear] ").split())
            if text != " ".join(segment.text.split()):
                changed.append((index, text, segment.uncertain))
        if not changed:
            return True
        expected = current_revision(self.store, session_id)
        base = load_revision(self.store, session_id).base_layer if expected else layer.lower()
        for index, text, uncertain in changed:
            expected = save_segment_edit(
                self.store,
                session_id,
                index,
                text,
                uncertain,
                base_layer=base,
                expected_revision=expected,
            ).revision
        self.sessions[session_id] = self.store.load_session(session_id)
        if layer == "Raw":
            # The edits now live in Edited; show it with the cursor where it was.
            position = self.transcript.textCursor().position()
            self._show_layer("Edited")
            cursor = self.transcript.textCursor()
            cursor.setPosition(min(position, self.transcript.document().characterCount() - 1))
            self.transcript.setTextCursor(cursor)
        self.status.setText("Your edits are saved as the Edited version. Original is unchanged.")
        self.saved_label.setText("Saved")
        return True

    def revert_edits(self) -> None:
        """Show the AI's version again; the user's copy stays in its history."""
        if self.rendered is None or self.rendered[1] not in USER_COPIES:
            return
        self.save_timer.stop()
        self.dirty = False
        session_id, layer = self.rendered
        try:
            discard_user_text(self.store, session_id, USER_COPIES[layer])
        except (CleanupError, OSError) as error:
            self.status.setText(f"Could not undo your edits: {error}")
            return
        self.rendered_key = None
        self.show_selected()
        self.status.setText("Showing the AI version again. Your edits are kept in history.")

    def _show_markdown(self, text: str) -> None:
        """Notes and summaries; headings a modest step above the text instead of Qt's
        default, which made top headings about twice the body size."""
        document = self.transcript.document()
        document.setMarkdown(text, QTextDocument.MarkdownFeature.MarkdownNoHTML)
        document.setIndentWidth(22)  # Qt's 40 px per level pushed bullets far right.
        block = document.begin()
        while block.isValid():
            listed = block.textList()
            if listed is not None:
                # Every list is styled by its depth. (Skipping lists already seen by id()
                # failed: Qt hands out fresh wrappers whose ids get reused.)
                marker = listed.format()
                if marker.style() in (
                    QTextListFormat.Style.ListDisc,
                    QTextListFormat.Style.ListCircle,
                    QTextListFormat.Style.ListSquare,
                ):
                    marker.setStyle(
                        (
                            QTextListFormat.Style.ListDisc,
                            QTextListFormat.Style.ListCircle,
                            QTextListFormat.Style.ListSquare,
                        )[min(2, max(0, marker.indent() - 1))]
                    )
                    listed.setFormat(marker)
            level = block.blockFormat().headingLevel()
            if level:
                heading = QTextCharFormat()
                # Size steps: 1 = large, 0 = normal (bold still marks lower levels).
                heading.setProperty(QTextFormat.Property.FontSizeAdjustment, 1 if level <= 2 else 0)
                cursor = QTextCursor(block)
                cursor.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
                )
                cursor.mergeCharFormat(heading)
            block = block.next()

    def _show_segments(self, segments: Iterable[Segment]) -> None:
        """One paragraph per segment with a light start time; the blank line between
        segments keeps block number // 2 equal to the segment index for seeking."""
        self.transcript.clear()
        cursor = QTextCursor(self.transcript.document())
        stamp = QTextCharFormat()
        stamp.setForeground(self.palette().color(QPalette.ColorRole.PlaceholderText))
        plain = QTextCharFormat()
        for index, segment in enumerate(segments):
            if index:
                cursor.insertBlock()
                cursor.insertBlock()
            cursor.insertText(timestamp(segment.start), stamp)
            # The spacing is plain so words typed right after the time are not grey.
            cursor.insertText(
                "   "
                + ("[unclear] " if segment.uncertain else "")
                + " ".join(segment.text.split()),
                plain,
            )
        self.transcript.moveCursor(QTextCursor.MoveOperation.Start)

    def _layer(self) -> str:
        """Internal name of the transcript version shown (Raw, Balanced, Edited, AI, ...)."""
        return str(self.layer.currentData())

    def _show_layer(self, name: str) -> None:
        self.layer.setCurrentIndex(self.layer.findData(name))

    def _existing_layers(self, session: Session) -> dict[str, bool]:
        directory = self.store.session_dir(session.id)
        return {
            name: (
                session.edited_transcript is not None
                if name == "Edited"
                else (directory / f"{name.lower()}-transcript.json").is_file()
            )
            for name in (str(self.layer.itemData(row)) for row in range(self.layer.count()))
        }

    def _best_layer(self, exists: dict[str, bool]) -> str:
        return next((name for name in BEST_VERSIONS if exists.get(name)), "Raw")

    def _tab_chosen(self, index: int) -> None:
        name = self.version_tabs.tabData(index)
        if name and name != self._layer():
            self._show_layer(str(name))

    def _offer_existing_layers(self, session: Session) -> None:
        """Only versions this lecture has can be chosen; a missing choice shows the best one."""
        exists = self._existing_layers(session)
        items = cast(QStandardItemModel, self.layer.model())
        for row in range(self.layer.count()):
            items.item(row).setEnabled(exists[str(self.layer.itemData(row))])
        if not exists.get(self._layer()):
            with QSignalBlocker(self.layer):
                self._show_layer(self._best_layer(exists))
        wanted = [name for name in VERSION_TABS if exists.get(name)]
        shown = [self.version_tabs.tabData(i) for i in range(self.version_tabs.count())]
        with QSignalBlocker(self.version_tabs):
            if wanted != shown:
                while self.version_tabs.count():
                    self.version_tabs.removeTab(0)
                for name in wanted:
                    index = self.version_tabs.addTab(VERSION_TABS[name])
                    self.version_tabs.setTabData(index, name)
                    tip = self.layer.itemData(
                        self.layer.findData(name), Qt.ItemDataRole.ToolTipRole
                    )
                    self.version_tabs.setTabToolTip(index, str(tip or ""))
            if self._layer() in wanted:
                self.version_tabs.setCurrentIndex(wanted.index(self._layer()))
        self.version_tabs.setVisible(len(wanted) > 1)

    def seek_selected_segment(self) -> None:
        index = self.transcript.textCursor().blockNumber() // 2
        if 0 <= index < len(self.segment_starts):
            self.player.seek(self.segment_starts[index])

    def find_text(self) -> None:
        query = self.search.text()
        if query and not self.transcript.find(query):
            self.transcript.moveCursor(QTextCursor.MoveOperation.Start)
            self.transcript.find(query)

    def edit_selected_segment(self) -> None:
        session_id = self.selected_id()
        if self.processing or session_id is None or not self.edit_button.isEnabled():
            return
        try:
            layer = self._layer().lower()
            expected = current_revision(self.store, session_id)
            base = load_revision(self.store, session_id).base_layer if expected else layer
            transcript = self.store.load_transcript(session_id, layer)
            index = self.transcript.textCursor().blockNumber() // 2
            if not 0 <= index < len(transcript.segments):
                return

            def commit(text: str, uncertain: bool) -> EditRevision:
                return save_segment_edit(
                    self.store,
                    session_id,
                    index,
                    text,
                    uncertain,
                    base_layer=base,
                    expected_revision=expected,
                )

            dialog = SegmentEditDialog(transcript.segments[index], commit)
            dialog.exec()
            if dialog.saved:
                self.sessions[session_id] = self.store.load_session(session_id)
                self._show_layer("Edited")
                self.show_selected()
                self.status.setText("Edit saved. The original stays unchanged.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status.setText(f"Could not open edit: {error}")

    def show_history(self) -> None:
        session_id = self.selected_id()
        if session_id:
            try:
                RevisionHistoryDialog(revision_history(self.store, session_id)).exec()
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.status.setText(f"Could not open history: {error}")

    def choose_export(self) -> None:
        session_id = self.selected_id()
        if not session_id:
            return
        self._save_edits()
        fmt = choice(self.export_format)
        if not self.export_actions[fmt].isEnabled():
            return
        suffix = {"markdown": "md", "text": "txt"}.get(fmt, fmt)
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Export transcript",
            f"{session_id}.{self._layer().lower()}.{suffix}",
            f"{fmt} (*.{suffix})",
        )
        if file:
            try:
                path = write_export(
                    self.store,
                    session_id,
                    self._layer().lower(),
                    fmt,
                    out_path=Path(file),
                    overwrite=True,
                )
                self.status.setText(f"Exported: {path}")
            except (OSError, ValueError, RuntimeError, ExportError) as error:
                QMessageBox.warning(self, "Export failed", str(error))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_edits()
        self.player.player.stop()
        if self.updates.task or (self.setup_dialog and self.setup_dialog.task):
            self.updates.pending_close = True
            if self.updates.task:
                self.updates.task.cancelled.set()
            if self.setup_dialog and self.setup_dialog.task:
                self.setup_dialog.task.cancelled.set()
                self.setup_dialog.task.finished.connect(self.close)
            event.ignore()
            return
        if self.busy:
            self.closing = True
            if self.dictation_window is not None:
                self.dictation_window.close()
            if self._job_active:
                self.pause()
            event.ignore()
        else:
            self.closing = False
            if self.dictation_window is not None:
                self.dictation_window.close()
            try:
                self._save_settings()  # Window size, layout, and last lecture.
            except OSError:
                pass
            # Exiting unloads every model: the model process ends with the app.
            self.background.clear()
            self.pause_background()
            self.preload_timer.stop()
            self.worker.stop()
            event.accept()
