"""The Trash window: deleted lectures stay restorable for 30 days."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .storage import SessionStore
from .trash import TRASH_DAYS, TrashError, empty_trash, list_trash, restore_lecture


def size_text(size: int) -> str:
    return f"{size / 1_000_000:.1f} MB" if size >= 100_000 else f"{max(1, size // 1000)} KB"


def days_left(expires_at: str) -> int:
    try:
        left = datetime.fromisoformat(expires_at) - datetime.now(UTC)
    except ValueError:
        return TRASH_DAYS
    return max(0, left.days + (1 if left.seconds else 0))


class TrashDialog(QDialog):
    def __init__(
        self, store: SessionStore, changed: Callable[[str], None], parent: QWidget | None = None
    ) -> None:
        """`changed(message)` is called after a restore or empty, to refresh the library."""
        super().__init__(parent)
        self.store = store
        self.changed = changed
        self.setWindowTitle("Trash")
        self.resize(620, 420)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Deleted lectures stay here for {TRASH_DAYS} days, then are removed "
                "permanently. Restore puts a lecture back in the library."
            )
        )
        self.list = QListWidget()
        self.list.setAccessibleName("Deleted lectures")
        self.list.currentItemChanged.connect(self._buttons)
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.restore_button = QPushButton("Restore")
        self.restore_button.clicked.connect(self.restore)
        row.addWidget(self.restore_button)
        self.empty_button = QPushButton("Empty Trash…")
        self.empty_button.setToolTip("Permanently delete everything in the Trash now.")
        self.empty_button.clicked.connect(self.empty)
        row.addWidget(self.empty_button)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)
        self.load()

    def load(self) -> None:
        self.list.clear()
        try:
            entries = list_trash(self.store)
        except (OSError, TrashError) as error:
            entries = []
            self.list.addItem(f"Could not read the Trash: {error}")
        for entry in entries:
            left = days_left(entry.expires_at)
            item = QListWidgetItem(
                f"{entry.title}\n"
                + ("Removed today" if left == 0 else f"Removed in {left} day(s)")
                + f" · {size_text(entry.size_bytes)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, entry.id)
            self.list.addItem(item)
        if not entries and self.list.count() == 0:
            self.list.addItem("The Trash is empty.")
        if entries:
            self.list.setCurrentRow(0)
        self.empty_button.setEnabled(bool(entries))
        self._buttons()

    def _buttons(self, *_: object) -> None:
        item = self.list.currentItem()
        self.restore_button.setEnabled(
            item is not None and item.data(Qt.ItemDataRole.UserRole) is not None
        )

    def restore(self) -> None:
        item = self.list.currentItem()
        session_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if session_id is None:
            return
        title = item.text().splitlines()[0]
        try:
            restore_lecture(self.store, str(session_id))
        except (OSError, TrashError) as error:
            QMessageBox.warning(self, "Could not restore", str(error))
            return
        self.load()
        self.changed(f"Restored {title}.")

    def empty(self) -> None:
        count = sum(
            self.list.item(row).data(Qt.ItemDataRole.UserRole) is not None
            for row in range(self.list.count())
        )
        answer = QMessageBox.question(
            self,
            "Empty Trash",
            f"Permanently delete {count} lecture(s) in the Trash? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = empty_trash(self.store, None)
        except (OSError, TrashError) as error:
            QMessageBox.warning(self, "Could not empty the Trash", str(error))
            return
        self.load()
        left = self.list.count() if self.empty_button.isEnabled() else 0
        self.changed(
            f"Permanently deleted {removed} lecture(s)."
            + (f" {left} could not be removed now and will be tried again later." if left else "")
        )
