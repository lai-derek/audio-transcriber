"""In-app Trash for deleted lectures.

Deleting moves the whole lecture folder, unchanged, into `trash/` beside a small
record. It can be restored for TRASH_DAYS; after that `empty_trash` removes it
for good. Nothing in the live library is ever removed automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .storage import SessionStore, atomic_json, safe_child

TRASH_DAYS = 30
BUSY_STATUSES = {"recording", "processing"}


class TrashError(Exception):
    pass


@dataclass(frozen=True)
class TrashedLecture:
    id: str
    title: str
    deleted_at: str  # ISO 8601, UTC
    size_bytes: int

    @property
    def expires_at(self) -> str:
        return (datetime.fromisoformat(self.deleted_at) + timedelta(days=TRASH_DAYS)).isoformat()


def trash_dir(store: SessionStore) -> Path:
    return store.data_dir / "trash"


def _folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _move(source: Path, destination: Path) -> None:
    """Rename a folder; Windows may briefly deny it while a scanner reads a file."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if attempt == 5:
                raise TrashError(
                    "a file in this lecture is in use; stop playback and any job using it, "
                    f"then try again ({error})"
                ) from error
            time.sleep(0.05 * 2**attempt)


def trash_lecture(store: SessionStore, session_id: str) -> TrashedLecture:
    """Move a lecture to the Trash; refused while it is being transcribed."""
    folder = store.session_dir(session_id)
    session = store.load_session(session_id)
    if session.status in BUSY_STATUSES:
        raise TrashError("this lecture is still being transcribed; pause it first")
    trash = trash_dir(store)
    trash.mkdir(parents=True, exist_ok=True)
    destination = safe_child(trash, session_id)
    if destination.exists():
        raise TrashError("a lecture with this id is already in the Trash")
    entry = TrashedLecture(
        session_id, session.title, datetime.now(UTC).isoformat(), _folder_size(folder)
    )
    record = trash / f"{session_id}.json"
    # The record is written first, so a moved folder always has one.
    atomic_json(record, {"schema_version": 1, **entry.__dict__})
    try:
        _move(folder, destination)
    except BaseException:
        record.unlink(missing_ok=True)
        raise
    return entry


def list_trash(store: SessionStore) -> list[TrashedLecture]:
    """Trashed lectures, most recently deleted first."""
    trash = trash_dir(store)
    if not trash.is_dir():
        return []
    entries = []
    for folder in trash.iterdir():
        if not folder.is_dir():
            continue
        try:
            value = json.loads((trash / f"{folder.name}.json").read_text(encoding="utf-8"))
            entries.append(
                TrashedLecture(
                    folder.name,
                    str(value.get("title", folder.name)),
                    str(value["deleted_at"]),
                    int(value.get("size_bytes", 0)),
                )
            )
        except OSError, ValueError, KeyError, TypeError:
            # No readable record: date it by the folder so it still expires.
            deleted = datetime.fromtimestamp(folder.stat().st_mtime, UTC).isoformat()
            entries.append(TrashedLecture(folder.name, folder.name, deleted, _folder_size(folder)))
    return sorted(entries, key=lambda e: e.deleted_at, reverse=True)


def restore_lecture(store: SessionStore, session_id: str) -> None:
    trash = trash_dir(store)
    source = safe_child(trash, session_id)
    if not source.is_dir():
        raise TrashError("this lecture is no longer in the Trash")
    store.lectures.mkdir(parents=True, exist_ok=True)
    destination = safe_child(store.lectures, session_id)
    if destination.exists():
        raise TrashError("a lecture with the same id is already in the library")
    _move(source, destination)
    (trash / f"{session_id}.json").unlink(missing_ok=True)


def _remove_readonly(function: object, path: str, _error: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def empty_trash(store: SessionStore, older_than_days: int | None = TRASH_DAYS) -> int:
    """Permanently remove trashed lectures (all of them when older_than_days is None).

    Returns how many were removed; one that cannot be removed now stays for next time.
    """
    now = datetime.now(UTC)
    removed = 0
    for entry in list_trash(store):
        age = now - datetime.fromisoformat(entry.deleted_at)
        if older_than_days is not None and age < timedelta(days=older_than_days):
            continue
        folder = safe_child(trash_dir(store), entry.id)
        try:
            shutil.rmtree(folder, onexc=_remove_readonly)
        except OSError:
            continue
        (trash_dir(store) / f"{entry.id}.json").unlink(missing_ok=True)
        removed += 1
    return removed
