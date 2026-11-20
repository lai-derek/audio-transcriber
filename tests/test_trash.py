"""Deleted lectures go to a Trash, can be restored, and expire after 30 days."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from audio_transcriber.engines import MockSpeechEngine
from audio_transcriber.pipeline import BatchOptions, BatchRunner
from audio_transcriber.storage import SessionStore
from audio_transcriber.trash import (
    TrashError,
    empty_trash,
    list_trash,
    restore_lecture,
    trash_dir,
    trash_lecture,
)


@pytest.fixture
def ready(tmp_path, three_second_wav):
    store = SessionStore(tmp_path / "library")
    session = BatchRunner(
        store,
        lambda _: MockSpeechEngine(),
        [],
        BatchOptions(model_id="mock", requested_device="CPU"),
    ).run_new(three_second_wav)
    return store, session


def test_trash_and_restore_keep_the_lecture_intact(ready):
    store, session = ready
    files = sorted(p.name for p in store.session_dir(session.id).rglob("*"))
    entry = trash_lecture(store, session.id)
    assert entry.size_bytes > 0 and entry.title == session.title
    assert not (store.lectures / session.id).exists()
    assert [e.id for e in list_trash(store)] == [session.id]
    restore_lecture(store, session.id)
    assert sorted(p.name for p in store.session_dir(session.id).rglob("*")) == files
    assert list_trash(store) == []
    with pytest.raises(TrashError):
        restore_lecture(store, session.id)


def test_busy_lectures_are_refused(ready):
    store, session = ready
    session.status = "processing"
    store.save_session(session)
    with pytest.raises(TrashError, match="transcribed"):
        trash_lecture(store, session.id)
    assert store.session_dir(session.id).is_dir()


def test_only_expired_entries_are_emptied_unless_asked(ready, tmp_path, three_second_wav):
    store, old = ready
    recent = BatchRunner(
        store,
        lambda _: MockSpeechEngine(),
        [],
        BatchOptions(model_id="mock", requested_device="CPU"),
    ).run_new(three_second_wav)
    trash_lecture(store, old.id)
    trash_lecture(store, recent.id)
    record = trash_dir(store) / f"{old.id}.json"
    value = json.loads(record.read_text(encoding="utf-8"))
    value["deleted_at"] = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    record.write_text(json.dumps(value), encoding="utf-8")
    assert empty_trash(store) == 1
    assert [e.id for e in list_trash(store)] == [recent.id]
    assert empty_trash(store, None) == 1
    assert list_trash(store) == []
    assert list(trash_dir(store).iterdir()) == []
