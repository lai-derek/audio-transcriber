"""Windows atomic replacement may briefly conflict with media/sync readers."""

from __future__ import annotations

import pytest

from audio_transcriber import storage


def test_transient_windows_denial_retries_without_losing_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_bytes(b"old")
    replace = storage.os.replace
    calls = []
    delays = []

    def transient(source, destination):
        calls.append(1)
        if len(calls) < 3:
            assert path.read_bytes() == b"old"
            error = PermissionError("temporary Windows sharing denial")
            error.winerror = 5
            raise error
        replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", transient)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    storage.atomic_write(path, b"new")
    assert path.read_bytes() == b"new"
    assert len(calls) == 3
    assert delays == [0.02, 0.04]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("winerror, expected_calls", [(5, 6), (32, 6), (33, 6), (None, 1)])
def test_persistent_denial_is_bounded_and_retains_previous_file(
    tmp_path, monkeypatch, winerror, expected_calls
):
    path = tmp_path / "session.json"
    path.write_bytes(b"old")
    calls = []

    def denied(*args):
        calls.append(1)
        error = PermissionError("persistent denial")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(storage.os, "replace", denied)
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        storage.atomic_write(path, b"new")
    assert len(calls) == expected_calls
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_finished_checkpoint_removal_retries_windows_progress_reader(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b"checkpoint")
    unlink = type(path).unlink
    calls = []
    delays = []

    def transient(target, *, missing_ok=False):
        calls.append(target)
        if len(calls) < 3:
            error = PermissionError("checkpoint is open in the progress reader")
            error.winerror = 32
            raise error
        unlink(target, missing_ok=missing_ok)

    monkeypatch.setattr(type(path), "unlink", transient)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    storage.remove_checkpoint(path)
    assert not path.exists()
    assert calls == [path] * 3
    assert delays == [0.02, 0.04]


@pytest.mark.parametrize("winerror, expected_calls", [(5, 6), (32, 6), (33, 6), (None, 1)])
def test_checkpoint_removal_preserves_persistent_denials(
    tmp_path, monkeypatch, winerror, expected_calls
):
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b"checkpoint")
    calls = []

    def denied(target, *, missing_ok=False):
        calls.append(target)
        error = PermissionError("persistent denial")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(type(path), "unlink", denied)
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        storage.remove_checkpoint(path)
    assert len(calls) == expected_calls
    assert path.read_bytes() == b"checkpoint"
