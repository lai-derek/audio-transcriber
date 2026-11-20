"""Saved AI results and the user's own copies of each AI version.

Every cleanup, formatting, and summary run is kept in its history folder. A run
can be saved without replacing the version on screen (background runs), and any
saved result can be made current again without rerunning the model.

The user's own copy of a version (typed into the app) is stored beside it and
never overwrites the AI output, so discarding the copy restores the AI version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ai_cleanup import CleanupError
from .models import InferenceProvenance, Segment, Transcript, utc_now
from .storage import SAFE_ID, SessionStore, atomic_json, sha256_file

HISTORY = {
    "ai": "ai-cleanup-history",
    "formatted": "formatting-history",
    "summary": "summary-history",
}
USER_LAYERS = tuple(HISTORY)


@dataclass(frozen=True)
class SavedResult:
    layer: str
    revision_id: str
    style: str | None
    mode: str | None
    source_layer: str
    source_sha256: str
    source_revision_id: str | None  # The AI cleanup a format or summary was made from.
    prompt_version: str
    created_at: str
    is_current: bool  # It is the version published as <layer>-transcript.json.
    current_prompt: bool  # Made with today's instructions for its kind.
    source_available: bool  # Its source still exists unchanged, so it can be used.


@dataclass(frozen=True)
class UserText:
    layer: str
    text: str
    revision_id: str
    based_on_revision: str | None  # The AI version shown when the user edited.
    created_at: str


class UserTextConflict(CleanupError):
    """The copy changed since the caller loaded it; reload before saving."""


def _check_layer(layer: str) -> None:
    if layer not in HISTORY:
        raise CleanupError("saved versions exist only for ai, formatted, and summary")


def _check_id(revision_id: str) -> None:
    if not SAFE_ID.fullmatch(revision_id):
        raise CleanupError("invalid revision id")


def history_dir(store: SessionStore, session_id: str, layer: str) -> Path:
    _check_layer(layer)
    return store.session_dir(session_id) / HISTORY[layer]


def save_result(
    store: SessionStore, session_id: str, layer: str, payload: dict[str, Any], publish: bool
) -> Path:
    """Keep a finished result in history; publish it as current unless saving only."""
    history = history_dir(store, session_id, layer)
    history.mkdir(exist_ok=True)
    saved = history / f"{payload['transformation']['revision_id']}.json"
    atomic_json(saved, payload)
    if not publish:
        return saved
    target = store.session_dir(session_id) / f"{layer}-transcript.json"
    atomic_json(target, payload)
    return target


def saved_path(store: SessionStore, session_id: str, layer: str, revision_id: str) -> Path:
    _check_id(revision_id)
    path = history_dir(store, session_id, layer) / f"{revision_id}.json"
    if not path.is_file():
        raise CleanupError(f"no saved {layer} result {revision_id}")
    return path


def read_saved(path: Path) -> Transcript:
    value = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(
        tuple(Segment(**s) for s in value["segments"]),
        InferenceProvenance(**value["provenance"]),
        str(value.get("created_at", "")),
        value.get("transformation"),
    )


def _prompt_versions() -> dict[str, str]:
    from .ai_cleanup import PROMPT_VERSION as CLEANUP
    from .formatting import PROMPT_VERSION as FORMAT
    from .summary import PROMPT_VERSION as SUMMARY

    return {"ai": CLEANUP, "formatted": FORMAT, "summary": SUMMARY}


def saved_results(store: SessionStore, session_id: str, layer: str) -> list[SavedResult]:
    """Every kept result for one layer, newest first."""
    directory = store.session_dir(session_id)
    history = history_dir(store, session_id, layer)
    if not history.is_dir():
        return []
    current = directory / f"{layer}-transcript.json"
    current_hash = sha256_file(current) if current.is_file() else None
    prompt = _prompt_versions()[layer]
    # Results made before source_revision_id was recorded: match the AI file by hash.
    ai_by_hash: dict[str, str] = {}
    if layer != "ai":
        ai_history = directory / HISTORY["ai"]
        if ai_history.is_dir():
            ai_by_hash = {sha256_file(p): p.stem for p in ai_history.glob("*.json")}
    source_hashes: dict[str, str | None] = {}
    results = []
    for path in history.glob("*.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))["transformation"]
        except OSError, ValueError, KeyError, TypeError:
            continue  # An unreadable record is skipped, never fatal.
        source_layer = str(info.get("source_layer", "raw"))
        source_hash = str(info.get("source_sha256", ""))
        source_revision = info.get("source_revision_id")
        if source_layer == "ai" and not source_revision:
            source_revision = ai_by_hash.get(source_hash)
        if source_layer == "ai" and source_revision:
            # A saved cleanup stays usable while its history entry exists.
            available = (directory / HISTORY["ai"] / f"{source_revision}.json").is_file()
        else:
            if source_layer not in source_hashes:
                source = directory / f"{source_layer}-transcript.json"
                source_hashes[source_layer] = sha256_file(source) if source.is_file() else None
            available = source_hashes[source_layer] == source_hash
        results.append(
            SavedResult(
                layer=layer,
                revision_id=path.stem,
                style=info.get("style"),
                mode=info.get("mode"),
                source_layer=source_layer,
                source_sha256=source_hash,
                source_revision_id=source_revision,
                prompt_version=str(info.get("prompt_version", "")),
                created_at=str(info.get("created_at", "")),
                is_current=current_hash is not None and sha256_file(path) == current_hash,
                current_prompt=info.get("prompt_version") == prompt,
                source_available=available,
            )
        )
    return sorted(results, key=lambda r: r.created_at, reverse=True)


def use_saved(store: SessionStore, session_id: str, layer: str, revision_id: str) -> Path:
    """Make a saved result the current version without rerunning the model."""
    saved = saved_path(store, session_id, layer, revision_id)
    match = next(
        (r for r in saved_results(store, session_id, layer) if r.revision_id == revision_id), None
    )
    if match is None or not match.source_available:
        raise CleanupError(
            f"the transcript this {layer} result was made from has changed; run it again"
        )
    target = store.session_dir(session_id) / f"{layer}-transcript.json"
    atomic_json(target, json.loads(saved.read_text(encoding="utf-8")))
    return target


# -- the user's own copy of an AI version ------------------------------------------


def _user_path(store: SessionStore, session_id: str, layer: str) -> Path:
    if layer not in USER_LAYERS:
        raise CleanupError("user copies are kept for ai, formatted, and summary")
    return store.session_dir(session_id) / f"{layer}-user.json"


def load_user_text(store: SessionStore, session_id: str, layer: str) -> UserText | None:
    path = _user_path(store, session_id, layer)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return UserText(
        layer,
        str(value["text"]),
        str(value["revision_id"]),
        value.get("based_on_revision"),
        str(value.get("created_at", "")),
    )


def save_user_text(
    store: SessionStore,
    session_id: str,
    layer: str,
    text: str,
    expected_revision: str | None,
) -> UserText:
    """Save the user's whole-document copy; `expected_revision` is the copy they loaded."""
    import uuid

    path = _user_path(store, session_id, layer)
    existing = load_user_text(store, session_id, layer)
    if (existing.revision_id if existing else None) != expected_revision:
        raise UserTextConflict("this version was changed elsewhere; reload it before saving")
    current = store.session_dir(session_id) / f"{layer}-transcript.json"
    based_on = None
    if current.is_file():
        info = json.loads(current.read_text(encoding="utf-8")).get("transformation") or {}
        based_on = info.get("revision_id")
    copy = UserText(layer, text, uuid.uuid4().hex, based_on, utc_now())
    value = {
        "schema_version": 1,
        "layer": layer,
        "text": copy.text,
        "revision_id": copy.revision_id,
        "parent": expected_revision,
        "based_on_revision": copy.based_on_revision,
        "created_at": copy.created_at,
        "author": "user",
    }
    history = store.session_dir(session_id) / f"{layer}-user-history"
    history.mkdir(exist_ok=True)
    atomic_json(history / f"{copy.revision_id}.json", value)
    atomic_json(path, value)
    return copy


def discard_user_text(store: SessionStore, session_id: str, layer: str) -> None:
    """Show the AI version again; the discarded copy stays in the user history."""
    _user_path(store, session_id, layer).unlink(missing_ok=True)
