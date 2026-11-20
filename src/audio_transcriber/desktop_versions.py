"""Choosing saved AI results and planning background runs for the lecture window.

Versions the user might pick next are prepared in the background for imported
lectures, so picking one later shows a saved result instead of waiting for the
model. By default these are the other note and summary styles of the cleanup
shown; Settings can add the other cleanup levels or leave styles out. Lookups go
through `versions.saved_results`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .ai_cleanup import STYLES as LEVELS
from .storage import SessionStore
from .versions import SavedResult, saved_results

FORMATS = ("prose", "mixed", "structured")
# Background order: the default style first, then the others (user's order).
BACKGROUND_STYLES = ("mixed", "prose", "structured")
# What background preparation makes, and the default for each (Settings menu).
PREPARE = {"notes": True, "summaries": True, "levels": False}


def current_info(store: SessionStore, session_id: str, layer: str) -> dict[str, Any]:
    """The transformation record of the version shown for `layer` (empty if none)."""
    path = store.session_dir(session_id) / f"{layer}-transcript.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("transformation")
    except OSError, ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def usable(result: SavedResult) -> bool:
    return result.current_prompt and result.source_available


def find_cleanup(results: list[SavedResult], level: str, mode: str) -> SavedResult | None:
    """The newest usable cleanup at `level`; `results` comes from saved_results(..., "ai")."""
    return next((r for r in results if r.style == level and r.mode == mode and usable(r)), None)


def find_derived(
    results: list[SavedResult], ai_revision: str | None, style: str | None = None
) -> SavedResult | None:
    """The newest usable notes (or summary) made from that cleanup, in `style` if given.
    Without a cleanup, notes made directly from the original transcript."""
    for result in results:
        # Summaries made before styles existed were outlines (structured).
        if style is not None and (result.style or "structured") != style or not usable(result):
            continue
        if ai_revision is not None and result.source_revision_id == ai_revision:
            return result
        if ai_revision is None and result.source_layer in ("raw", "balanced"):
            return result
    return None


def made_after(result: SavedResult, since: datetime | None) -> bool:
    """Whether a result is newer than `since` (a Redo); history mixes UTC and local times."""
    if since is None:
        return True
    try:
        made = datetime.fromisoformat(result.created_at)
    except ValueError:
        return False
    if made.tzinfo is None:
        made = made.replace(tzinfo=UTC)
    return made >= since


def background_plan(
    store: SessionStore,
    session_id: str,
    model: str,
    default_format: str,
    since: datetime | None = None,
    notes: bool = True,
    summaries: bool = True,
    levels: bool = True,
) -> list[list[str]]:
    """Worker arguments for the versions not saved yet, most likely picks first: the
    current cleanup's note styles and summary styles, then (with `levels`) the other
    cleanup levels and their styles. With `since` (after Redo), older results count
    as missing."""
    current = current_info(store, session_id, "ai")
    revision, mode = current.get("revision_id"), current.get("mode")
    if not revision or not mode:
        return []  # Without a cleanup there is nothing to vary.
    want_notes, want_summaries = notes, summaries
    cleanups = [r for r in saved_results(store, session_id, "ai") if made_after(r, since)]
    made_notes = [r for r in saved_results(store, session_id, "formatted") if made_after(r, since)]
    made_summaries = [
        r for r in saved_results(store, session_id, "summary") if made_after(r, since)
    ]

    def derived_from(ai_revision: str) -> list[list[str]]:
        # Each style's notes, then its summary, in the background order.
        jobs: list[list[str]] = []
        for style in BACKGROUND_STYLES:
            if want_notes and find_derived(made_notes, ai_revision, style) is None:
                jobs.append(
                    [
                        "format",
                        session_id,
                        "--source",
                        "ai",
                        "--style",
                        style,
                        "--source-revision",
                        ai_revision,
                        "--model",
                        model,
                        "--save-only",
                    ]
                )
            if want_summaries and find_derived(made_summaries, ai_revision, style) is None:
                jobs.append(
                    [
                        "summarize",
                        session_id,
                        "--style",
                        style,
                        "--source-revision",
                        ai_revision,
                        "--model",
                        model,
                        "--save-only",
                    ]
                )
        return jobs

    jobs = derived_from(str(revision))
    for level in LEVELS if levels else ():
        if level == current.get("style"):
            continue
        saved = find_cleanup(cleanups, level, mode)
        if saved is None:
            # Its note styles and summary are planned once this cleanup is saved.
            jobs.append(
                [
                    "cleanup",
                    session_id,
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--style",
                    level,
                    "--formatting",
                    default_format,
                    "--summary",
                    "off",
                    "--save-only",
                ]
            )
        else:
            jobs += derived_from(saved.revision_id)
    return jobs
