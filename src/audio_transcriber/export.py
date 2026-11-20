"""Transcript exports: versioned JSON, Markdown, plain text, and SRT."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, Transcript
from .storage import atomic_write

FORMATS = {"json", "markdown", "text", "srt"}
LAYERS = {"raw", "balanced", "edited", "ai", "summary", "formatted"}


class ExportError(Exception):
    pass


def plain_text(transcript: Transcript) -> str:
    if transcript.transformation:
        return "\n\n".join(s.text.strip() for s in transcript.segments if s.text.strip()) + "\n"
    return transcript.text + "\n"


def markdown(transcript: Transcript, title: str) -> str:
    lines = [f"# {title}", "", f"Layer: {transcript.provenance.model}", ""]
    lines.extend(f"- [{_stamp(s.start)}] {s.text}" for s in transcript.segments)
    return "\n".join(lines) + "\n"


def srt(transcript: Transcript) -> str:
    blocks = [
        f"{i}\n{_srt_stamp(s.start)} --> {_srt_stamp(s.end)}\n"
        + "\n".join(line.strip() for line in s.text.splitlines() if line.strip())
        for i, s in enumerate((s for s in transcript.segments if s.text.strip()), 1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def validate_srt(content: str) -> None:
    """Reject malformed cue numbering or timestamps before anything is published."""
    blocks = [b for b in content.strip().split("\n\n") if b]
    previous_start = -1.0
    for expected_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 2 or lines[0].strip() != str(expected_index):
            raise ExportError(f"SRT cue {expected_index} has invalid sequence numbering")
        timing = lines[1]
        if " --> " not in timing:
            raise ExportError(f"SRT cue {expected_index} has no timing arrow")
        start_text, end_text = timing.split(" --> ", 1)
        start = _parse_srt_stamp(start_text)
        end = _parse_srt_stamp(end_text)
        if start is None or end is None:
            raise ExportError(f"SRT cue {expected_index} has malformed timestamps")
        if end < start or start < previous_start:
            raise ExportError(f"SRT cue {expected_index} is out of order")
        previous_start = start


def structured_json(transcript: Transcript, layer: str) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "created_at": transcript.created_at,
        "provenance": asdict(transcript.provenance),
        "segments": [asdict(segment) for segment in transcript.segments],
    }
    if transcript.transformation:
        value["transformation"] = transcript.transformation
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def render(transcript: Transcript, layer: str, fmt: str, title: str) -> str:
    if layer not in LAYERS:
        raise ExportError(f"unknown transcript layer '{layer}'")
    if layer in ("summary", "formatted") and fmt == "srt":
        raise ExportError(
            "summaries and formatted documents are not subtitles; choose text, markdown, or json"
        )
    if fmt == "json":
        return structured_json(transcript, layer)
    if fmt == "markdown":
        header = (
            f"# {title}\n\n"
            f"- layer: {layer}\n"
            f"- model: {transcript.provenance.model}\n"
            f"- actual device: {transcript.provenance.actual_device}\n"
        )
        if transcript.transformation:
            header += "- timing: source blocks; AI words are not individually aligned\n"
        if transcript.transformation:
            body = "\n\n".join(s.text for s in transcript.segments)
        else:
            body = "\n".join(
                f"- [{_stamp(s.start)}] {s.text}{' (uncertain)' if s.uncertain else ''}"
                for s in transcript.segments
            )
        return header + "\n" + body + "\n"
    if fmt == "text":
        return plain_text(transcript)
    if fmt == "srt":
        content = srt(transcript)
        validate_srt(content)
        return content
    raise ExportError(f"unknown export format '{fmt}'")


def write_export(
    store: Any,
    session_id: str,
    layer: str,
    fmt: str,
    overwrite: bool = False,
    out_path: Path | None = None,
    use_user_copy: bool = True,
) -> Path:
    """Deterministic, atomic export; existing files are never silently replaced.

    When the user has their own copy of an AI version, the export is that copy
    (what the app shows), unless use_user_copy is False.
    """
    user = None
    if use_user_copy and layer in ("ai", "formatted", "summary"):
        from .versions import load_user_text

        user = load_user_text(store, session_id, layer)
        if user is not None:
            current = store.load_transcript(session_id, layer)
            revision_id = (current.transformation or {}).get("revision_id")
            if user.based_on_revision != revision_id:
                user = None
    if user is not None:
        content = render_user_copy(store, session_id, layer, fmt, user)
        return _publish(store, session_id, layer, fmt, content, overwrite, out_path)
    revision = None
    if layer == "edited":
        from .editing import load_revision

        revision = load_revision(store, session_id)
        transcript = revision.transcript
    else:
        transcript = store.load_transcript(session_id, layer)
    session = store.load_session(session_id)
    content = render(transcript, layer, fmt, session.title)
    if revision is not None and fmt == "json":
        payload = json.loads(content)
        payload["edit_provenance"] = {
            "author": "user",
            "operation": "manual-segment-edit",
            "revision": revision.revision,
            "parent": revision.parent,
            "base_layer": revision.base_layer,
            "base_sha256": revision.base_sha256,
            "changed_segment": revision.changed_segment,
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return _publish(store, session_id, layer, fmt, content, overwrite, out_path)


def render_user_copy(store: Any, session_id: str, layer: str, fmt: str, user: Any) -> str:
    """The user's whole-document copy, marked as edited by the user."""
    if fmt == "srt":
        raise ExportError(
            "your edited copy has no timing for subtitles; choose text, markdown, or json"
        )
    if fmt not in FORMATS:
        raise ExportError(f"unknown export format '{fmt}'")
    text = str(user.text).strip() + "\n"
    if fmt == "text":
        return text
    if fmt == "markdown":
        title = str(store.load_session(session_id).title)
        return f"# {title}\n\n- layer: {layer}\n- edited by the user\n\n{text}"
    ai_provenance = None
    try:
        ai_provenance = asdict(store.load_transcript(session_id, layer).provenance)
    except FileNotFoundError:
        pass
    value = {
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "created_at": user.created_at,
        "text": user.text,
        "edit_provenance": {
            "author": "user",
            "operation": "whole-document-edit",
            "revision": user.revision_id,
            "based_on_revision": user.based_on_revision,
            "ai_provenance": ai_provenance,
        },
    }
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _publish(
    store: Any,
    session_id: str,
    layer: str,
    fmt: str,
    content: str,
    overwrite: bool,
    out_path: Path | None,
) -> Path:
    extension = {"markdown": "md"}.get(fmt, fmt)
    destination = out_path or store.exports_dir(session_id) / f"{session_id}.{layer}.{extension}"
    if destination.exists() and not overwrite:
        raise ExportError(f"export already exists: {destination.name}; pass --overwrite to replace")
    atomic_write(destination, content.encode("utf-8"))
    return destination


def _stamp(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02}:{minute:02}:{second:02}"


def _srt_stamp(seconds: float) -> str:
    whole, millis = divmod(round(seconds * 1000), 1000)
    return f"{_stamp(whole)},{millis:03}"


def _parse_srt_stamp(text: str) -> float | None:
    parts = text.strip().split(",")
    if len(parts) != 2 or not parts[1].isdigit() or len(parts[1]) != 3:
        return None
    clock = parts[0].split(":")
    if len(clock) != 3 or not all(c.isdigit() and len(c) == 2 for c in clock):
        return None
    hours, minutes, secs = (int(c) for c in clock)
    return hours * 3600 + minutes * 60 + secs + int(parts[1]) / 1000.0
