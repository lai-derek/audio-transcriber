"""Saved results, background (save-only) runs, and the user's own copies."""

import json

import pytest

from audio_transcriber.ai_cleanup import DEFAULT_MODEL, CleanupError, cleanup_session
from audio_transcriber.cli import main
from audio_transcriber.engines import MockSpeechEngine
from audio_transcriber.export import ExportError, write_export
from audio_transcriber.formatting import format_session
from audio_transcriber.pipeline import BatchOptions, BatchRunner
from audio_transcriber.storage import SessionStore, atomic_json, sha256_file
from audio_transcriber.versions import (
    UserTextConflict,
    discard_user_text,
    load_user_text,
    save_user_text,
    saved_results,
    use_saved,
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


class Fixture:
    device = "CPU"

    def generate(self, text, mode, style):
        if mode == "format":
            return f"- {style}: {text}"
        return f"{style}: {text.capitalize()}."


def clean(store, session, style, publish=True):
    return cleanup_session(
        store, session.id, Fixture(), DEFAULT_MODEL, "CPU", "lecture", style, publish=publish
    )


def revision(path):
    return json.loads(path.read_text(encoding="utf-8"))["transformation"]["revision_id"]


def test_save_only_keeps_the_current_version_and_can_be_used_later(ready):
    store, session = ready
    current = clean(store, session, "medium")
    before = sha256_file(current)
    heavy = revision(clean(store, session, "heavy", publish=False))
    assert sha256_file(current) == before  # The background run replaced nothing.
    results = {r.style: r for r in saved_results(store, session.id, "ai")}
    assert results["medium"].is_current and not results["heavy"].is_current
    assert all(r.current_prompt and r.source_available for r in results.values())
    use_saved(store, session.id, "ai", heavy)
    assert store.load_transcript(session.id, "ai").text.startswith("heavy:")
    assert {r.style: r.is_current for r in saved_results(store, session.id, "ai")} == {
        "medium": False,
        "heavy": True,
    }


def test_notes_from_a_saved_cleanup_record_which_cleanup_they_came_from(ready):
    store, session = ready
    medium = revision(clean(store, session, "medium"))
    heavy = revision(clean(store, session, "heavy", publish=False))
    format_session(store, session.id, "mixed", "ai", Fixture(), DEFAULT_MODEL)
    path = format_session(
        store,
        session.id,
        "mixed",
        "ai",
        Fixture(),
        DEFAULT_MODEL,
        publish=False,
        source_revision=heavy,
    )
    assert "heavy:" in json.loads(path.read_text(encoding="utf-8"))["segments"][0]["text"]
    sources = {r.source_revision_id for r in saved_results(store, session.id, "formatted")}
    assert sources == {medium, heavy}
    with pytest.raises(CleanupError):
        format_session(store, session.id, "mixed", "raw", Fixture(), source_revision=heavy)


def test_cli_save_only_chain_formats_and_summarizes_the_new_cleanup(ready, monkeypatch, capsys):
    store, session = ready
    monkeypatch.setattr("audio_transcriber.cli.make_model", lambda *a: Fixture())
    common = ["--data-dir", str(store.data_dir)]
    assert main(common + ["cleanup", session.id, "--style", "medium", "--summary", "off"]) == 0
    current = store.load_transcript(session.id, "ai").text
    args = ["cleanup", session.id, "--style", "heavy", "--formatting", "mixed", "--save-only"]
    assert main(common + args) == 0
    out = capsys.readouterr().out
    saved = dict(
        line.removeprefix("saved_revision: ").split(" ", 1)
        for line in out.splitlines()
        if line.startswith("saved_revision: ")
    )
    assert set(saved) == {"ai", "formatted", "summary"}
    assert store.load_transcript(session.id, "ai").text == current  # Nothing replaced.
    notes = {r.revision_id: r for r in saved_results(store, session.id, "formatted")}
    assert notes[saved["formatted"]].source_revision_id == saved["ai"]
    summaries = {r.revision_id: r for r in saved_results(store, session.id, "summary")}
    assert summaries[saved["summary"]].source_revision_id == saved["ai"]


def test_user_copy_is_kept_beside_the_ai_version_and_exported(ready):
    store, session = ready
    clean(store, session, "medium")
    ai_text = store.load_transcript(session.id, "ai").text
    first = save_user_text(store, session.id, "ai", "My own wording.", None)
    assert (
        first.based_on_revision
        == store.load_transcript(session.id, "ai").transformation["revision_id"]
    )
    with pytest.raises(UserTextConflict):
        save_user_text(store, session.id, "ai", "Stale edit.", None)
    second = save_user_text(store, session.id, "ai", "My own wording, again.", first.revision_id)
    assert load_user_text(store, session.id, "ai") == second
    assert store.load_transcript(session.id, "ai").text == ai_text  # AI output untouched.
    text = write_export(store, session.id, "ai", "text").read_text(encoding="utf-8")
    assert text == "My own wording, again.\n"
    exported = json.loads(write_export(store, session.id, "ai", "json").read_text("utf-8"))
    assert exported["edit_provenance"]["author"] == "user"
    with pytest.raises(ExportError, match="no timing"):
        write_export(store, session.id, "ai", "srt")
    discard_user_text(store, session.id, "ai")
    assert load_user_text(store, session.id, "ai") is None
    srt = write_export(store, session.id, "ai", "srt").read_text(encoding="utf-8")
    assert "-->" in srt
    history = store.session_dir(session.id) / "ai-user-history"
    assert len(list(history.glob("*.json"))) == 2  # Discarded copies stay recoverable.


@pytest.mark.parametrize("layer", ["ai", "formatted", "summary"])
def test_export_uses_edits_only_for_the_selected_version(ready, layer):
    store, session = ready
    source = clean(store, session, "medium")
    path = store.session_dir(session.id) / f"{layer}-transcript.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    atomic_json(path, original)
    copy = save_user_text(store, session.id, layer, "My medium-version edits.", None)
    assert write_export(store, session.id, layer, "text").read_text("utf-8") == copy.text + "\n"

    other = json.loads(source.read_text(encoding="utf-8"))
    other["transformation"]["revision_id"] = "another-version"
    other["segments"][0]["text"] = "The newly selected version."
    atomic_json(path, other)
    exported = write_export(store, session.id, layer, "text", overwrite=True).read_text("utf-8")
    assert exported.strip() == store.load_transcript(session.id, layer).text
    assert copy.text not in exported
    # Switching back restores the edited copy, which was never deleted.
    atomic_json(path, original)
    assert load_user_text(store, session.id, layer) == copy
    assert (
        write_export(store, session.id, layer, "text", overwrite=True).read_text("utf-8")
        == copy.text + "\n"
    )
