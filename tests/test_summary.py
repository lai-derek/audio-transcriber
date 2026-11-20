"""The written summary covers the whole session, never invents numbers silently, and is safe."""

import pytest

from audio_transcriber.ai_cleanup import DEFAULT_MODEL, CleanupCancelled
from audio_transcriber.cli import main
from audio_transcriber.engines import MockSpeechEngine
from audio_transcriber.export import ExportError, write_export
from audio_transcriber.models import InferenceProvenance, Segment
from audio_transcriber.pipeline import BatchOptions, BatchRunner
from audio_transcriber.storage import SessionStore, atomic_json, sha256_file
from audio_transcriber.summary import PART_CHARS, summarize_session


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


def write_ai(store, session_id, text):
    atomic_json(
        store.session_dir(session_id) / "ai-transcript.json",
        {
            "provenance": InferenceProvenance("fixture", "fixture", "CPU", "CPU").__dict__,
            "segments": [Segment(0, 3, text).__dict__],
        },
    )


class Notes:
    device = "GPU"

    def __init__(self):
        self.calls = []

    def generate(self, text, mode, style):
        assert mode == "summary"
        self.calls.append((style, text))
        return "## Overview\n\nA synthetic lecture." if style == "combine" else "- " + text[:40]


def test_cli_summary_is_separate_markdown_and_not_subtitles(ready, monkeypatch):
    store, session = ready
    write_ai(store, session.id, "The pump moves 5 liters per minute.")
    directory = store.session_dir(session.id)
    hashes = {layer: sha256_file(directory / f"{layer}-transcript.json") for layer in ("raw", "ai")}
    model = Notes()
    monkeypatch.setattr("audio_transcriber.cli.make_model", lambda *args: model)
    assert main(["--data-dir", str(store.data_dir), "summarize", session.id]) == 0
    summary = store.load_transcript(session.id, "summary")
    assert summary.text == "## Overview\n\nA synthetic lecture."
    assert [style for style, _ in model.calls] == ["part", "combine"]
    assert summary.transformation["source_layer"] == "ai"
    assert summary.transformation["prompt_version"] == "summary-v6"
    assert summary.transformation["style"] == "structured"
    assert summary.provenance.actual_device == "GPU"
    assert len(list((directory / "summary-history").glob("*.json"))) == 1
    model.calls.clear()
    common = ["--data-dir", str(store.data_dir), "summarize", session.id]
    assert main([*common, "--style", "prose", "--save-only"]) == 0
    # Part notes stay outlines; only the final merge uses the chosen layout.
    assert [style for style, _ in model.calls] == ["part", "combine-prose"]
    assert store.load_transcript(session.id, "summary").text == summary.text  # Not replaced.
    assert len(list((directory / "summary-history").glob("*.json"))) == 2  # Kept in history.
    assert all(
        sha256_file(directory / f"{layer}-transcript.json") == digest
        for layer, digest in hashes.items()
    )
    assert "## Overview" in write_export(store, session.id, "summary", "markdown").read_text()
    with pytest.raises(ExportError, match="not subtitles"):
        write_export(store, session.id, "summary", "srt")


def test_long_sessions_are_summarized_in_bounded_parts_then_merged(ready):
    store, session = ready
    write_ai(store, session.id, " ".join(f"Idea {i} matters." for i in range(2000)))

    class Echo:
        """Merged notes as long as their inputs, the worst case for shrinking."""

        def __init__(self):
            self.calls = []

        def generate(self, text, mode, style):
            self.calls.append((style, len(text)))
            return text[: PART_CHARS // 2] if style == "combine" else text[:3000]

    model = Echo()
    progress = []
    summarize_session(
        store,
        session.id,
        model,
        DEFAULT_MODEL,
        progress=lambda done, total: progress.append((done, total)),
    )
    parts = [size for style, size in model.calls if style == "part"]
    assert len(parts) > 1 and all(size <= PART_CHARS for size in parts)
    assert model.calls[-1][0] == "combine"
    assert progress[-1][0] == progress[-1][1] == len(model.calls)
    assert len(store.load_transcript(session.id, "summary").segments) == 1


def test_parts_are_sent_together_and_their_time_is_recorded(ready):
    import time

    store, session = ready
    write_ai(store, session.id, " ".join(f"Idea {i} matters." for i in range(2000)))

    class Batching:
        device = "GPU"

        def __init__(self):
            self.batches = []

        def generate_many(self, texts, mode, style, progress=None):
            self.batches.append((style, len(texts)))
            time.sleep(0.05)  # The time is spent here, not in generate().

        def generate(self, text, mode, style):
            return "## Overview\n\nMerged." if style.startswith("combine") else "- " + text[:40]

    model = Batching()
    summarize_session(store, session.id, model, DEFAULT_MODEL)
    assert model.batches[0][0] == "part" and model.batches[0][1] > 1
    summary = store.load_transcript(session.id, "summary")
    assert summary.provenance.inference_seconds >= 0.1


def test_invented_numbers_are_flagged_for_review(ready):
    store, session = ready
    write_ai(store, session.id, "The pump moves 5 liters per minute.")

    class Inventor:
        def generate(self, text, mode, style):
            return "- The pump moves 50 liters per minute."

    summarize_session(store, session.id, Inventor(), DEFAULT_MODEL)
    summary = store.load_transcript(session.id, "summary")
    assert summary.transformation["warnings"] == ["new-number"]
    assert summary.segments[0].uncertain


def test_summary_uses_raw_before_cleanup_exists(ready):
    store, session = ready
    model = Notes()
    summarize_session(store, session.id, model, DEFAULT_MODEL)
    summary = store.load_transcript(session.id, "summary")
    assert summary.transformation["source_layer"] == "raw"
    assert "synthetic transcript" in model.calls[0][1]


def test_cancellation_keeps_the_previous_summary(ready):
    store, session = ready
    target = summarize_session(store, session.id, Notes(), DEFAULT_MODEL)
    previous = target.read_bytes()

    class Cancel:
        def generate(self, *args):
            raise CleanupCancelled("stopped")

    with pytest.raises(CleanupCancelled):
        summarize_session(store, session.id, Cancel(), DEFAULT_MODEL)
    assert target.read_bytes() == previous


def test_dictation_does_not_summarize_by_default(ready, monkeypatch):
    store, session = ready
    model = Notes()
    model.generate = lambda text, mode, style: (model.calls.append(mode), text)[1]
    monkeypatch.setattr("audio_transcriber.cli.make_model", lambda *args: model)
    common = ["--data-dir", str(store.data_dir), "cleanup", session.id]
    assert main([*common, "--mode", "dictation"]) == 0
    assert model.calls == ["dictation"]
    assert main([*common, "--mode", "dictation", "--summary", "on"]) == 0
    assert model.calls[1:] == ["dictation", "summary", "summary"]
