"""Formatting condenses the source into notes, marks unsaid numbers, and protects prior versions."""

import pytest

from audio_transcriber.ai_cleanup import DEFAULT_MODEL, CleanupCancelled, CleanupError
from audio_transcriber.cli import main
from audio_transcriber.engines import MockSpeechEngine
from audio_transcriber.export import ExportError, write_export
from audio_transcriber.formatting import NOTES_CHARS, check_notes, format_session, tidy_notes
from audio_transcriber.models import InferenceProvenance, Segment
from audio_transcriber.pipeline import BatchOptions, BatchRunner
from audio_transcriber.storage import SessionStore, atomic_json, sha256_file


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


SOURCE = (
    "so the first pump moves 5 liters per minute and the second one moves 8 but it is "
    "louder and we have not measured the power use of either pump yet"
)


@pytest.mark.parametrize(
    ("notes", "reason"),
    [
        ("", "empty-output"),
        ("- Pumps\n" * 80, "large-addition"),
        ("## Pumps\n\n- First pump: 5 liters per minute\n- Second: 9, louder", "new-number"),
    ],
)
def test_notes_that_fail_checks_are_reported(notes, reason):
    assert check_notes(SOURCE, notes) == [reason]


def test_condensed_notes_may_drop_details_and_reword_numbers():
    # Dropping a number is condensing, not an error.
    assert check_notes(SOURCE, "## Pumps\n\n- Second pump is louder") == []
    source = (
        "We made pieces of five to seven thousand letters, then 24,000, out of three and a "
        "half billion years of evolution, for six and a half billion people."
    )
    notes = "- Pieces: 5,000-7k, then 24k\n- 3.5 billion years\n- 6.5B people"
    assert check_notes(source, notes) == []


def test_only_lines_with_unsaid_numbers_are_marked():
    from audio_transcriber.formatting import FLAG, write_notes

    class Model:
        def generate(self, *args):
            return "## Pumps\n- First moves 5 liters\n- Second moves 9, louder\n- Not measured"

    notes, warnings = write_notes(Model(), "structured", SOURCE, "")
    assert warnings == ["new-number"]
    assert notes.splitlines() == [
        "## Pumps",
        "- First moves 5 liters",
        "- Second moves 9, louder" + FLAG,
        "- Not measured",
    ]


def test_tidy_notes_strips_fences_and_a_repeated_heading():
    assert tidy_notes("```markdown\n## A\n\n- x\n```", "") == "## A\n\n- x"
    previous = "## Intro\n\n- a\n\n## Pumps\n\n- b"
    assert tidy_notes("## Pumps\n\n- more", previous) == "- more"
    assert tidy_notes("## Power\n\n- more", previous).startswith("## Power")
    # The same topic reworded only in case or punctuation is still one section.
    assert tidy_notes("## pumps:\n### Noise\n- loud", previous + "\n### Noise") == "- loud"


def test_outline_bullets_are_normalized_to_two_levels():
    from audio_transcriber.formatting import normalize_outline

    notes = "## A\n* point\n    * detail\n        - deeper\n\t+ tabbed\nplain line"
    assert normalize_outline(notes) == (
        "## A\n- point\n  - detail\n  - deeper\n  - tabbed\nplain line"
    )


def test_rejected_structured_block_keeps_the_outline(ready):
    store, session = ready

    class Bad:
        def generate(self, *args):
            return ""

    format_session(store, session.id, "structured", "raw", Bad(), DEFAULT_MODEL)
    output = store.load_transcript(session.id, "formatted")
    raw = store.load_transcript(session.id, "raw").text
    assert output.text == f"- {raw}"
    assert output.transformation["blocks"][0]["used_source"]


def test_prose_needs_no_model_and_keeps_history_and_source(ready):
    store, session = ready
    directory = store.session_dir(session.id)
    before = {
        layer: sha256_file(directory / f"{layer}-transcript.json") for layer in ("raw", "balanced")
    }
    format_session(store, session.id, source_layer="raw")
    format_session(store, session.id, source_layer="raw")
    output = store.load_transcript(session.id, "formatted")
    assert output.provenance.actual_device == "none"
    assert output.transformation["source_layer"] == "raw"
    assert output.text == store.load_transcript(session.id, "raw").text
    assert len(list((directory / "formatting-history").glob("*.json"))) == 2
    assert all(
        sha256_file(directory / f"{layer}-transcript.json") == digest
        for layer, digest in before.items()
    )
    for fmt in ("text", "markdown", "json"):
        assert write_export(store, session.id, "formatted", fmt).is_file()
    with pytest.raises(ExportError, match="not subtitles"):
        write_export(store, session.id, "formatted", "srt")


def test_rejected_notes_fall_back_to_complete_source(ready):
    store, session = ready

    class Bad:
        device = "GPU"

        def generate(self, *args):
            return ""

    format_session(store, session.id, "mixed", "raw", Bad(), DEFAULT_MODEL)
    output = store.load_transcript(session.id, "formatted")
    assert output.text == store.load_transcript(session.id, "raw").text
    assert output.transformation["blocks"][0]["used_source"]
    assert output.provenance.actual_device == "GPU"


def test_cancellation_failure_and_source_change_do_not_publish(ready):
    store, session = ready
    target = format_session(store, session.id, source_layer="raw")
    before = sha256_file(target)
    with pytest.raises(CleanupCancelled):
        format_session(store, session.id, source_layer="raw", cancel_requested=lambda: True)

    class Broken:
        def generate(self, *args):
            raise RuntimeError("device failed")

    with pytest.raises(RuntimeError):
        format_session(store, session.id, "mixed", "raw", Broken(), DEFAULT_MODEL)

    class ChangesSource:
        def generate(self, *args):
            source = store.session_dir(session.id) / "raw-transcript.json"
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return "- synthetic transcript"

    with pytest.raises(CleanupError, match="source changed"):
        format_session(store, session.id, "mixed", "raw", ChangesSource(), DEFAULT_MODEL)
    assert sha256_file(target) == before
    assert len(list((target.parent / "formatting-history").glob("*.json"))) == 1


def test_format_reads_ai_not_raw_and_records_source_hash(ready):
    store, session = ready
    directory = store.session_dir(session.id)
    source = {
        "provenance": InferenceProvenance("fixture", "fixture", "CPU", "CPU").__dict__,
        "segments": [Segment(0, 3, "Do not change the 5 volt supply.").__dict__],
    }
    atomic_json(directory / "ai-transcript.json", source)

    class Notes:
        device = "CPU"

        def generate(self, text, mode, style):
            assert "5 volt" in text and mode == "format" and style == "structured"
            return "- Do not change the 5 volt supply."

    format_session(store, session.id, "structured", "ai", Notes(), DEFAULT_MODEL)
    output = store.load_transcript(session.id, "formatted")
    assert output.text == "- Do not change the 5 volt supply."
    assert output.transformation["source_sha256"] == sha256_file(directory / "ai-transcript.json")


def test_cli_cleanup_reuses_one_model_for_notes_and_summary(ready, monkeypatch):
    store, session = ready
    common = ["--data-dir", str(store.data_dir)]
    assert main(common + ["format", session.id, "--source", "raw"]) == 0
    calls = []

    class Shared:
        device = "GPU"

        def generate(self, text, mode, style):
            calls.append(mode)
            return "- " + text if mode == "format" else text

    made = []

    def factory(*args):
        made.append(1)
        return Shared()

    monkeypatch.setattr("audio_transcriber.cli.make_model", factory)
    assert main(common + ["cleanup", session.id, "--formatting", "structured"]) == 0
    # One model load serves cleanup, notes, and the automatic lecture summary.
    assert made == [1] and calls == ["lecture", "format", "summary", "summary"]
    assert store.load_transcript(session.id, "formatted").transformation["source_layer"] == "ai"
    assert store.load_transcript(session.id, "summary").transformation["source_layer"] == "ai"
    calls.clear()
    assert main(common + ["cleanup", session.id, "--summary", "off"]) == 0
    assert calls == ["lecture"]


def test_transcribe_off_cleanup_can_format_without_text_model(
    tmp_path, three_second_wav, monkeypatch
):
    def forbidden(*args):
        raise AssertionError("prose must not load a text model")

    monkeypatch.setattr("audio_transcriber.cli.make_model", forbidden)
    data = tmp_path / "library"
    assert (
        main(
            [
                "--data-dir",
                str(data),
                "transcribe",
                "--cleanup",
                "off",
                str(three_second_wav),
                "--model",
                "mock",
                "--formatting",
                "prose",
            ]
        )
        == 0
    )
    store = SessionStore(data)
    session = next(store.lectures.iterdir()).name
    assert store.load_transcript(session, "formatted").transformation["source_layer"] == "raw"


def test_long_sources_are_split_into_bounded_requests(ready):
    store, session = ready
    text = " ".join(f"Passage {i} must remain." for i in range(300))
    path = store.session_dir(session.id) / "ai-transcript.json"
    atomic_json(
        path,
        {
            "provenance": InferenceProvenance("fixture", "fixture", "CPU", "CPU").__dict__,
            "segments": [Segment(0, 3, text).__dict__],
        },
    )
    requests = []

    class Echo:
        def generate(self, request, mode, style):
            requests.append(request)
            return "- " + request

    format_session(store, session.id, "mixed", "ai", Echo(), DEFAULT_MODEL)
    output = store.load_transcript(session.id, "formatted")
    assert len(requests) > 1 and all(len(r) <= NOTES_CHARS for r in requests)
    assert not any(record["used_source"] for record in output.transformation["blocks"])
    # Blocks may split a sentence; together they still carry the whole source once.
    assert " ".join(output.text.replace("- ", "").split()) == text


def test_invented_abbreviation_expansions_are_removed():
    from audio_transcriber.formatting import strip_invented_expansions

    source = "She is now the CTO and the team joins ACM, the Association for Computing Machinery."
    notes = (
        "- She is the CTO (Chief Technology Officer)\n"
        "- Team joins ACM (Association for Computing Machinery)"
    )
    assert strip_invented_expansions(notes, source) == (
        "- She is the CTO\n- Team joins ACM (Association for Computing Machinery)"
    )


def test_runaway_notes_fall_back_to_source_without_failing_the_job(ready):
    from audio_transcriber.ai_cleanup import RunawayOutput

    store, session = ready

    class Runaway:
        def generate(self, *args):
            raise RunawayOutput("too long")

    format_session(store, session.id, "mixed", "raw", Runaway(), DEFAULT_MODEL)
    output = store.load_transcript(session.id, "formatted")
    assert output.text == store.load_transcript(session.id, "raw").text
    assert output.transformation["blocks"][0]["warnings"] == ["runaway-output"]
