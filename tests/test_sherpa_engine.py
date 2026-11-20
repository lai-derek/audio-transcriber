"""Parakeet through sherpa-onnx: segments, long-input splits, engine choice, chunk defaults."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from audio_transcriber import cli, resident
from audio_transcriber.acquisition import DEFAULT_SPEECH_MODEL, PARAKEET_V2, cpu_only
from audio_transcriber.engines import (
    MockSpeechEngine,
    OpenVINOWhisperEngine,
    SherpaTransducerEngine,
    split_points,
    token_segments,
)

RATE = 16_000
BASE = "whisper-base.en-int4-ov"


def tokens(*words: tuple[str, float, float]) -> tuple[list[str], list[float], list[float]]:
    """(token, start, duration) triples as sherpa-onnx returns them."""
    return [w[0] for w in words], [w[1] for w in words], [w[2] for w in words]


def test_tokens_join_into_words_and_break_after_sentences() -> None:
    found = token_segments(
        *tokens(
            (" I", 0.4, 0.08),
            ("'", 0.48, 0.16),
            ("ve", 0.64, 0.08),
            (" talk", 0.72, 0.16),
            ("ed", 0.88, 0.32),
            (".", 1.2, 0.0),
            (" We", 1.5, 0.2),
            (" are", 1.7, 0.2),
            ("?", 1.9, 0.08),
        ),
        limit=5.0,
        offset=10.0,
    )
    assert [s.text for s in found] == ["I've talked.", "We are?"]
    assert [(s.start, s.end) for s in found] == [
        pytest.approx((10.4, 11.2)),
        pytest.approx((11.5, 11.98)),
    ]


def test_long_pauses_and_long_runs_start_new_segments() -> None:
    pause = token_segments(*tokens((" so", 0.0, 0.2), (" then", 1.1, 0.2)), limit=5.0)
    assert [s.text for s in pause] == ["so", "then"]  # 0.9 s gap > 0.8 s.
    short = token_segments(*tokens((" so", 0.0, 0.2), (" then", 0.9, 0.2)), limit=5.0)
    assert [s.text for s in short] == ["so then"]
    words = [(f" w{i}", i * 0.5, 0.4) for i in range(40)]  # 20 s without punctuation.
    long = token_segments(*tokens(*words), limit=30.0)
    assert len(long) == 2
    assert long[0].end - long[0].start < 12.0
    assert long[1].start == 12.0


def test_times_are_clamped_monotonic_and_empty_output_has_no_segments() -> None:
    found = token_segments(*tokens((" end", 2.9, 0.4), (".", 3.3, 0.1)), limit=3.0)
    assert [(s.start, s.end) for s in found] == [(2.9, 3.0)]
    assert token_segments([], [], [], limit=3.0) == []
    # Zero-length punctuation never moves a segment before the previous one ends.
    overlap = token_segments(*tokens((" a", 0.0, 1.0), (".", 1.0, 0.0), (" b", 0.5, 0.2)), 5.0)
    assert all(b.start >= a.end for a, b in zip(overlap, overlap[1:], strict=False))


def test_split_points_cut_at_the_quietest_frame_within_the_limit() -> None:
    loud = [0.5] * (35 * RATE)
    quiet_at = 25 * RATE
    loud[quiet_at : quiet_at + RATE // 10] = [0.0] * (RATE // 10)
    cuts = split_points(loud, 30 * RATE, 10 * RATE, RATE // 10)
    assert cuts == [quiet_at + RATE // 20]
    assert split_points([0.1] * (30 * RATE), 30 * RATE, 10 * RATE, RATE // 10) == []
    many = split_points([0.2] * (95 * RATE), 30 * RATE, 10 * RATE, RATE // 10)
    pieces = [b - a for a, b in zip([0, *many], [*many, 95 * RATE], strict=True)]
    assert all(0 < piece <= 30 * RATE for piece in pieces)


class FakeRecognizer:
    """Says one sentence per piece, starting 0.5 s into it."""

    built: list[dict] = []

    def __init__(self) -> None:
        self.lengths: list[int] = []

    @classmethod
    def from_transducer(cls, **options):
        cls.built.append(options)
        return cls()

    def create_stream(self):
        stream = types.SimpleNamespace(samples=None)
        stream.accept_waveform = lambda rate, samples: setattr(stream, "samples", samples)
        return stream

    def decode_stream(self, stream) -> None:
        self.lengths.append(len(stream.samples))
        silent = not any(stream.samples)
        stream.result = types.SimpleNamespace(
            text="" if silent else " Hello there.",
            tokens=[] if silent else [" Hello", " there", "."],
            timestamps=[] if silent else [0.5, 0.9, 1.3],
            durations=[] if silent else [0.4, 0.4, 0.0],
        )


@pytest.fixture
def fake_sherpa(monkeypatch):
    FakeRecognizer.built = []
    module = types.SimpleNamespace(OfflineRecognizer=FakeRecognizer)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return FakeRecognizer


def test_engine_runs_on_cpu_and_splits_long_input(tmp_path: Path, fake_sherpa) -> None:
    engine = SherpaTransducerEngine(tmp_path, PARAKEET_V2)
    engine.load("CPU")
    engine.load("CPU")
    assert len(fake_sherpa.built) == 1  # Loaded once, then reused.
    options = fake_sherpa.built[0]
    assert options["num_threads"] == 2 and options["provider"] == "cpu"
    assert options["model_type"] == "nemo_transducer" and options["feature_dim"] == 128
    assert options["encoder"] == str(tmp_path / "encoder.int8.onnx")

    samples = [0.3] * (40 * RATE)
    samples[20 * RATE : 20 * RATE + RATE // 10] = [0.0] * (RATE // 10)
    result = engine.transcribe_samples(samples, "GPU")
    assert engine._recognizer.lengths == [20 * RATE + RATE // 20, 20 * RATE - RATE // 20]
    assert [s.text for s in result.segments] == ["Hello there."] * 2
    assert result.segments[0].start == 0.5
    assert result.segments[1].start == pytest.approx(20.05 + 0.5)
    provenance = result.provenance
    assert (provenance.requested_device, provenance.actual_device) == ("GPU", "CPU")
    assert provenance.fallback_reason == "GPU unsupported by sherpa-onnx"
    assert engine.transcribe_samples([0.0] * RATE).segments == ()
    assert engine.transcribe_samples([0.0] * RATE).provenance.fallback_reason is None


def installed(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "models"
    for model_id in (PARAKEET_V2, BASE):
        (root / model_id).mkdir(parents=True)
    monkeypatch.setattr(cli, "verify_installed", lambda root, spec: root / spec.id)
    return root


def test_engine_is_chosen_by_runtime(tmp_path: Path, monkeypatch) -> None:
    root = installed(tmp_path, monkeypatch)
    parakeet = cli.make_engine_provider(PARAKEET_V2, tmp_path, root)
    assert isinstance(parakeet("GPU"), SherpaTransducerEngine)
    assert isinstance(cli.make_engine_provider(BASE, tmp_path, root)("CPU"), OpenVINOWhisperEngine)
    assert isinstance(cli.make_engine_provider("mock", tmp_path, root)("CPU"), MockSpeechEngine)
    assert cpu_only(PARAKEET_V2) and not cpu_only(BASE) and not cpu_only("mock")
    resident.CACHE = {}
    try:
        # The resident worker keeps one Parakeet engine whatever device was asked for.
        assert parakeet("GPU") is parakeet("CPU")
    finally:
        resident.CACHE = None


def test_parakeet_skips_device_benchmarks(tmp_path: Path, monkeypatch) -> None:
    def unexpected(*args):
        raise AssertionError("Parakeet runs on the CPU; nothing to benchmark")

    monkeypatch.setattr(cli, "benchmark_devices", unexpected)
    for requested in ("auto", "GPU", "NPU", "CPU"):
        assert cli.resolve_device(requested, unexpected, PARAKEET_V2, tmp_path) == ("CPU", [])


def test_resident_preload_uses_cpu_for_parakeet(tmp_path: Path, monkeypatch, capsys) -> None:
    loaded = []

    class Engine:
        def load(self, device: str) -> None:
            loaded.append(device)

    monkeypatch.setattr(
        "audio_transcriber.acquisition.verify_installed", lambda root, spec: root / spec.id
    )
    monkeypatch.setattr(cli, "make_engine_provider", lambda *args: lambda device: Engine())
    monkeypatch.setattr(cli, "benchmark_devices", lambda *args: pytest.fail("benchmarked"))
    resident.CACHE = {}
    try:
        resident.load(
            [{"kind": "speech", "model": PARAKEET_V2, "device": "auto", "data_dir": str(tmp_path)}]
        )
    finally:
        resident.CACHE = None
    assert loaded == ["CPU"]
    assert f"resident_loaded: {PARAKEET_V2} on CPU" in capsys.readouterr().out


def test_chunk_settings_come_from_the_model() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["transcribe", "in.wav"])
    assert args.model == DEFAULT_SPEECH_MODEL == PARAKEET_V2
    assert cli.chunk_settings(args, PARAKEET_V2) == (28.0, "pause-v1")
    assert cli.chunk_settings(args, BASE) == (30.0, "fixed")
    assert cli.chunk_settings(args, "mock") == (30.0, "fixed")
    explicit = parser.parse_args(
        ["transcribe", "in.wav", "--chunk-seconds", "12", "--chunk-strategy", "pause-v2"]
    )
    assert cli.chunk_settings(explicit, PARAKEET_V2) == (12.0, "pause-v2")
    evaluate = parser.parse_args(["evaluate", "manifest.json"])
    assert cli.chunk_settings(evaluate, PARAKEET_V2) == (28.0, "pause-v1")


def test_new_sessions_use_model_chunking_and_resume_keeps_saved(
    tmp_path: Path, three_second_wav: Path, monkeypatch
) -> None:
    from audio_transcriber.storage import SessionStore

    monkeypatch.setattr(cli, "make_engine_provider", lambda *args: lambda d: MockSpeechEngine())
    data = tmp_path / "library"
    stop = tmp_path / "pause"
    stop.touch()
    common = ["--cleanup", "off", "--formatting", "off", "--stop-file", str(stop)]
    assert cli.main(["--data-dir", str(data), "transcribe", str(three_second_wav), *common]) == 130
    session_id = next((data / "lectures").iterdir()).name
    store = SessionStore(data)
    diagnostics = store.load_session(session_id).diagnostics
    assert (diagnostics["chunk_seconds"], diagnostics["chunk_strategy"]) == (28.0, "pause-v1")
    assert diagnostics["requested_device"] == "CPU"
    stop.unlink()
    # A resume asking for other chunking still uses the plan the session was started with.
    resume = ["resume", session_id, *common[:4], "--chunk-seconds", "5"]
    assert cli.main(["--data-dir", str(data), *resume, "--chunk-strategy", "fixed"]) == 0
    session = store.load_session(session_id)
    assert session.status == "ready"
    assert session.diagnostics["chunk_strategy"] == "pause-v1"
