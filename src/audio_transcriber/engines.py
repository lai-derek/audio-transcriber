"""Speech engines.

`MockSpeechEngine` keeps orchestration deterministic in offline tests.
`OpenVINOWhisperEngine` is a lazy adapter over the pinned OpenVINO GenAI runtime.

Verified against openvino-genai 2026.3.0.0 (installed package introspection and a
live CPU run): `WhisperPipeline.generate(samples, return_timestamps=True)` returns
a `WhisperDecodedResults` with `.texts`, `.chunks` (each with `start_ts`, `end_ts`,
`text`, `token_ids`, seconds relative to the input audio), `.language`, `.words`
(only when word timestamps are enabled at construction), `.scores`, and
`.perf_metrics`. English-only checkpoints reject explicit `language`/`task`
arguments; multilingual checkpoints accept them and are forced to English here.

`SherpaTransducerEngine` runs NeMo transducers such as Parakeet TDT through
sherpa-onnx 1.13.4 on the CPU. Verified with a live run: a decoded stream's
`result` has `.text` (cased and punctuated), `.tokens` (BPE pieces; a leading
space starts a word), and per-token `.timestamps` and `.durations` in seconds
relative to the input.
"""

from __future__ import annotations

import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .media import TARGET_RATE, inspect_pcm_wav
from .models import InferenceProvenance, Segment, Transcript


class SpeechEngine(Protocol):
    def transcribe(self, audio: Path, device: str = "CPU") -> Transcript: ...


class ChunkTranscriber(Protocol):
    def transcribe_samples(self, samples: list[float], device: str = "CPU") -> Transcript: ...


class RewriteEngine(Protocol):
    def rewrite(self, text: str, mode: str) -> str: ...


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def inspect_audio(path: Path) -> AudioInfo:
    rate, channels, frames = inspect_pcm_wav(path)
    return AudioInfo(rate, channels, frames)


class MockSpeechEngine:
    """Deterministic engine for contract, orchestration, and offline tests."""

    def __init__(self, segments: Sequence[Segment] | None = None):
        self.segments = tuple(segments or (Segment(0, 1, "synthetic transcript"),))

    def transcribe(self, audio: Path, device: str = "CPU") -> Transcript:
        inspect_audio(audio)
        return Transcript(
            self.segments,
            InferenceProvenance("mock", "deterministic-v1", device, "MOCK"),
        )

    def transcribe_samples(self, samples: list[float], device: str = "CPU") -> Transcript:
        duration = max(len(samples) / TARGET_RATE, 1e-6)
        segments = []
        for segment in self.segments:
            end = min(segment.end, duration)
            if segment.start < duration:
                segments.append(Segment(segment.start, end, segment.text))
        return Transcript(
            tuple(segments),
            InferenceProvenance("mock", "deterministic-v1", device, "MOCK"),
        )


class OpenVINOWhisperEngine:
    """Lazy OpenVINO GenAI adapter; model acquisition is always a separate action."""

    engine_name = "openvino-genai"

    def __init__(
        self,
        model_dir: Path,
        model_id: str | None = None,
        multilingual: bool = False,
    ):
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        self.model_dir = model_dir
        self.model_id = model_id or model_dir.name
        self.multilingual = multilingual
        self._pipelines: dict[str, tuple[Any, float]] = {}

    def _pipeline(self, device: str) -> tuple[Any, float]:
        requested = device.upper()
        # On the tested OpenVINO NPU, a second generate() call on the same
        # WhisperPipeline returned only a few stale words. A fresh pipeline
        # transcribed that same chunk correctly, so do not reuse NPU state.
        # Upstream tracks a matching 2026.3.0 reset symptom:
        # https://github.com/openvinotoolkit/openvino/issues/37937
        if requested == "NPU" or requested not in self._pipelines:
            import openvino as ov  # type: ignore[import-untyped]
            import openvino_genai  # type: ignore[import-untyped]

            available = tuple(ov.Core().available_devices)
            if requested not in available:
                raise RuntimeError(
                    f"requested device {requested} is unavailable; enumerated: {available}"
                )
            started = time.perf_counter()
            pipeline = openvino_genai.WhisperPipeline(str(self.model_dir), requested)
            loaded = time.perf_counter() - started
            if requested == "NPU":
                return pipeline, loaded
            self._pipelines[requested] = (pipeline, loaded)
        return self._pipelines[requested]

    def load(self, device: str) -> None:
        """Load now (the desktop app preloads); NPU pipelines are never reused (see above)."""
        if device.upper() != "NPU":
            self._pipeline(device)

    @staticmethod
    def _read_samples(audio: Path) -> list[float]:
        import array

        with wave.open(str(audio), "rb") as source:
            frames = source.readframes(source.getnframes())
        return [sample / 32768.0 for sample in array.array("h", frames)]

    def transcribe_samples(self, samples: list[float], device: str = "CPU") -> Transcript:
        """Transcribe one bounded chunk; timestamps are relative to the chunk start."""
        pipeline, loaded = self._pipeline(device.upper())
        if self.multilingual:
            result = pipeline.generate(
                samples, language="en", task="transcribe", return_timestamps=True
            )
        else:
            result = pipeline.generate(samples, return_timestamps=True)
        duration = len(samples) / TARGET_RATE
        chunks = list(getattr(result, "chunks", []) or [])
        segments: list[Segment] = []
        for chunk in chunks:
            text = str(chunk.text).strip()
            if not text:
                continue
            start = max(0.0, float(chunk.start_ts))
            end = min(float(chunk.end_ts), duration)
            # Some runtime results place hallucinated text at or beyond the
            # chunk boundary. Such spans contain no audio and cannot be given
            # a trustworthy timestamp.
            if start >= duration or end <= start:
                continue
            # Runtime float timestamps can put adjacent boundaries a few
            # microseconds apart. Snap only this numerical drift.
            if segments and 0 < segments[-1].end - start <= 1e-5 and end > segments[-1].end:
                start = segments[-1].end
            segments.append(Segment(start, end, text))
        if any(right.start < left.end for left, right in zip(segments, segments[1:], strict=False)):
            # Substantial overlap invalidates sentence timing, not the words.
            # Retain valid-span text in runtime order with explicit uncertainty.
            segments = [Segment(0.0, duration, " ".join(s.text for s in segments), uncertain=True)]
        if not chunks:
            texts = [str(t).strip() for t in result.texts]
            joined = " ".join(t for t in texts if t)
            if joined:
                # The runtime returned no timestamp chunks; do not fabricate precision,
                # so the fallback spans the whole chunk with a documented caveat.
                segments.append(Segment(0.0, duration, joined, uncertain=True))
        actual_device = device.upper()
        provenance = InferenceProvenance(
            self.engine_name,
            self.model_id,
            actual_device,
            actual_device,
            load_seconds=loaded,
        )
        return Transcript(tuple(segments), provenance)

    def transcribe(self, audio: Path, device: str = "CPU") -> Transcript:
        """Whole-file convenience path for short inputs."""
        info = inspect_audio(audio)
        if info.sample_rate != TARGET_RATE or info.channels != 1:
            raise ValueError("OpenVINO input must be normalized mono 16 kHz PCM WAV")
        started = time.perf_counter()
        transcript = self.transcribe_samples(self._read_samples(audio), device)
        elapsed = time.perf_counter() - started
        provenance = InferenceProvenance(
            transcript.provenance.engine,
            transcript.provenance.model,
            transcript.provenance.requested_device,
            transcript.provenance.actual_device,
            transcript.provenance.fallback_reason,
            transcript.provenance.load_seconds,
            elapsed,
        )
        offset_segments = tuple(transcript.segments)
        return Transcript(offset_segments, provenance, transcript.created_at)


# Segment breaks for transducer output: the Original tab, seeking, and SRT need lines.
SENTENCE_END = (".", "?", "!")
PAUSE_BREAK_SECONDS = 0.8
MAX_SEGMENT_SECONDS = 12.0


def token_segments(
    tokens: Sequence[str],
    starts: Sequence[float],
    durations: Sequence[float],
    limit: float,
    offset: float = 0.0,
) -> list[Segment]:
    """Group transducer tokens into words, then words into timed segments.

    A token that begins with a space starts a new word. Segments end after
    sentence punctuation, before a pause longer than PAUSE_BREAK_SECONDS, or once
    they reach MAX_SEGMENT_SECONDS. Times are clamped to [0, limit] and shifted by
    `offset`.
    """
    words: list[tuple[str, float, float]] = []
    for index, token in enumerate(tokens):
        start = min(max(0.0, float(starts[index])), limit)
        length = float(durations[index]) if index < len(durations) else 0.0
        end = min(max(start, start + max(0.0, length)), limit)
        if not words or token.startswith(" "):
            words.append((token, start, end))
        else:
            text, first, last = words[-1]
            words[-1] = (text + token, first, max(last, end))
    segments: list[Segment] = []
    current: list[tuple[str, float, float]] = []

    def flush() -> None:
        text = " ".join("".join(word for word, _, _ in current).split())
        if text:
            start = max(current[0][1], segments[-1].end - offset if segments else 0.0)
            end = max(start, current[-1][2])
            segments.append(Segment(start + offset, end + offset, text))
        current.clear()

    for word in words:
        if current and (
            word[1] - current[-1][2] > PAUSE_BREAK_SECONDS
            or word[1] - current[0][1] >= MAX_SEGMENT_SECONDS
        ):
            flush()
        current.append(word)
        if word[0].rstrip().endswith(SENTENCE_END):
            flush()
    if current:
        flush()
    return segments


def split_points(samples: Sequence[float], limit: int, search: int, frame: int) -> list[int]:
    """Cut positions keeping every piece at most `limit` samples long.

    Each cut is the middle of the quietest `frame` within `search` samples before
    the limit, so a word is rarely split.
    """
    cuts: list[int] = []
    start = 0
    while len(samples) - start > limit:
        low = start + max(limit - search, frame)
        best, best_energy = start + limit, float("inf")
        for position in range(low, start + limit - frame + 1, frame):
            energy = sum(abs(x) for x in samples[position : position + frame])
            if energy < best_energy:
                best, best_energy = position + frame // 2, energy
        cuts.append(best)
        start = best
    return cuts


class SherpaTransducerEngine:
    """NeMo transducer (Parakeet) through sherpa-onnx, always on the CPU."""

    engine_name = "sherpa-onnx"
    threads = 2  # Fastest on the target laptop (docs/TESTED_DECISIONS.md).
    max_input_seconds = 30.0

    def __init__(self, model_dir: Path, model_id: str | None = None):
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        self.model_dir = model_dir
        self.model_id = model_id or model_dir.name
        self._recognizer: Any = None
        self._load_seconds = 0.0

    def _get(self) -> Any:
        if self._recognizer is None:
            import sherpa_onnx  # type: ignore[import-untyped]

            folder = self.model_dir
            started = time.perf_counter()
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(folder / "encoder.int8.onnx"),
                decoder=str(folder / "decoder.int8.onnx"),
                joiner=str(folder / "joiner.int8.onnx"),
                tokens=str(folder / "tokens.txt"),
                num_threads=self.threads,
                sample_rate=TARGET_RATE,
                feature_dim=128,
                model_type="nemo_transducer",
                decoding_method="greedy_search",
                provider="cpu",
            )
            self._load_seconds = time.perf_counter() - started
        return self._recognizer

    def load(self, device: str) -> None:
        self._get()

    def transcribe_samples(self, samples: list[float], device: str = "CPU") -> Transcript:
        """Transcribe one chunk; timestamps are relative to the chunk start."""
        recognizer = self._get()
        requested = device.upper()
        fallback = None if requested == "CPU" else f"{requested} unsupported by sherpa-onnx"
        limit = round(self.max_input_seconds * TARGET_RATE)
        cuts = split_points(samples, limit, 10 * TARGET_RATE, TARGET_RATE // 10)
        segments: list[Segment] = []
        for begin, end in zip([0, *cuts], [*cuts, len(samples)], strict=True):
            if end - begin < TARGET_RATE // 10:
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(TARGET_RATE, samples[begin:end])
            recognizer.decode_stream(stream)
            result = stream.result
            offset = begin / TARGET_RATE
            piece = (end - begin) / TARGET_RATE
            tokens = list(result.tokens)
            if tokens and len(result.timestamps) == len(tokens):
                found = token_segments(
                    tokens, list(result.timestamps), list(result.durations), piece, offset
                )
                if segments and found and found[0].start < segments[-1].end:
                    first = found[0]
                    start = segments[-1].end
                    found[0] = Segment(start, max(start, first.end), first.text)
                segments.extend(found)
            elif str(result.text).strip():
                # No token times: span the piece rather than invent precision.
                text = " ".join(str(result.text).split())
                segments.append(Segment(offset, offset + piece, text, uncertain=True))
        provenance = InferenceProvenance(
            self.engine_name,
            self.model_id,
            requested,
            "CPU",
            fallback,
            load_seconds=self._load_seconds,
        )
        return Transcript(tuple(segments), provenance)

    def transcribe(self, audio: Path, device: str = "CPU") -> Transcript:
        """Whole-file convenience path for short inputs."""
        info = inspect_audio(audio)
        if info.sample_rate != TARGET_RATE or info.channels != 1:
            raise ValueError("sherpa-onnx input must be normalized mono 16 kHz PCM WAV")
        started = time.perf_counter()
        transcript = self.transcribe_samples(OpenVINOWhisperEngine._read_samples(audio), device)
        p = transcript.provenance
        provenance = InferenceProvenance(
            p.engine,
            p.model,
            p.requested_device,
            p.actual_device,
            p.fallback_reason,
            p.load_seconds,
            time.perf_counter() - started,
        )
        return Transcript(transcript.segments, provenance, transcript.created_at)
