"""Command-line interface for Milestone 1 batch transcription.

Exit codes: 0 success, 1 operational failure, 2 argument errors, 130 safe pause.
Ordinary transcription never performs network activity; network is used only by
the explicit `models download` and `evaluate` commands.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import resident
from .acquisition import (
    DEFAULT_SPEECH_MODEL,
    MANIFEST,
    AcquisitionError,
    cpu_only,
    download_model,
    get_spec,
    verify_installed,
)
from .ai_cleanup import (
    DEFAULT_MODEL,
    CleanupCancelled,
    cleanup_session,
    cleanup_text,
    make_model,
    text_sha256,
)
from .ai_cleanup import STYLES as CLEANUP_STYLES
from .checkpoint import CheckpointError
from .chunking import DEFAULT_CHUNK_SECONDS, DEFAULT_OVERLAP_SECONDS
from .config import PROCESSING_DEFAULTS
from .devices import Benchmark, cache_key, choose_device, load_cached, run_benchmark, save_cached
from .engines import MockSpeechEngine, OpenVINOWhisperEngine, SherpaTransducerEngine
from .export import ExportError, write_export
from .formatting import STYLES as FORMAT_STYLES
from .formatting import format_session
from .media import MediaError
from .pipeline import BatchOptions, BatchRunner, PipelineError
from .storage import SessionStore
from .summary import summarize_session
from .versions import read_saved


def default_data_dir() -> Path:
    import os

    override = os.environ.get("AUDIO_TRANSCRIBER_DATA_DIR")
    if override:
        return Path(override)
    try:
        import platformdirs

        return Path(platformdirs.user_data_path("audio-transcriber", appauthor=False))
    except Exception:  # noqa: BLE001 - fall back beside the platform default
        return Path.home() / ".local" / "share" / "audio-transcriber"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-transcriber")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="list or download approved models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list")
    models_download = models_sub.add_parser("download")
    models_download.add_argument("model_id")

    subparsers.add_parser("devices", help="enumerate devices and cached benchmarks")
    subparsers.add_parser(
        "serve", help="keep models loaded and run jobs from stdin (used by the desktop app)"
    )

    transcribe = subparsers.add_parser("transcribe")
    transcribe.add_argument("input", type=Path)
    add_run_arguments(transcribe)

    resume = subparsers.add_parser("resume")
    resume.add_argument("session_id")
    add_run_arguments(resume)

    export_cmd = subparsers.add_parser("export")
    export_cmd.add_argument("session_id")
    export_cmd.add_argument("--format", required=True, choices=["json", "markdown", "text", "srt"])
    export_cmd.add_argument(
        "--layer",
        default="raw",
        choices=["raw", "balanced", "edited", "ai", "summary", "formatted"],
    )
    export_cmd.add_argument("--overwrite", action="store_true")

    cleanup = subparsers.add_parser("cleanup", help="create a separate AI-cleaned session version")
    cleanup.add_argument("session_id")
    add_cleanup_arguments(cleanup)
    cleanup.add_argument("--stop-file", type=Path, default=None)
    cleanup.add_argument("--formatting", choices=("off", *FORMAT_STYLES), default="off")
    add_summary_argument(cleanup)
    cleanup.add_argument(
        "--summary-style",
        choices=FORMAT_STYLES,
        default=None,
        help="summary layout (default: the --formatting style, or structured when off)",
    )
    add_save_only_argument(cleanup)

    formatting = subparsers.add_parser(
        "format", help="save a complete transcript with a separate layout"
    )
    formatting.add_argument("session_id")
    formatting.add_argument("--source", choices=("raw", "balanced", "ai"), default="ai")
    formatting.add_argument("--style", choices=FORMAT_STYLES, default="prose")
    formatting.add_argument("--model", default=DEFAULT_MODEL)
    formatting.add_argument("--device", choices=("AUTO", "CPU", "GPU", "NPU"), default="AUTO")
    formatting.add_argument("--stop-file", type=Path, default=None)
    add_save_only_argument(formatting, source_revision=True)

    summary = subparsers.add_parser("summarize", help="save a written summary of a session")
    summary.add_argument("session_id")
    summary.add_argument("--model", default=DEFAULT_MODEL)
    summary.add_argument("--device", choices=("AUTO", "CPU", "GPU", "NPU"), default="AUTO")
    summary.add_argument("--stop-file", type=Path, default=None)
    summary.add_argument("--style", choices=FORMAT_STYLES, default="structured")
    add_save_only_argument(summary, source_revision=True)

    text_cleanup = subparsers.add_parser(
        "cleanup-text", help="clean a local UTF-8 text file with a local model"
    )
    text_cleanup.add_argument("input", type=Path)
    text_cleanup.add_argument("--output", type=Path, required=True)
    text_cleanup.add_argument("--format", choices=("text", "json"), default="text")
    text_cleanup.add_argument("--overwrite", action="store_true")
    add_cleanup_arguments(text_cleanup)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("--model", default=None)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    evaluation.add_argument(
        "--chunk-strategy",
        choices=("fixed", "pause-v1", "pause-v2"),
        default=None,
        help="how audio is cut (default: the model's own setting)",
    )
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_SPEECH_MODEL,
        help="installed speech model (mock is an explicit synthetic test fixture)",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="longest piece of audio per step (default: the model's own setting)",
    )
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument(
        "--chunk-strategy",
        choices=("fixed", "pause-v1", "pause-v2"),
        default=None,
        help="how audio is cut (default: the model's own setting)",
    )
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument(
        "--cleanup",
        choices=("off", *CLEANUP_STYLES),
        default=None,
        help="cleanup level (default: light for lectures, medium for dictation)",
    )
    parser.add_argument(
        "--formatting",
        choices=("off", *FORMAT_STYLES),
        default=None,
        help="layout (default: structured for lectures, prose for dictation; off with cleanup off)",
    )
    parser.add_argument("--cleanup-model", default=DEFAULT_MODEL)
    parser.add_argument("--cleanup-device", choices=("AUTO", "CPU", "GPU", "NPU"), default="AUTO")
    parser.add_argument("--cleanup-mode", choices=("lecture", "dictation"), default="lecture")
    add_summary_argument(parser)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="pause safely between chunks when this file exists",
    )


def add_cleanup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("AUTO", "CPU", "GPU", "NPU"), default="AUTO")
    parser.add_argument("--mode", choices=("lecture", "dictation"), default="lecture")
    parser.add_argument("--style", choices=CLEANUP_STYLES, default="light")


def add_summary_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--summary",
        choices=("on", "off"),
        default=None,
        help="write a summary after cleanup (default: on for lectures, off for dictation)",
    )


def add_save_only_argument(parser: argparse.ArgumentParser, source_revision: bool = False) -> None:
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="keep the result as a saved version without replacing the current one",
    )
    if source_revision:
        parser.add_argument(
            "--source-revision",
            default=None,
            help="use this saved AI cleanup (a revision id) instead of the current one",
        )


def wants_summary(args: argparse.Namespace, mode: str) -> bool:
    choice = getattr(args, "summary", None)
    return choice == "on" or (choice is None and mode == "lecture")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return resident.serve()
    if args.command in ("transcribe", "resume"):
        cleanup_default, formatting_default = PROCESSING_DEFAULTS[args.cleanup_mode]
        if args.cleanup is None:
            args.cleanup = cleanup_default
        if args.formatting is None:
            args.formatting = "off" if args.cleanup == "off" else formatting_default
    data_dir = (args.data_dir or default_data_dir()).expanduser()
    store = SessionStore(data_dir)
    try:
        return dispatch(args, store, data_dir)
    except CleanupCancelled as error:
        print(str(error), file=sys.stderr)
        return 130
    except (
        AcquisitionError,
        CheckpointError,
        ExportError,
        MediaError,
        PipelineError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def dispatch(args: argparse.Namespace, store: SessionStore, data_dir: Path) -> int:
    if args.command == "models":
        return run_models(args, data_dir)
    if args.command == "devices":
        return run_devices(args, store, data_dir)
    if args.command == "transcribe":
        return run_transcribe(args, store, data_dir)
    if args.command == "resume":
        return run_resume(args, store, data_dir)
    if args.command in ("cleanup", "cleanup-text", "summarize"):
        return run_cleanup(args, store, data_dir)
    if args.command == "format":
        return run_format(args, store, data_dir)
    if args.command == "export":
        path = write_export(store, args.session_id, args.layer, args.format, args.overwrite)
        print(f"exported {args.session_id} {args.layer} layer to {path.name}")
        return 0
    if args.command == "evaluate":
        return run_evaluate(args, data_dir)
    raise PipelineError(f"unknown command {args.command}")


def run_models(args: argparse.Namespace, data_dir: Path) -> int:
    if args.models_command == "list":
        for spec in MANIFEST.values():
            print(
                f"{spec.id}  {spec.role}  {spec.download_bytes / 1e6:.1f} MB  "
                f"{spec.task}  {spec.language}  {spec.license}"
            )
        return 0
    spec = get_spec(args.model_id)
    model_root = getattr(args, "model_root", None) or data_dir / "models"
    target = download_model(spec.id, model_root)
    verified = verify_installed(model_root, spec)
    print(f"model ready: {target.name}" if verified else f"model staged at {target.name}")
    return 0


def run_devices(args: argparse.Namespace, store: SessionStore, data_dir: Path) -> int:
    devices: list[str] = []
    runtime_version = "openvino-not-installed"
    try:
        import openvino as ov  # type: ignore[import-untyped]

        core = ov.Core()
        devices = list(core.available_devices)
        runtime_version = str(ov.__version__)
    except ImportError:
        pass
    print(f"runtime: {runtime_version}")
    for device in devices:
        print(f"device: {device}")
    cache_path = data_dir / "device-benchmarks.json"
    key = cache_key("any", runtime_version, ",".join(devices))
    cached = load_cached(cache_path, key)
    if cached:
        for bench in cached:
            state = "ok" if bench.succeeded else f"failed ({bench.fallback_reason})"
            print(f"cached benchmark {bench.device}: {bench.median_seconds:.3f}s [{state}]")
    return 0


type EngineFactory = Callable[[str], Any]


def chunk_settings(args: argparse.Namespace, model_id: str) -> tuple[float, str]:
    """The requested chunking, or the model's own (each model was tested with its own)."""
    spec = MANIFEST.get(model_id)
    seconds = getattr(args, "chunk_seconds", None)
    strategy = getattr(args, "chunk_strategy", None)
    if seconds is None:
        seconds = spec.chunk_seconds if spec else DEFAULT_CHUNK_SECONDS
    return float(seconds), strategy or (spec.chunk_strategy if spec else "fixed")


def make_engine_provider(
    model_id: str, data_dir: Path, model_root: Path | None = None
) -> EngineFactory:
    if model_id in MANIFEST:
        spec = get_spec(model_id)
        if spec.task != "speech":
            raise AcquisitionError("a cleanup model cannot be used for speech transcription")
        root = model_root or data_dir / "models"
        # The resident worker checks a loaded model's files once, not on every job.
        install = resident.cached(
            ("installed", str(root), model_id), lambda: verify_installed(root, spec)
        )
        if install is None:
            raise AcquisitionError(
                f"model '{model_id}' is not installed; "
                f"run 'audio-transcriber models download {model_id}'"
            )

        if spec.runtime == "sherpa-onnx-transducer":

            def provider(device: str) -> Any:
                return resident.cached(
                    resident.speech_key(install, spec.id, "CPU"),
                    lambda: SherpaTransducerEngine(install, spec.id),
                )

            return provider
        if spec.runtime != "openvino-whisper":
            raise AcquisitionError(f"unknown speech runtime '{spec.runtime}' for {model_id}")

        def provider(device: str) -> Any:
            return resident.cached(
                resident.speech_key(install, spec.id, device),
                lambda: OpenVINOWhisperEngine(install, spec.id, multilingual=spec.multilingual),
            )

        return provider
    if model_id == "mock":

        def provider(device: str) -> Any:
            return MockSpeechEngine()

        return provider
    raise AcquisitionError(
        f"model '{model_id}' is not manifest-approved; run 'audio-transcriber models list'"
    )


def benchmark_devices(
    provider: EngineFactory, model_id: str, data_dir: Path, model_root: Path | None = None
) -> list[Benchmark]:
    """Same representative workload per device; warmup kept separate from runs."""
    try:
        import openvino as ov

        runtime_version = str(ov.__version__)
        devices = [d for d in ov.Core().available_devices]
    except ImportError:
        return []
    cache_path = data_dir / "device-benchmarks.json"
    key = cache_key(
        _model_identity(model_id, data_dir, model_root), runtime_version, ",".join(devices)
    )
    cached = load_cached(cache_path, key)
    if cached:
        return cached
    samples = [0.0] * 16_000

    def workload(device: str) -> float:
        import time

        started = time.perf_counter()
        provider(device).transcribe_samples(samples, device)
        return time.perf_counter() - started

    results = [
        run_benchmark(workload, device, "speech-chunk") for device in devices if device != "AUTO"
    ]
    save_cached(cache_path, key, results)
    return results


def _model_identity(model_id: str, data_dir: Path, model_root: Path | None = None) -> str:
    from .storage import sha256_file

    install = (model_root or data_dir / "models") / model_id
    if install.is_dir():
        parts = sorted(p.name for p in install.iterdir())
        return sha256_file(install / parts[0]) if parts else model_id
    return model_id


def resolve_device(
    requested: str,
    provider: EngineFactory,
    model_id: str,
    data_dir: Path,
    model_root: Path | None = None,
) -> tuple[str, list[Benchmark]]:
    if cpu_only(model_id):
        return "CPU", []  # No device choice and nothing to benchmark.
    is_manual = requested.upper() not in ("AUTO", "")
    if is_manual:
        return requested.upper(), []
    benchmarks = benchmark_devices(provider, model_id, data_dir, model_root)
    if benchmarks:
        # Reject an entirely failed benchmark set before creating a session.
        choose_device(benchmarks, "fastest")
    # Keep the user's policy: BatchRunner ranks and falls back among verified devices.
    return "auto", benchmarks


def run_transcribe(args: argparse.Namespace, store: SessionStore, data_dir: Path) -> int:
    model_root = getattr(args, "model_root", None)
    provider = make_engine_provider(args.model, data_dir, model_root)
    requested, benchmarks = resolve_device(args.device, provider, args.model, data_dir, model_root)
    chunk_seconds, chunk_strategy = chunk_settings(args, args.model)
    options = BatchOptions(
        model_id=args.model,
        requested_device=requested,
        chunk_seconds=chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        ffmpeg_path=args.ffmpeg,
        chunk_strategy=chunk_strategy,
    )
    stop_file = getattr(args, "stop_file", None)
    runner = BatchRunner(
        store,
        provider,
        benchmarks,
        options,
        cancel_requested=stop_file.exists if stop_file else None,
    )
    session = runner.run_new(args.input)
    report_session(session)
    return finish_transcription(args, store, data_dir, session)


def run_resume(args: argparse.Namespace, store: SessionStore, data_dir: Path) -> int:
    model_root = getattr(args, "model_root", None)
    provider = make_engine_provider(args.model, data_dir, model_root)
    requested, benchmarks = resolve_device(args.device, provider, args.model, data_dir, model_root)
    chunk_seconds, chunk_strategy = chunk_settings(args, args.model)
    options = BatchOptions(
        model_id=args.model,
        requested_device=requested,
        chunk_seconds=chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        ffmpeg_path=args.ffmpeg,
        chunk_strategy=chunk_strategy,
    )
    stop_file = getattr(args, "stop_file", None)
    runner = BatchRunner(
        store,
        provider,
        benchmarks,
        options,
        cancel_requested=stop_file.exists if stop_file else None,
    )
    session = runner.resume(args.session_id)
    report_session(session)
    return finish_transcription(args, store, data_dir, session)


def finish_transcription(
    args: argparse.Namespace, store: SessionStore, data_dir: Path, session: Any
) -> int:
    if session.status != "ready":
        return 130
    level = getattr(args, "cleanup", "off")
    if level == "off":
        if getattr(args, "formatting", "off") == "off":
            return 0
        return run_format(
            argparse.Namespace(
                session_id=session.id,
                source="raw",
                style=args.formatting,
                model=args.cleanup_model,
                model_root=args.model_root,
                device=args.cleanup_device,
                stop_file=getattr(args, "stop_file", None),
            ),
            store,
            data_dir,
        )
    cleanup_args = argparse.Namespace(
        command="cleanup",
        session_id=session.id,
        model=getattr(args, "cleanup_model", DEFAULT_MODEL),
        model_root=getattr(args, "model_root", None),
        device=getattr(args, "cleanup_device", "AUTO"),
        mode=getattr(args, "cleanup_mode", "lecture"),
        style=level,
        formatting=getattr(args, "formatting", "off"),
        summary=getattr(args, "summary", None),
        stop_file=getattr(args, "stop_file", None),
    )
    try:
        return run_cleanup(cleanup_args, store, data_dir)
    except CleanupCancelled:
        raise
    except (RuntimeError, OSError, ValueError) as error:
        print(
            f"Transcription is ready and can be exported. Cleanup or formatting failed: {error}",
            file=sys.stderr,
        )
        return 1


def report_session(session: Any) -> None:
    """Stable machine-readable lines; the Windows harness parses the prefixes."""
    diagnostics = session.diagnostics or {}
    chunk_records = diagnostics.get("chunk_records", [])
    fallback_events = diagnostics.get("fallback_events", [])
    print(f"session: {session.id}")
    print(f"status: {session.status}")
    print(f"actual_device: {diagnostics.get('actual_device', 'unknown')}")
    print(f"requested_device: {diagnostics.get('requested_device', 'unknown')}")
    print(f"chunks_completed: {len(chunk_records)}")
    print(f"fallback_events: {len(fallback_events)}")
    inference = diagnostics.get("inference_seconds")
    if inference is not None:
        print(f"inference_seconds: {inference}")
    duration = diagnostics.get("source_duration_seconds")
    if duration is not None:
        print(f"source_duration_seconds: {duration}")
    print(f"location: lectures/{session.id}/ under the application data directory")


def text_model(
    model_id: str,
    models_root: Path,
    device: str,
    cache_root: Path,
    cancelled: Callable[[], bool] | None,
) -> Any:
    """A cleanup model; the resident worker reuses its loaded copy for each job."""
    if resident.CACHE is None:
        return make_model(model_id, models_root, device, cache_root, cancelled)
    model = resident.cached(
        resident.cleanup_key(model_id, models_root, device),
        lambda: make_model(model_id, models_root, device, cache_root),
    )
    # Per-job state: this job's stop file and its own load time.
    model.cancel_requested = cancelled or (lambda: False)
    model.load_seconds = 0.0
    return model


def run_cleanup(args: argparse.Namespace, store: SessionStore, data_dir: Path) -> int:
    stop = getattr(args, "stop_file", None)
    cancelled = stop.exists if stop else None
    model = text_model(
        args.model,
        args.model_root or data_dir / "models",
        args.device,
        data_dir / "cache" / "cleanup",
        cancelled,
    )
    save_only = getattr(args, "save_only", False)
    if args.command == "summarize":
        return run_summary(
            args.session_id,
            args.model,
            args.device,
            model,
            store,
            save_only,
            getattr(args, "source_revision", None),
            getattr(args, "style", "structured"),
        )
    if args.command == "cleanup":
        path = cleanup_session(
            store,
            args.session_id,
            model,
            args.model,
            args.device,
            args.mode,
            args.style,
            cancelled,
            lambda done, total: print(f"cleanup_blocks: {done}/{total}", flush=True),
            publish=not save_only,
        )
        transcript = read_saved(path)
        revision = (transcript.transformation or {})["revision_id"]
        print(f"AI cleanup ready: {path}")
        print(f"saved_revision: ai {revision}")
        print(f"cleanup_device: {getattr(model, 'device', args.device)}")
        if getattr(model, "fallback_reason", None):
            print(f"cleanup_fallback: {model.fallback_reason}")
        records = (transcript.transformation or {}).get("blocks", [])
        if any(record["used_source"] for record in records):
            print("Some blocks retained source text after validation warnings; review the result.")
        # A saved-only cleanup is not current, so later steps read it by revision.
        source_revision = revision if save_only else None
        status = 0
        if getattr(args, "formatting", "off") != "off":
            format_args = argparse.Namespace(
                session_id=args.session_id,
                source="ai",
                style=args.formatting,
                model=args.model,
                model_root=args.model_root,
                device=args.device,
                stop_file=stop,
                save_only=save_only,
                source_revision=source_revision,
            )
            status = run_format(format_args, store, data_dir, model)
        if wants_summary(args, args.mode):
            formatting = getattr(args, "formatting", "off")
            summary_style = getattr(args, "summary_style", None) or (
                formatting if formatting != "off" else "structured"
            )
            try:
                run_summary(
                    args.session_id,
                    args.model,
                    args.device,
                    model,
                    store,
                    save_only,
                    source_revision,
                    summary_style,
                )
            except CleanupCancelled:
                raise
            except (RuntimeError, OSError, ValueError) as error:
                print(f"AI cleanup is ready. The summary failed: {error}", file=sys.stderr)
                return 1
        return status
    source, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output:
        raise ValueError("cleanup output must differ from its source")
    if output.exists() and not args.overwrite:
        raise FileExistsError("cleanup output already exists; use --overwrite to replace it")
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("cleanup text input exceeds 2 MB")
    text = source.read_text(encoding="utf-8")
    cleaned, records = cleanup_text(text, model, args.mode, args.style)
    if args.format == "json":
        import json

        spec = get_spec(args.model)
        content = (
            json.dumps(
                {
                    "text": cleaned,
                    "source_sha256": text_sha256(text),
                    "mode": args.mode,
                    "style": args.style,
                    "model": args.model,
                    "model_revision": spec.revision,
                    "requested_device": args.device,
                    "actual_device": getattr(model, "device", args.device),
                    "device_fallback": getattr(model, "fallback_reason", None),
                    "blocks": records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        content = cleaned + "\n"
    atomic_text(output, content)
    print(f"AI cleanup ready: {output}")
    if any(record["used_source"] for record in records):
        print("Some blocks retained source text after validation warnings; review the result.")
    return 0


def run_summary(
    session_id: str,
    model_id: str,
    device: str,
    model: Any,
    store: SessionStore,
    save_only: bool = False,
    source_revision: str | None = None,
    style: str = "structured",
) -> int:
    path = summarize_session(
        store,
        session_id,
        model,
        model_id,
        device,
        getattr(model, "cancel_requested", None),
        lambda done, total: print(f"cleanup_blocks: {done}/{total}", flush=True),
        publish=not save_only,
        source_revision=source_revision,
        style=style,
    )
    info = read_saved(path).transformation or {}
    print(f"Summary ready: {path}")
    print(f"saved_revision: summary {info['revision_id']}")
    if info.get("warnings"):
        print("The summary has validation warnings; review it against the transcript.")
    return 0


def run_format(
    args: argparse.Namespace, store: SessionStore, data_dir: Path, model: Any = None
) -> int:
    stop = getattr(args, "stop_file", None)
    cancelled = stop.exists if stop else None
    if args.style != "prose" and model is None:
        model = text_model(
            args.model,
            args.model_root or data_dir / "models",
            args.device,
            data_dir / "cache" / "cleanup",
            cancelled,
        )
    path = format_session(
        store,
        args.session_id,
        args.style,
        args.source,
        model,
        args.model,
        args.device,
        cancelled,
        lambda done, total: print(f"format_blocks: {done}/{total}", flush=True),
        publish=not getattr(args, "save_only", False),
        source_revision=getattr(args, "source_revision", None),
    )
    transcript = read_saved(path)
    print(f"Formatted transcript ready: {path}")
    print(f"saved_revision: formatted {(transcript.transformation or {})['revision_id']}")
    print(f"format_device: {transcript.provenance.actual_device}")
    if any(record["used_source"] for record in (transcript.transformation or {}).get("blocks", [])):
        print("Some layouts were rejected; those blocks use complete source text in prose.")
    return 0


def run_evaluate(args: argparse.Namespace, data_dir: Path) -> int:
    from .evaluate import (
        EvaluationReport,
        evaluate_case,
        fetch_case_audio,
        load_manifest,
        select_model,
        write_report,
    )

    manifest = load_manifest(args.manifest)
    model_id = args.model or str(manifest.get("model_id", ""))
    model_root = getattr(args, "model_root", None)
    provider = make_engine_provider(model_id, data_dir, model_root)
    requested, _benchmarks = resolve_device(args.device, provider, model_id, data_dir, model_root)
    chunk_seconds, chunk_strategy = chunk_settings(args, model_id)
    report = EvaluationReport(manifest_name=str(manifest.get("name", args.manifest.stem)))
    eval_media = data_dir / "evaluation" / "media"
    eval_store = SessionStore(data_dir)
    runner = BatchRunner(
        eval_store,
        provider,
        _benchmarks,
        BatchOptions(
            model_id=model_id,
            requested_device=requested,
            chunk_seconds=chunk_seconds,
            overlap_seconds=getattr(args, "overlap_seconds", DEFAULT_OVERLAP_SECONDS),
            chunk_strategy=chunk_strategy,
        ),
    )

    for source_case in manifest["cases"]:
        case = {**source_case, "model_id": model_id, "device": requested}

        def transcribe(
            path: Path, _case: dict[str, Any] = case
        ) -> tuple[str, float, None, str, str]:
            session = runner.run_new(path)
            transcript = eval_store.load_transcript(session.id, "raw")
            inference = float(session.diagnostics["inference_seconds"])
            runtime = "unavailable"
            try:
                import openvino as ov

                runtime = str(ov.__version__)
            except ImportError:
                pass
            actual = str(session.diagnostics["actual_device"])
            return transcript.text, inference, None, actual, runtime

        audio_path = fetch_case_audio(case, eval_media)
        result = evaluate_case(case, audio_path, transcribe)
        report.results.append(result)
    report.selection = select_model(report.results)

    reports_dir = data_dir / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{args.manifest.stem}.report.json"
    write_report(report, json_path)
    md_path = reports_dir / f"{args.manifest.stem}.report.md"
    lines = [f"# Evaluation: {report.manifest_name}", ""]
    for r in report.results:
        lines.append(
            f"- {r.case_id}: WER={r.wer} CER={r.cer} RTF={r.rtf} "
            f"device={r.actual_device} status={r.status}"
        )
    lines.append(f"- selection: {report.selection.get('selected_model_id')}")
    atomic_text(md_path, "\n".join(lines) + "\n")
    print(f"report written: {json_path.name}")
    return 1 if any(result.status != "measured" for result in report.results) else 0


def atomic_text(path: Path, content: str) -> None:
    from .storage import atomic_write

    atomic_write(path, content.encode("utf-8"))


if __name__ == "__main__":
    sys.exit(main())
