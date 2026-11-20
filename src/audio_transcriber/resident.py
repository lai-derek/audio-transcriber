"""Resident worker: one long-lived process that keeps the selected models loaded.

The desktop app starts it when it opens and asks it to load the selected models.
Jobs are ordinary CLI argument lists run in this process, so they reuse the
loaded models instead of loading them again. Exiting the process unloads all
of them.

Protocol, one JSON object per line on stdin:
  {"load": [{"kind": "speech" | "cleanup", "model": id, "device": name,
             "model_root": path, "data_dir": path}, ...]}
      Keep exactly these models loaded; anything else is unloaded first.
  {"unload": true}   Unload every model (for example before a model download).
  {"run": [CLI arguments]}   Run one job; its output streams on stdout.
  {"quit": true}     Exit, which unloads every model.
Each request ends with one stdout line starting with DONE followed by its exit
code (always 0 for load and unload). Load results are LOADED or LOAD_FAILED lines.
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TextIO

DONE = "resident_done: "
LOADED = "resident_loaded: "
LOAD_FAILED = "resident_load_failed: "

# None outside the resident worker, so one-off CLI runs load and release as before.
CACHE: dict[tuple[str, ...], Any] | None = None


def cached[T](key: tuple[str, ...], make: Callable[[], T]) -> T:
    """Reuse a loaded object in the resident worker; build it fresh otherwise."""
    if CACHE is None:
        return make()
    if key not in CACHE:
        value = make()
        if value is None:  # Not installed (yet); look again next time.
            return value
        CACHE[key] = value
    return CACHE[key]  # type: ignore[no-any-return]


def speech_key(install: Path, model_id: str, device: str) -> tuple[str, ...]:
    return ("speech", str(install), model_id, device.upper())


def cleanup_key(model_id: str, models_root: Path, device: str) -> tuple[str, ...]:
    return ("cleanup", model_id, str(Path(models_root).resolve()), device.upper())


def unload(keep: set[tuple[str, ...]] | None = None) -> None:
    """Drop cached models (and install checks) not in `keep`; memory is freed on release."""
    if CACHE is None:
        return
    for key in [k for k in CACHE if keep is None or k not in keep]:
        del CACHE[key]
    gc.collect()


def load(items: list[dict[str, Any]]) -> None:
    from .acquisition import cpu_only, get_spec, verify_installed
    from .cli import benchmark_devices, make_engine_provider, text_model
    from .devices import device_order

    keep: set[tuple[str, ...]] = set()
    loaders: list[tuple[tuple[str, ...], str, str, Callable[[], None]]] = []
    for item in items:
        model_id, device = str(item["model"]), str(item["device"])
        data_dir = Path(item["data_dir"])
        root = Path(item.get("model_root") or data_dir / "models")
        try:
            installed_key = ("installed", str(root), model_id)
            install = cached(installed_key, partial(verify_installed, root, get_spec(model_id)))
            if install is None:
                continue  # Not downloaded yet; nothing to load.
            keep.add(installed_key)
            if item["kind"] == "speech":
                provider = make_engine_provider(model_id, data_dir, root)
                if cpu_only(model_id):
                    device = "CPU"
                elif device.lower() == "auto":
                    # Load where the job will start: the fastest benchmarked device.
                    order = device_order(benchmark_devices(provider, model_id, data_dir, root))
                    device = (order or ["CPU"])[0]
                key = speech_key(install, model_id, device)
                loaders.append((key, model_id, device, partial(_load_speech, provider, device)))
            else:
                key = cleanup_key(model_id, root, device)
                cache_root = data_dir / "cache" / "cleanup"
                load_text = partial(_load_text, text_model, model_id, root, device, cache_root)
                loaders.append((key, model_id, device, load_text))
            keep.add(key)
        except Exception as error:  # noqa: BLE001 - report and keep serving
            print(f"{LOAD_FAILED}{model_id}: {error}", flush=True)
    # Release what is no longer selected before loading the new selection.
    unload(keep)
    for key, model_id, device, loader in loaders:
        try:
            loader()
            print(f"{LOADED}{model_id} on {device.upper()}", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed preload must not stop jobs
            if CACHE is not None:
                CACHE.pop(key, None)
            print(f"{LOAD_FAILED}{model_id}: {error}", flush=True)


def _load_speech(provider: Callable[[str], Any], device: str) -> None:
    provider(device).load(device)


def _load_text(
    text_model: Callable[..., Any], model_id: str, root: Path, device: str, cache_root: Path
) -> None:
    text_model(model_id, root, device, cache_root, None).load()


def run(arguments: list[str]) -> int:
    from .cli import main

    try:
        return main(arguments)
    except SystemExit as exit:  # Argument errors exit through argparse.
        return exit.code if isinstance(exit.code, int) else 2
    except Exception as error:  # noqa: BLE001 - one failed job must not end the worker
        traceback.print_exc(file=sys.stdout)
        print(f"error: {error}", flush=True)
        return 1


def serve(requests: TextIO | None = None) -> int:
    global CACHE
    source = requests or sys.stdin
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    # One ordered stream: the app reads job errors and progress from the same pipe.
    stderr, sys.stderr = sys.stderr, sys.stdout
    CACHE = {}
    try:
        while line := source.readline():
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"error: unreadable request: {error}")
                print(f"{DONE}2")
                continue
            if request.get("quit"):
                break
            code = 0
            if "load" in request:
                load(list(request["load"]))
            elif request.get("unload"):
                unload()
            elif "run" in request:
                code = run([str(a) for a in request["run"]])
            print(f"{DONE}{code}", flush=True)
    finally:
        unload()
        CACHE = None
        sys.stderr = stderr
    return 0
