from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path

import pytest

from audio_transcriber.acquisition import (
    MANIFEST,
    AcquisitionError,
    ModelFile,
    ModelSpec,
    download_model,
    get_spec,
    verify_installed,
)


def local_fetcher(root: Path):
    """Deterministic stand-in for HTTPS: copies files from a fixture directory."""

    def fetch(url: str, destination: Path, expected_size: int) -> None:
        assert url.startswith("https://huggingface.co/")
        name = destination.name
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(name)
        destination.write_bytes(source.read_bytes())

    return fetch


@pytest.fixture
def tiny_fixture(tmp_path: Path) -> Path:
    """A synthetic two-file 'model' with real checksums recorded in a local spec."""
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "openvino_encoder_model.bin").write_bytes(b"\x01" * 1000)
    (root / "tokenizer.json").write_bytes(b"{}\n")
    return root


def local_spec(tmp_path: Path) -> ModelSpec:
    def digest(name: str) -> tuple[int, str]:
        data = (tmp_path / "fixture" / name).read_bytes()
        return len(data), hashlib.sha256(data).hexdigest()

    sizes = {name: digest(name) for name in ("openvino_encoder_model.bin", "tokenizer.json")}
    spec = ModelSpec(
        id="synthetic-model",
        repo="example/synthetic",
        revision="deadbeef",
        role="development-proof",
        language="en",
        multilingual=False,
        license="Apache-2.0",
        license_url="https://example.com/license",
        requires_remote_code=False,
        serialization="openvino-ir+tokenizers (no pickle)",
        min_openvino_genai="2026.3.0",
        files=tuple(ModelFile(n, sha, size) for n, (size, sha) in sorted(sizes.items())),
        source_url="synthetic fixture",
    )
    return spec


def test_bundled_manifest_entries_are_safe() -> None:
    assert MANIFEST, "manifest must not be empty"
    for spec in MANIFEST.values():
        assert spec.requires_remote_code is False
        if spec.runtime == "sherpa-onnx-transducer":
            assert spec.serialization == "onnx (no pickle)"
            assert all(f.name.endswith(".onnx") or f.name == "tokens.txt" for f in spec.files)
        else:
            assert spec.serialization.startswith("openvino-ir")
        assert ".pkl" not in " ".join(f.name for f in spec.files)
        assert spec.license
        assert spec.revision and len(spec.revision) >= 40
        total = spec.download_bytes
        assert 0 < total <= 6 * 1024**3, f"{spec.id} exceeds the text-model asset budget"
        names = [f.name for f in spec.files]
        assert len(names) == len(set(names))
        assert all(not n.endswith((".py", ".pkl", ".pt", ".pth")) for n in names)


def test_unknown_model_is_refused() -> None:
    with pytest.raises(AcquisitionError, match="not manifest-approved"):
        get_spec("some-random-model")


def test_download_with_synthetic_spec(tmp_path: Path, tiny_fixture: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    result = download_model(spec.id, models, local_fetcher(tiny_fixture))
    assert result == models / spec.id
    assert sorted(p.name for p in result.iterdir()) == sorted(f.name for f in spec.files)
    assert verify_installed(models, spec) == result
    # Idempotent: a second run verifies rather than re-downloading.
    again = download_model(spec.id, models, lambda *a: (_ for _ in ()).throw(AssertionError()))
    assert again == result


def test_checksum_failure_quarantines_staging(
    tmp_path: Path, tiny_fixture: Path, monkeypatch
) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)
    tampered = ModelFile(spec.files[0].name, "0" * 64, spec.files[0].size)
    files = list(spec.files)
    files[0] = tampered
    bad_spec = ModelSpec(**{**spec.__dict__, "files": tuple(files)})
    monkeypatch.setitem(MANIFEST, bad_spec.id, bad_spec)
    with pytest.raises(AcquisitionError, match="SHA-256"):
        download_model(bad_spec.id, models, local_fetcher(tiny_fixture))
    assert not models.exists() or not any(models.iterdir())


def test_size_mismatch_rejected_and_cleaned(
    tmp_path: Path, tiny_fixture: Path, monkeypatch
) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)
    short = ModelFile(spec.files[1].name, spec.files[1].sha256, spec.files[1].size + 5)
    files = list(spec.files)
    files[1] = short
    bad_spec = ModelSpec(**{**spec.__dict__, "files": tuple(files)})
    monkeypatch.setitem(MANIFEST, bad_spec.id, bad_spec)
    with pytest.raises(AcquisitionError, match="size mismatch"):
        download_model(bad_spec.id, models, local_fetcher(tiny_fixture))
    assert not models.exists() or not any(
        p for p in models.iterdir() if p.name.startswith(".partial-")
    )


def test_unexpected_content_rejected(tmp_path: Path, tiny_fixture: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)

    def sneaky_fetch(url: str, destination: Path, expected_size: int) -> None:
        local_fetcher(tiny_fixture)(url, destination, expected_size)
        if destination.name == "tokenizer.json":
            (destination.parent / "extra.py").write_text("malicious")

    monkeypatch.setitem(MANIFEST, spec.id, spec)
    with pytest.raises(AcquisitionError, match="unexpected staged content"):
        download_model(spec.id, models, sneaky_fetch)
    assert not models.exists() or not any(models.iterdir())


def test_symlinked_stage_entry_rejected(tmp_path: Path, tiny_fixture: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)

    def link_fetch(url: str, destination: Path, expected_size: int) -> None:
        if destination.name == "openvino_encoder_model.bin":
            try:
                os.symlink(tiny_fixture / destination.name, destination)
            except OSError as error:
                if os.name == "nt" and error.winerror == 1314:
                    pytest.skip("Windows symlink privilege is unavailable")
                raise
        else:
            local_fetcher(tiny_fixture)(url, destination, expected_size)

    monkeypatch.setitem(MANIFEST, spec.id, spec)
    with pytest.raises(AcquisitionError, match="not a plain file"):
        download_model(spec.id, models, link_fetch)
    assert not models.exists() or not any(models.iterdir())


def test_installed_corruption_detected(tmp_path: Path, tiny_fixture: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    result = download_model(spec.id, models, local_fetcher(tiny_fixture))
    (result / "tokenizer.json").write_text("{}\n tampered with different length")
    with pytest.raises(AcquisitionError, match="unexpected size|integrity check"):
        verify_installed(models, spec)


def test_interrupted_download_resumes_without_refetching_verified_files(
    tmp_path, tiny_fixture, monkeypatch
):
    from audio_transcriber.acquisition import DownloadInterrupted, partial_download

    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    models = tmp_path / "models"
    fetched = []

    def drops_second_file(url, destination, expected_size):
        fetched.append(destination.name)
        if len(fetched) == 2:
            raise DownloadInterrupted("network dropped")
        local_fetcher(tiny_fixture)(url, destination, expected_size)

    with pytest.raises(DownloadInterrupted):
        download_model(spec.id, models, drops_second_file)
    assert partial_download(models, spec.id)
    assert not (models / spec.id).exists()
    fetched.clear()
    progress = []
    result = download_model(
        spec.id,
        models,
        lambda *a: (fetched.append(a[1].name), local_fetcher(tiny_fixture)(*a)),
        progress=lambda done, total: progress.append((done, total)),
    )
    assert fetched == [spec.files[1].name]
    # The kept file counts toward progress immediately (test fetchers report nothing).
    assert progress == [(spec.files[0].size, spec.download_bytes)]
    assert verify_installed(models, spec) == result
    assert not partial_download(models, spec.id)


def test_stopped_download_keeps_progress(tmp_path, tiny_fixture, monkeypatch):
    from audio_transcriber.acquisition import partial_download

    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    models = tmp_path / "models"
    with pytest.raises(AcquisitionError, match="Download stopped"):
        download_model(spec.id, models, local_fetcher(tiny_fixture), cancelled=lambda: True)
    assert partial_download(models, spec.id)


def test_partial_download_with_unexpected_content_starts_over(tmp_path, tiny_fixture, monkeypatch):
    from audio_transcriber.acquisition import _stage_name

    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    models = tmp_path / "models"
    stage = models / _stage_name(spec)
    stage.mkdir(parents=True)
    (stage / "extra.py").write_text("unexpected")
    (stage / spec.files[0].name).write_bytes(b"wrong contents")
    old_revision = models / f".partial-{spec.id}-0000000000000000"
    old_revision.mkdir()
    result = download_model(spec.id, models, local_fetcher(tiny_fixture))
    assert verify_installed(models, spec) == result
    assert not old_revision.exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_download_repairs_existing_model_and_keeps_old_files(
    tmp_path, tiny_fixture, monkeypatch, damage
):
    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    models = tmp_path / "models"
    installed = download_model(spec.id, models, local_fetcher(tiny_fixture))
    if damage == "missing":
        (installed / "tokenizer.json").unlink()
    else:
        (installed / "tokenizer.json").write_text("broken")
    (installed / "personal-note.txt").write_text("keep this file")
    repaired = download_model(spec.id, models, local_fetcher(tiny_fixture))
    assert verify_installed(models, spec) == repaired
    previous = list(models.glob(f".previous-{spec.id}-*"))
    assert len(previous) == 1
    assert (previous[0] / "personal-note.txt").read_text() == "keep this file"


def test_failed_repair_keeps_existing_model_in_place(tmp_path, tiny_fixture, monkeypatch):
    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    models = tmp_path / "models"
    installed = download_model(spec.id, models, local_fetcher(tiny_fixture))
    (installed / "tokenizer.json").write_text("broken")
    old = {p.name: p.read_bytes() for p in installed.iterdir()}
    rename = os.rename

    def fail_promotion(source, destination):
        if Path(source).name.startswith(".partial-"):
            raise OSError("promotion failed")
        return rename(source, destination)

    monkeypatch.setattr(os, "rename", fail_promotion)
    with pytest.raises(OSError, match="promotion failed"):
        download_model(spec.id, models, local_fetcher(tiny_fixture))
    assert {p.name: p.read_bytes() for p in installed.iterdir()} == old
    assert not list(models.glob(".partial-*"))


def test_url_construction_is_manifest_only(tmp_path: Path, tiny_fixture: Path) -> None:
    spec = local_spec(tmp_path)
    url = spec.url_for("tokenizer.json")
    assert url.startswith("https://huggingface.co/")
    assert spec.revision in url
    hostile = ModelSpec(**{**spec.__dict__, "repo": "../escape"})
    with pytest.raises(AcquisitionError):
        hostile.url_for("tokenizer.json")


def test_no_runtime_network_for_transcription(
    tmp_path: Path, three_second_wav: Path, monkeypatch
) -> None:
    """After acquisition, ordinary transcription must never open a socket."""
    from audio_transcriber.engines import MockSpeechEngine
    from audio_transcriber.pipeline import BatchOptions, BatchRunner
    from audio_transcriber.storage import SessionStore

    class Denied(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("transcription attempted network access")

    monkeypatch.setattr(socket, "socket", Denied)
    store = SessionStore(tmp_path / "data")
    options = BatchOptions(model_id="mock", requested_device="auto")
    runner = BatchRunner(store, lambda device: MockSpeechEngine(), [], options)
    session = runner.run_new(three_second_wav)
    assert session.status == "ready"


def test_large_model_range_download_checks_server_range_and_contents(tmp_path, monkeypatch):
    import io

    import audio_transcriber.acquisition as acquisition

    payload = b"verified model bytes"

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}

    def open_request(request, timeout):
        assert request.get_header("Range") == f"bytes=0-{len(payload) - 1}"
        return Response(payload)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", open_request)
    destination = tmp_path / "model.bin"
    acquisition._fetch_ranges("https://huggingface.co/model", destination, len(payload))
    assert destination.read_bytes() == payload
    Response.headers = {"Content-Range": "bytes 1-10/99"}
    with pytest.raises(AcquisitionError, match="unexpected byte range"):
        acquisition._fetch_ranges("https://huggingface.co/model", destination, len(payload))


def test_brief_network_failures_are_retried_without_double_counting(tmp_path, monkeypatch):
    import io
    import urllib.error

    import audio_transcriber.acquisition as acquisition

    monkeypatch.setattr(acquisition, "RETRY_DELAYS", (0, 0, 0))
    payload = b"0123456789"
    attempts = []

    def open_url(url, timeout):
        attempts.append(url)
        if len(attempts) == 1:
            raise urllib.error.URLError(TimeoutError("The handshake operation timed out"))
        if len(attempts) == 2:
            return io.BytesIO(payload[:4])  # Connection closed partway through.
        return io.BytesIO(payload)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", open_url)
    progress = []
    fetch = acquisition.https_fetcher(report=progress.append)
    destination = tmp_path / "model.bin"
    fetch("https://huggingface.co/model", destination, len(payload))
    assert len(attempts) == 3
    assert destination.read_bytes() == payload
    assert sum(progress) == len(payload)


def test_persistent_network_failure_explains_recovery(tmp_path, monkeypatch):
    import urllib.error

    import audio_transcriber.acquisition as acquisition

    monkeypatch.setattr(acquisition, "RETRY_DELAYS", (0, 0, 0))
    attempts = []

    def offline(url, timeout):
        attempts.append(url)
        raise urllib.error.URLError(TimeoutError("_ssl.c:983: The handshake operation timed out"))

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", offline)
    fetch = acquisition.https_fetcher()
    with pytest.raises(AcquisitionError, match="handshake.*Check your internet connection"):
        fetch("https://huggingface.co/model", tmp_path / "model.bin", 10)
    assert len(attempts) == 4


def test_missing_file_is_not_retried(tmp_path, monkeypatch):
    import urllib.error

    import audio_transcriber.acquisition as acquisition

    attempts = []

    def missing(url, timeout):
        attempts.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", missing)
    with pytest.raises(urllib.error.HTTPError):
        acquisition.https_fetcher()("https://huggingface.co/model", tmp_path / "model.bin", 10)
    assert len(attempts) == 1


def test_range_download_retries_a_failed_range(tmp_path, monkeypatch):
    import io
    import urllib.error

    import audio_transcriber.acquisition as acquisition

    monkeypatch.setattr(acquisition, "RETRY_DELAYS", (0, 0, 0))
    payload = bytes(range(256)) * 4
    failed = []

    def open_request(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        if not failed:
            failed.append(start)
            raise urllib.error.URLError(ConnectionResetError("reset by peer"))

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

        return Response(payload[start : end + 1])

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", open_request)
    progress = []
    destination = tmp_path / "model.bin"
    acquisition._fetch_ranges(
        "https://huggingface.co/model", destination, len(payload), progress.append
    )
    assert destination.read_bytes() == payload
    assert sum(progress) == len(payload)


def test_range_download_resumes_from_recorded_ranges(tmp_path, monkeypatch):
    import io
    import json

    import audio_transcriber.acquisition as acquisition

    monkeypatch.setattr(acquisition, "RANGE_BLOCK", 4)
    payload = b"abcdefghijkl"
    destination = tmp_path / "model.bin"
    destination.write_bytes(b"abcd" + bytes(8))  # First range finished earlier.
    parts = tmp_path / ("model.bin" + acquisition.PARTS_SUFFIX)
    parts.write_text(json.dumps([0]))
    requested = []

    def open_request(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        requested.append(start)

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

        return Response(payload[start : end + 1])

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", open_request)
    progress = []
    acquisition._fetch_ranges(
        "https://huggingface.co/model", destination, len(payload), progress.append
    )
    assert sorted(requested) == [4, 8]
    assert destination.read_bytes() == payload
    assert sum(progress) == len(payload)
    assert not parts.exists()


def test_long_checksums_are_announced(tmp_path, tiny_fixture, monkeypatch):
    import audio_transcriber.acquisition as acquisition

    spec = local_spec(tmp_path)
    monkeypatch.setitem(MANIFEST, spec.id, spec)
    monkeypatch.setattr(acquisition, "NOTICEABLE_CHECK_BYTES", 1000)
    events = []
    download_model(
        spec.id, tmp_path / "models", local_fetcher(tiny_fixture), checking=events.append
    )
    assert events == [True, False]  # Only the 1000-byte file is long enough to announce.
