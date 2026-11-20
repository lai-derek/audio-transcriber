"""Explicit model acquisition: manifest-approved downloads with staged verification.

Ordinary transcription NEVER touches this module or the network. Downloads happen
only through explicit model setup in the GUI, `models download`, or evaluation.

Security policy enforced here:
- Only manifest-approved model identifiers and HTTPS hosts are allowed.
- URLs are constructed from manifest fields; user input never becomes URL syntax.
- No pickle-based files, no `trust_remote_code`, no repository scripts: every
  approved artifact is OpenVINO IR plus OpenVINO Tokenizers files, or ONNX
  weights plus a plain-text token list (sherpa-onnx models).
- Files are staged in a temporary directory, size-capped, checksum-verified, and
  checked for symlinks or unexpected content before an atomic rename promotion.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ALLOWED_HOSTS = frozenset({"huggingface.co"})
MAX_MODEL_BYTES = 6 * 1024 * 1024 * 1024  # Speech and local text-model asset budget.
READ_CHUNK = 1024 * 1024
# A multi-gigabyte download makes over a hundred requests; one dropped connection
# or TLS handshake timeout should not discard the whole download.
RETRY_DELAYS = (2.0, 5.0, 15.0)
RANGE_BLOCK = 32 * 1024 * 1024
NOTICEABLE_CHECK_BYTES = 64 * 1024**2  # Checksums of larger files take a visible moment.
PARTS_SUFFIX = ".parts.json"  # Finished byte ranges of a partially downloaded large file.


class AcquisitionError(Exception):
    """Clear failure for disallowed models, bad manifests, or corrupt downloads."""


class DownloadInterrupted(AcquisitionError):
    """Stopped by the user or the network; verified progress is kept for the next attempt."""


def _stopped() -> DownloadInterrupted:
    return DownloadInterrupted(
        "Download stopped. Downloaded parts are kept; start the download again to continue."
    )


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    revision: str
    role: str  # "development-proof" | "candidate"
    language: str
    multilingual: bool
    license: str
    license_url: str
    requires_remote_code: bool
    serialization: str
    min_openvino_genai: str
    files: tuple[ModelFile, ...]
    source_url: str
    notes: str = ""
    task: str = "speech"
    # Speech models: which engine runs them, and how the audio is cut for them.
    runtime: str = "openvino-whisper"  # or "sherpa-onnx-transducer"
    chunk_seconds: float = 30.0
    chunk_strategy: str = "fixed"

    @property
    def download_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def integrity(self) -> str:
        digest = hashlib.sha256()
        for entry in sorted(self.files, key=lambda f: f.name):
            digest.update(f"{entry.name}:{entry.sha256}\n".encode())
        return digest.hexdigest()

    def url_for(self, name: str) -> str:
        segments = f"{self.repo}/{self.revision}/{name}".split("/")
        if any(part in ("", ".", "..") or "/" in part for part in segments):
            raise AcquisitionError("manifest fields contain unsafe path segments")
        url = f"https://{sorted(ALLOWED_HOSTS)[0]}/{self.repo}/resolve/{self.revision}/{name}"
        host = urllib.parse.urlsplit(url).hostname
        if host not in ALLOWED_HOSTS:
            raise AcquisitionError(f"host {host} is not approved")
        if not url.startswith("https://"):
            raise AcquisitionError("only HTTPS sources are allowed")
        return url


def _spec(
    model_id: str,
    repo: str,
    revision: str,
    role: str,
    files: dict[str, tuple[int, str]],
    source_note: str,
    notes: str = "",
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        repo=repo,
        revision=revision,
        role=role,
        language="en",
        multilingual=False,
        license="Apache-2.0",
        license_url="https://huggingface.co/" + repo + "/blob/main/LICENSE",
        requires_remote_code=False,
        serialization="openvino-ir+tokenizers (no pickle)",
        min_openvino_genai="2026.3.0",
        files=tuple(ModelFile(n, sha, size) for n, (size, sha) in sorted(files.items())),
        source_url=f"https://huggingface.co/{repo} @ {revision} ({source_note})",
        notes=notes,
    )


_TINY_FILES: dict[str, tuple[int, str]] = {
    "README.md": (4309, "7b3a4bf3ddb965113afb0ab8051fc877b0ef8d4cf8a37c941e7762ed4f345237"),
    "added_tokens.json": (
        34604,
        "560be47bea388757f8d4cc185c5d82067426cbb6361e38016dd90ddc01ab203a",
    ),
    "config.json": (1272, "8e97a29f1f8d18ccb5e681d82ef13d68f4f7f74f0880c737e5e12315bf3404d6"),
    "generation_config.json": (
        1616,
        "6a93098954b90eaf273a21e63cf3e45f814ef72275faa74589b645e44c40ac9a",
    ),
    "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "normalizer.json": (
        52666,
        "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
    ),
    "openvino_config.json": (
        449,
        "275065fa9dd91cc44574367783325c05ca53cf6e39f3cddb867dc7cf9b89edf0",
    ),
    "openvino_decoder_model.bin": (
        25255392,
        "e133f3fd1745dff94562ba8282fecab354141c38e9a95ec7832a13365dd1a271",
    ),
    "openvino_decoder_model.xml": (
        430933,
        "7662e7b7bc65c0b263529cdcb310d2c929f75900ea5f7ea91f603654660f4997",
    ),
    "openvino_detokenizer.bin": (
        749930,
        "7045e2ab69c216fa4ef1d129fcce4d36bcf5583f01b823279278b52626238e0c",
    ),
    "openvino_detokenizer.xml": (
        9699,
        "df817c6e58fe9280f9d8680b8b1fcd9af5954204917b9de0cece77a369227f14",
    ),
    "openvino_encoder_model.bin": (
        6882592,
        "34b25ed703164b7bac0d57a6176348087346e083368e6d2eb0d6ce3778519252",
    ),
    "openvino_encoder_model.xml": (
        227102,
        "983a43e864e217f3475a0751f7230bbe497c1301242ff8d302a5b65b536df5ee",
    ),
    "openvino_tokenizer.bin": (
        1926439,
        "8c9def49b61ff1cdd929b1c4b035e6714f69157587ada9cbb61cb15d6248ea2b",
    ),
    "openvino_tokenizer.xml": (
        27011,
        "64e0c8309a4901ef7369dc22d1271b9188c14813259f8006141d5b7267053f71",
    ),
    "preprocessor_config.json": (
        356,
        "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf",
    ),
    "special_tokens_map.json": (
        2173,
        "98bdf3ec5b32e31575b02f64b0a32bde7c0449075d34484a7df9bdd3cdeb9fb9",
    ),
    "tokenizer.json": (
        3855707,
        "287537d5be89a39bd18e7e3875ad9900faa668493fb759392b8f52a492eca5db",
    ),
    "tokenizer_config.json": (
        282692,
        "7498445adabf4fd836db90b0f0d979ca9dc0b543528e5d9f1912430a5879e212",
    ),
    "vocab.json": (798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7"),
}

_BASE_FILES: dict[str, tuple[int, str]] = {
    "README.md": (4256, "7bbae3c5e222ab05c1cac21f52bc1e81a03ea829316d9fcc81fcbd7a8240e2e2"),
    "added_tokens.json": (
        34604,
        "560be47bea388757f8d4cc185c5d82067426cbb6361e38016dd90ddc01ab203a",
    ),
    "config.json": (1272, "ae9ba0d02f244aa48df4d8637fcf3ce517229fb9148ecef13d44ac83b7e2a6c3"),
    "generation_config.json": (
        1526,
        "7eb6f9dca9df06ca5ae8ed43ee5b05b200af22fedbb3fe7e0f10ea354e44b641",
    ),
    "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "normalizer.json": (
        52666,
        "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
    ),
    "openvino_config.json": (
        449,
        "0df030689716fda7e84b202a90c49e5a8123d43b3b74a444c913c2b8ada484d5",
    ),
    "openvino_decoder_model.bin": (
        40228320,
        "c1c6c3c15596ebdb06cce5d73441a361112314869a1807a28164ba5bae64331e",
    ),
    "openvino_decoder_model.xml": (
        627401,
        "5c7ff1a29d92d56d0ef73aa2ba3987b34ed8d83b826ea2b2d987e029d868ed5d",
    ),
    "openvino_detokenizer.bin": (
        749930,
        "7045e2ab69c216fa4ef1d129fcce4d36bcf5583f01b823279278b52626238e0c",
    ),
    "openvino_detokenizer.xml": (
        9699,
        "15699b574239b9d4e6ce9e37a97db2b179d3907f4afb8ee7f7950b61bb9f6902",
    ),
    "openvino_encoder_model.bin": (
        14451360,
        "a416a67095135c5cab37d2cad66ca637638257f39081ce987983e2209689405a",
    ),
    "openvino_encoder_model.xml": (
        332326,
        "9dffc149d8726bdbb980c45f191a61bc8cc3e1beb9729dbf6c92bd326b55e017",
    ),
    "openvino_tokenizer.bin": (
        1926439,
        "8c9def49b61ff1cdd929b1c4b035e6714f69157587ada9cbb61cb15d6248ea2b",
    ),
    "openvino_tokenizer.xml": (
        27011,
        "e23e9eb65cd4e0cfd7425a6a329dda7a961453fcdcec6ba4d8e8d0614b239e9a",
    ),
    "preprocessor_config.json": (
        356,
        "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf",
    ),
    "special_tokens_map.json": (
        2173,
        "98bdf3ec5b32e31575b02f64b0a32bde7c0449075d34484a7df9bdd3cdeb9fb9",
    ),
    "tokenizer.json": (
        3855707,
        "287537d5be89a39bd18e7e3875ad9900faa668493fb759392b8f52a492eca5db",
    ),
    "tokenizer_config.json": (
        282692,
        "7498445adabf4fd836db90b0f0d979ca9dc0b543528e5d9f1912430a5879e212",
    ),
    "vocab.json": (798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7"),
}

_SMALL_FILES: dict[str, tuple[int, str]] = {
    "added_tokens.json": (
        34604,
        "560be47bea388757f8d4cc185c5d82067426cbb6361e38016dd90ddc01ab203a",
    ),
    "config.json": (1277, "99bab42bf2040dc12c4f905fa98d8891ad783db26e7b2221a8fcc730c5fe5c90"),
    "generation_config.json": (
        1926,
        "18ac89af389de6a5d999b412bcd25dbae257a896269fbd27053800a507f1ec5b",
    ),
    "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "normalizer.json": (
        52666,
        "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
    ),
    "openvino_config.json": (
        451,
        "6202273df47265ad2b84dcdbe6ff0bd7127838b8d442747e04e3206e0961bddf",
    ),
    "openvino_decoder_model.bin": (
        99800544,
        "7d37c7de0d950d54c7554b29e9523a6370320422897a33d709a6c259383bf3d6",
    ),
    "openvino_decoder_model.xml": (
        1218530,
        "792abfc5c4b7b3e454c753cbd296ad63c409afc7083127741774db598662060a",
    ),
    "openvino_detokenizer.bin": (
        749930,
        "7045e2ab69c216fa4ef1d129fcce4d36bcf5583f01b823279278b52626238e0c",
    ),
    "openvino_detokenizer.xml": (
        9699,
        "6b6e686c29b3b0475c6f3a4e8bc9e64356da058a1d00d86788790ff3ae7ed36c",
    ),
    "openvino_encoder_model.bin": (
        52283296,
        "e3c12534d7d2283d591208844a98bb4ec71fa4af951602dc23f1892c533ea443",
    ),
    "openvino_encoder_model.xml": (
        649587,
        "a0537fbea58d16e2c49213e9c8cc5166203ae3ba520480e0b6a26420abaf535f",
    ),
    "openvino_tokenizer.bin": (
        1926439,
        "8c9def49b61ff1cdd929b1c4b035e6714f69157587ada9cbb61cb15d6248ea2b",
    ),
    "openvino_tokenizer.xml": (
        27011,
        "c7528b7f29f3c425c9b3a8d75a617766d1acfe66e67472fddfdad8cb8702a55b",
    ),
    "preprocessor_config.json": (
        356,
        "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf",
    ),
    "special_tokens_map.json": (
        2173,
        "98bdf3ec5b32e31575b02f64b0a32bde7c0449075d34484a7df9bdd3cdeb9fb9",
    ),
    "tokenizer.json": (
        3855707,
        "287537d5be89a39bd18e7e3875ad9900faa668493fb759392b8f52a492eca5db",
    ),
    "tokenizer_config.json": (
        282692,
        "7498445adabf4fd836db90b0f0d979ca9dc0b543528e5d9f1912430a5879e212",
    ),
    "vocab.json": (798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7"),
}

# NVIDIA Parakeet TDT 0.6B v2 (CC BY 4.0), int8 ONNX export by the sherpa-onnx project.
_PARAKEET_V2_FILES: dict[str, tuple[int, str]] = {
    "decoder.int8.onnx": (
        7257753,
        "4d097bae3bac5c96018d5293ba39f1b657296102b142affda40378f38b020f41",
    ),
    "encoder.int8.onnx": (
        652282309,
        "92a10bd9c23936ea0566c29f963bc57c2a5df4261b6a8921c39abea66557ca8b",
    ),
    "joiner.int8.onnx": (
        1739080,
        "77a29f3e5ac7d37074e0af465391b80f314015d5b97f0f90c6c7767f1d56f428",
    ),
    "tokens.txt": (9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
}
PARAKEET_V2 = "parakeet-tdt-0.6b-v2-int8"
DEFAULT_SPEECH_MODEL = PARAKEET_V2

MANIFEST: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id=PARAKEET_V2,
            repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            revision="84b478d541cdbf0c296a7ab5f1e20a6f8f998114",
            role="default",
            language="en",
            multilingual=False,
            license="CC-BY-4.0",
            license_url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
            requires_remote_code=False,
            serialization="onnx (no pickle)",
            min_openvino_genai="not used (sherpa-onnx)",
            files=tuple(
                ModelFile(n, sha, size) for n, (size, sha) in sorted(_PARAKEET_V2_FILES.items())
            ),
            source_url=(
                "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
                " @ 84b478d541cdbf0c296a7ab5f1e20a6f8f998114 (checksums verified 2026-11-18)"
            ),
            notes=(
                "Default speech model since 2026-11-18: most accurate local model in the "
                "speech comparison (docs/TESTED_DECISIONS.md). English; runs on the CPU."
            ),
            runtime="sherpa-onnx-transducer",
            # Pieces of at most 28 s (the model accepts 30 s), cut at pauses. On the TED
            # check, pause-v2 rarely found a pause deep enough in lecture audio and cut
            # mid-speech (3.3% word error); pause-v1 matched the test harness (3.0%).
            chunk_seconds=28.0,
            chunk_strategy="pause-v1",
        ),
        _spec(
            "whisper-small.en-int4-ov",
            "OpenVINO/whisper-small.en-int4-ov",
            "00df8882e77a1f0358fcfccf16ee1b33594dd401",
            "candidate",
            _SMALL_FILES,
            "checksums recorded 2026-11-17",
            "Alternative to Parakeet; slower, on the OpenVINO runtime.",
        ),
        _spec(
            "whisper-tiny.en-int4-ov",
            "OpenVINO/whisper-tiny.en-int4-ov",
            "2c4a4cb35a33f827f324c55f474d9197c586b485",
            "development-proof",
            _TINY_FILES,
            "checksums recorded from the sandbox feasibility download",
            "Lightweight development/proof model only; not recommended for real use.",
        ),
        _spec(
            "whisper-base.en-int4-ov",
            "OpenVINO/whisper-base.en-int4-ov",
            "b0da25f7e43548df7f35fb69fc7609c08242150a",
            "candidate",
            _BASE_FILES,
            "checksums recorded during manifest authoring",
            "Alternative to Parakeet; the default before 2026-11-18, on the OpenVINO runtime.",
        ),
    ]
}


# Pinned public OpenVINO export. Runtime assets are acquired explicitly, never bundled.
_TEXT_FILES: dict[str, tuple[int, str]] = {
    "added_tokens.json": (605, "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b"),
    "config.json": (773, "47107d0e2030a880ae170474241652978a54ef24ce349d275739624020b55c83"),
    "generation_config.json": (
        243,
        "7b5b9caf2e2ed66b9e6a9b7f1f8f77af4dfad41eee90f9d52e58465d0e3f6abf",
    ),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "openvino_detokenizer.bin": (
        2189639,
        "3ca47601554a3b871c1e45e8f31aa6e17d726365c075fb430d1a2089484ae8d6",
    ),
    "openvino_detokenizer.xml": (
        9594,
        "6d1a7a8cd1a8e8b2527ecbe4df2d4953213005c69c3685059ce86616f9301d53",
    ),
    "openvino_model.bin": (
        4456584796,
        "96eaf78c4e0ecfa5ca207805880b5b28e66bc15005d76f9fe5fd3cb09a33a590",
    ),
    "openvino_model.xml": (
        2441511,
        "2218d6519fc28130411e5905e81a8dd71c05d0972f17ddb85fbf5a0cd26cf36d",
    ),
    "openvino_tokenizer.bin": (
        5588628,
        "03c324c45a915c87e03c095b739c567371e5e5e131bf13b2052ac9e659b582cc",
    ),
    "openvino_tokenizer.xml": (
        25114,
        "4822836acc22026d3428a69fa24ee25fb6b06544afb33697dd9c81e9b277ab84",
    ),
    "special_tokens_map.json": (
        613,
        "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    ),
    "tokenizer.json": (
        11421896,
        "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    ),
    "tokenizer_config.json": (
        7336,
        "57fea41fa99fce34f91450643e48b3bfbb078c32c6831a3c06834f122c200510",
    ),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
MANIFEST["qwen2.5-7b-instruct-int4-ov"] = ModelSpec(
    id="qwen2.5-7b-instruct-int4-ov",
    repo="OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
    revision="51f38f02586876c08ca2a604224da20ea61685b8",
    role="cleanup-candidate",
    language="en",
    multilingual=True,
    license="Apache-2.0",
    license_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE",
    requires_remote_code=False,
    serialization="openvino-ir+tokenizers (no pickle)",
    min_openvino_genai="2026.3.0",
    files=tuple(ModelFile(n, sha, size) for n, (size, sha) in sorted(_TEXT_FILES.items())),
    source_url="https://huggingface.co/OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
    notes="Experimental local cleanup model; quality and device suitability require validation.",
    task="text-cleanup",
)


def cpu_only(model_id: str) -> bool:
    """Speech models outside OpenVINO run on the CPU; the device choice does not apply."""
    spec = MANIFEST.get(model_id)
    return spec is not None and spec.task == "speech" and spec.runtime != "openvino-whisper"


def get_spec(model_id: str) -> ModelSpec:
    try:
        return MANIFEST[model_id]
    except KeyError:
        raise AcquisitionError(
            f"model '{model_id}' is not manifest-approved; run 'audio-transcriber models list'"
        ) from None


Fetcher = Callable[[str, Path, int], None]


def https_fetcher(
    max_bytes: int = MAX_MODEL_BYTES,
    *,
    report: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Fetcher:
    def fetch(url: str, destination: Path, expected_size: int) -> None:
        if expected_size > max_bytes:
            raise AcquisitionError("manifest file exceeds the configured size limit")
        if expected_size > 256 * 1024 * 1024:
            _fetch_ranges(url, destination, expected_size, report, cancelled)
            return

        def attempt(counted: Callable[[int], None]) -> None:
            received = 0
            # URL is manifest-constructed HTTPS only (see ModelSpec.url_for).
            with (
                urllib.request.urlopen(url, timeout=20) as response,  # noqa: S310
                destination.open("wb") as out,
            ):
                while chunk := response.read(READ_CHUNK):
                    if cancelled and cancelled():
                        raise _stopped()
                    received += len(chunk)
                    if received > expected_size:
                        raise AcquisitionError("download exceeded the expected size limit")
                    out.write(chunk)
                    counted(len(chunk))
            if received < expected_size:
                raise ConnectionError("the connection closed before the file finished")

        _retrying(attempt, report, cancelled)

    return fetch


class _Counter:
    """Forward progress while remembering how much one attempt reported."""

    def __init__(self, report: Callable[[int], None] | None) -> None:
        self.report = report
        self.total = 0

    def __call__(self, count: int) -> None:
        self.total += count
        if self.report:
            self.report(count)


def _retryable(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or error.code >= 500
    return isinstance(error, (OSError, http.client.HTTPException))


def _retrying(
    attempt: Callable[[Callable[[int], None]], None],
    report: Callable[[int], None] | None,
    cancelled: Callable[[], bool] | None,
) -> None:
    """Retry brief network failures; progress from a failed attempt is withdrawn."""
    for delay in (*RETRY_DELAYS, None):
        counter = _Counter(report)
        try:
            attempt(counter)
            return
        except Exception as error:
            if counter.total and report:
                report(-counter.total)
            if not _retryable(error):
                raise
            if delay is None:
                reason = getattr(error, "reason", None) or error
                raise DownloadInterrupted(
                    f"The download was interrupted ({reason}). Check your internet "
                    "connection, then start the download again to continue where it stopped."
                ) from error
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if cancelled and cancelled():
                raise _stopped()
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def _fetch_ranges(
    url: str,
    destination: Path,
    expected_size: int,
    report: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Bound large public-model responses; verify all ranges and the final file.

    The model's normal SHA-256/size checks still run before promotion. Range
    requests avoid a multi-gigabyte response stalling before its first bytes.
    """
    block = RANGE_BLOCK
    starts = range(0, expected_size, block)
    # Finished ranges are recorded so an interrupted download resumes; the final
    # SHA-256 check still decides whether the assembled file is accepted.
    parts = destination.with_name(destination.name + PARTS_SUFFIX)
    done: set[int] = set()
    if parts.is_file() and destination.is_file():
        try:
            if destination.stat().st_size == expected_size:
                done = set(json.loads(parts.read_text(encoding="utf-8"))) & set(starts)
        except OSError, ValueError, TypeError:
            done = set()
    if not done:
        with destination.open("wb") as output:
            output.truncate(expected_size)
    elif report:
        report(sum(min(block, expected_size - start) for start in done))
    # The first failure stops the remaining ranges instead of letting them download.
    abort = threading.Event()
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def record(start: int) -> None:
        with failure_lock:
            done.add(start)
            temporary = parts.with_name(parts.name + ".tmp")
            temporary.write_text(json.dumps(sorted(done)), encoding="utf-8")
            os.replace(temporary, parts)

    def stopped() -> bool:
        return abort.is_set() or bool(cancelled and cancelled())

    def fetch_range(start: int) -> None:
        end = min(start + block, expected_size) - 1

        def attempt(counted: Callable[[int], None]) -> None:
            if stopped():
                raise _stopped()
            request = urllib.request.Request(  # noqa: S310 - manifest-constructed HTTPS
                url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                expected_range = f"bytes {start}-{end}/{expected_size}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise AcquisitionError("model server returned an unexpected byte range")
                remaining = end - start + 1
                with destination.open("r+b") as output:
                    output.seek(start)
                    while remaining:
                        if stopped():
                            raise _stopped()
                        chunk = response.read(min(READ_CHUNK, remaining))
                        if not chunk:
                            raise ConnectionError("the connection closed before the range finished")
                        output.write(chunk)
                        remaining -= len(chunk)
                        counted(len(chunk))
                    if response.read(1):
                        raise AcquisitionError("model byte range exceeded its expected size")

        try:
            _retrying(attempt, report, stopped)
            record(start)
        except BaseException as error:
            with failure_lock:
                if not abort.is_set():
                    failures.append(error)
                    abort.set()
            raise

    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in starts:
            if start not in done:
                workers.submit(fetch_range, start)
    if failures:
        raise failures[0]
    parts.unlink(missing_ok=True)


def verify_installed(models_root: Path, spec: ModelSpec) -> Path | None:
    """Return the install directory when complete and intact; otherwise None."""
    final_dir = models_root / spec.id
    if not final_dir.is_dir():
        return None
    for entry in final_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise AcquisitionError(f"installed model contains unsafe entries: {entry.name}")
    for expected in spec.files:
        path = final_dir / expected.name
        if not path.is_file():
            return None
        if path.stat().st_size != expected.size:
            raise AcquisitionError(f"installed {expected.name} has an unexpected size")
        if _sha256(path) != expected.sha256:
            raise AcquisitionError(f"installed {expected.name} fails its integrity check")
    extra = {p.name for p in final_dir.iterdir()} - {f.name for f in spec.files}
    if extra:
        raise AcquisitionError(f"installed model contains unexpected files: {sorted(extra)}")
    return final_dir


def download_model(
    model_id: str,
    models_root: Path,
    fetcher: Fetcher | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    checking: Callable[[bool], None] | None = None,
) -> Path:
    """Stage, verify, and atomically promote a manifest-approved model.

    `checking(True/False)` brackets checksum passes long enough to notice, so a
    window can say "Checking files" rather than implying a slow download.
    """
    spec = get_spec(model_id)

    @contextmanager
    def announce(size: int) -> Iterator[None]:
        noticeable = checking is not None and size >= NOTICEABLE_CHECK_BYTES
        if noticeable and checking:
            checking(True)
        try:
            yield
        finally:
            if noticeable and checking:
                checking(False)

    if spec.requires_remote_code:
        raise AcquisitionError("models requiring remote code execution are rejected")
    final_dir = models_root / spec.id
    if final_dir.is_symlink():
        raise AcquisitionError("The model directory must not be a symbolic link.")
    try:
        with announce(spec.download_bytes if final_dir.is_dir() else 0):
            existing = verify_installed(models_root, spec)
    except AcquisitionError:
        existing = None  # Explicit download repairs damaged files after staging a verified copy.
    if existing is not None:
        return existing
    received = 0
    progress_lock = threading.Lock()

    def report(count: int) -> None:
        nonlocal received
        with progress_lock:
            received += count
            if progress:
                progress(received, spec.download_bytes)

    fetch = fetcher or https_fetcher(report=report, cancelled=cancelled)
    models_root.mkdir(parents=True, exist_ok=True)
    stage = _resumable_stage(models_root, spec)
    staged = sum(path.stat().st_size for path in stage.iterdir() if path.is_file())
    if shutil.disk_usage(models_root).free < spec.download_bytes - staged + 64 * 1024**2:
        raise AcquisitionError("There is not enough free space for this model.")
    try:
        expected_names = set()
        for entry in spec.files:
            if cancelled and cancelled():
                raise _stopped()
            expected_names.add(entry.name)
            target = stage / entry.name
            with announce(entry.size if target.is_file() else 0):
                kept = _already_staged(target, entry)
            if kept:
                report(entry.size)
                continue
            fetch(spec.url_for(entry.name), target, entry.size)
            if target.is_symlink() or not target.is_file():
                raise AcquisitionError(f"staged {entry.name} is not a plain file")
            actual_size = target.stat().st_size
            if actual_size != entry.size:
                raise AcquisitionError(
                    f"staged {entry.name} size mismatch ({actual_size} != {entry.size})"
                )
            with announce(entry.size):
                actual_digest = _sha256(target)
            if actual_digest != entry.sha256:
                raise AcquisitionError(f"staged {entry.name} failed its SHA-256 check")
        extras = {p.name for p in stage.iterdir()} - expected_names
        if extras:
            raise AcquisitionError(f"unexpected staged content: {sorted(extras)}")
        if cancelled and cancelled():
            raise _stopped()
        previous = None
        if final_dir.exists():
            if final_dir.is_symlink() or not final_dir.is_dir():
                raise AcquisitionError("The model location must be a normal directory.")
            previous = models_root / f".previous-{spec.id}-{uuid.uuid4().hex}"
            os.rename(final_dir, previous)
        try:
            os.rename(stage, final_dir)
        except OSError:
            if previous is not None and not final_dir.exists():
                os.rename(previous, final_dir)
            raise
        return final_dir
    except DownloadInterrupted:
        raise  # Keep verified files and finished ranges for the next attempt.
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def partial_download(models_root: Path, model_id: str) -> bool:
    """Whether an interrupted download of this model can continue."""
    spec = get_spec(model_id)
    return (models_root / _stage_name(spec)).is_dir()


def _stage_name(spec: ModelSpec) -> str:
    # Tied to the exact file list, so a manifest update never mixes revisions.
    return f".partial-{spec.id}-{spec.integrity[:16]}"


def _resumable_stage(models_root: Path, spec: ModelSpec) -> Path:
    stage = models_root / _stage_name(spec)
    for other in models_root.glob(f".partial-{spec.id}-*"):
        if other != stage:
            shutil.rmtree(other, ignore_errors=True)
    if stage.is_symlink() or (stage.exists() and not stage.is_dir()):
        raise AcquisitionError("The partial download location must be a normal directory.")
    if stage.is_dir():
        allowed = {f.name for f in spec.files} | {f.name + PARTS_SUFFIX for f in spec.files}
        entries = list(stage.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            shutil.rmtree(stage)  # Never build on unexpected content; start over.
        else:
            for entry in entries:
                if entry.name not in allowed:
                    entry.unlink()
    stage.mkdir(exist_ok=True)
    return stage


def _already_staged(target: Path, entry: ModelFile) -> bool:
    """A file finished in an earlier attempt, still intact."""
    if target.with_name(target.name + PARTS_SUFFIX).exists() or target.is_symlink():
        return False
    return (
        target.is_file() and target.stat().st_size == entry.size and _sha256(target) == entry.sha256
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
