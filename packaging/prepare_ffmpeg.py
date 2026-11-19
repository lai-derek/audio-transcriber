"""Prepare the exact source used by build_ffmpeg.sh; no binary downloads."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    records = json.loads((ROOT / "packaging/third-party-sources.json").read_text())
    record = next(r for r in records if r["file"] == "ffmpeg-9.0.2.tar.xz")
    archive = ROOT / "build/source-cache" / record["file"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(record["url"], timeout=60) as response:  # noqa: S310
            archive.write_bytes(response.read(record["bytes"] + 1))
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise RuntimeError("FFmpeg source checksum mismatch")
    destination = ROOT / "build/ffmpeg-source"
    if not destination.exists():
        destination.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
    print(destination / "ffmpeg-9.0.2")


if __name__ == "__main__":
    main()
