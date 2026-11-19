"""GitHub release discovery and bounded, verified installer downloads."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = "lai-derek/audio-transcriber"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"
LATEST_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
MAX_INSTALLER_BYTES = 2 * 1024**3
DOWNLOAD_HOSTS = {
    "github.com",
    "api.github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}
Progress = Callable[[int, int], None]


class UpdateError(Exception):
    """An update cannot be safely discovered, downloaded, or installed."""


def version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise UpdateError("The release has an unsupported version number.")
    return tuple(map(int, value.split(".")))


def allowed_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in DOWNLOAD_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise UpdateError("The update URL is not an approved HTTPS download.")


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url: str):  # type: ignore[no-untyped-def]
    allowed_url(url)
    request = urllib.request.Request(  # noqa: S310 - HTTPS host validation above
        url, headers={"User-Agent": "Audio-Transcriber", "Accept": "application/vnd.github+json"}
    )
    return urllib.request.build_opener(ReleaseRedirect()).open(request, timeout=20)


@dataclass(frozen=True)
class Release:
    version: str
    notes: str
    url: str
    size: int
    sha256: str

    @property
    def filename(self) -> str:
        return f"audio-transcriber-{self.version}-windows-x64.exe"


def parse_release(payload: object, current: str) -> Release | None:
    if not isinstance(payload, dict):
        raise UpdateError("GitHub returned invalid release information.")
    if payload.get("draft") or payload.get("prerelease"):
        return None
    tag = payload.get("tag_name", "")
    if not isinstance(tag, str):
        raise UpdateError("The release version is missing.")
    version = tag.removeprefix("v")
    if version_tuple(version) <= version_tuple(current):
        return None
    name = f"audio-transcriber-{version}-windows-x64.exe"
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        raise UpdateError("The release assets are invalid.")
    matches = [a for a in assets if isinstance(a, dict) and a.get("name") == name]
    if len(matches) != 1:
        raise UpdateError("This release does not have a Windows x64 installer yet.")
    asset = matches[0]
    digest, size, url = (
        asset.get("digest", ""),
        asset.get("size"),
        asset.get("browser_download_url"),
    )
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise UpdateError("The installer has no valid SHA-256 checksum.")
    if type(size) is not int or not 0 < size <= MAX_INSTALLER_BYTES:
        raise UpdateError("The installer size is invalid.")
    expected_url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
    if url != expected_url:
        raise UpdateError("The installer is not attached to the expected release.")
    notes = payload.get("body") or "No release notes supplied."
    return Release(version, str(notes)[:100_000], expected_url, size, digest[7:])


def latest_release(current: str) -> Release | None:
    try:
        with open_url(LATEST_URL) as response:
            data = response.read(1024**2 + 1)
        if len(data) > 1024**2:
            raise UpdateError("The release response is too large.")
        return parse_release(json.loads(data), current)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise UpdateError(f"Could not check for updates (HTTP {error.code}).") from error
    except (OSError, ValueError) as error:
        raise UpdateError(f"Could not check for updates: {error}") from error


def verify_installer(path: Path, release: Release) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != release.size:
        raise UpdateError("The downloaded installer has the wrong size.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != release.sha256:
        raise UpdateError("The installer failed its SHA-256 check. Please retry the download.")


def download_installer(
    release: Release, cache: Path, progress: Progress, cancelled: Callable[[], bool]
) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < release.size * 2 + 256 * 1024**2:
        raise UpdateError("There is not enough free space to download this update.")
    destination = cache / release.filename
    partial = destination.with_suffix(".part")
    try:
        received = 0
        with open_url(release.url) as response, partial.open("wb") as output:
            while True:
                if cancelled():
                    raise UpdateError("Update download cancelled.")
                chunk = response.read(1024**2)
                if not chunk:
                    break
                received += len(chunk)
                if received > release.size:
                    raise UpdateError("The download exceeded its expected size.")
                output.write(chunk)
                progress(received, release.size)
        verify_installer(partial, release)
        if cancelled():
            raise UpdateError("Update download cancelled.")
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)
