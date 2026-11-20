"""Release trust boundaries, interrupted downloads, and installer handoff."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from audio_transcriber import updates
from audio_transcriber.update_helper import installer_command, read_request


def payload(data=b"installer", version="0.2.1"):
    name = f"audio-transcriber-{version}-windows-x64.exe"
    return {
        "tag_name": f"v{version}",
        "body": "Changes",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": name,
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "browser_download_url": f"https://github.com/{updates.REPOSITORY}"
                f"/releases/download/v{version}/{name}",
            }
        ],
    }


def test_only_newer_complete_stable_release():
    release = updates.parse_release(payload(), "0.2.0")
    assert release.version == "0.2.1"
    assert updates.parse_release(payload(), "0.2.1") is None
    assert updates.parse_release(payload(version="0.1.9"), "0.2.0") is None
    value = payload()
    value["prerelease"] = True
    assert updates.parse_release(value, "0.2.0") is None


def test_update_check_and_asset_validation_use_the_renamed_repository(monkeypatch):
    requested = []
    value = payload()
    value["assets"][0]["browser_download_url"] = (
        "https://github.com/lai-derek/audio-transcriber/releases/download/v0.2.1/"
        "audio-transcriber-0.2.1-windows-x64.exe"
    )

    def renamed_release(url):
        requested.append(url)
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(updates, "open_url", renamed_release)
    release = updates.latest_release("0.2.0")
    assert requested == ["https://api.github.com/repos/lai-derek/audio-transcriber/releases/latest"]
    assert release is not None and release.url == value["assets"][0]["browser_download_url"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("digest", None),
        ("digest", "sha256:wrong"),
        ("size", -1),
        ("size", True),
        ("size", updates.MAX_INSTALLER_BYTES + 1),
        ("browser_download_url", "https://evil.example/installer.exe"),
        ("name", "audio-transcriber-0.2.1-linux.exe"),
    ],
)
def test_rejects_incomplete_or_untrusted_asset(field, value):
    release = payload()
    release["assets"][0][field] = value
    with pytest.raises(updates.UpdateError):
        updates.parse_release(release, "0.2.0")


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/file",
        "file:///C:/file",
        "https://github.com@evil.example/file",
        "https://github.com:444/file",
    ],
)
def test_redirects_must_remain_on_approved_https_hosts(url):
    with pytest.raises(updates.UpdateError):
        updates.allowed_url(url)


def test_download_checks_hash_and_cleans_partial(tmp_path, monkeypatch):
    release = updates.parse_release(payload(), "0.2.0")
    monkeypatch.setattr(updates, "open_url", lambda _: io.BytesIO(b"installer"))
    progress = []
    path = updates.download_installer(
        release, tmp_path, lambda n, t: progress.append((n, t)), lambda: False
    )
    assert path.read_bytes() == b"installer"
    assert progress[-1] == (9, 9)
    path.unlink()
    monkeypatch.setattr(updates, "open_url", lambda _: io.BytesIO(b"tampering"))
    with pytest.raises(updates.UpdateError, match="SHA-256"):
        updates.download_installer(release, tmp_path, lambda *_: None, lambda: False)
    assert not list(tmp_path.iterdir())


def test_cancel_and_low_disk_leave_installed_app_alone(tmp_path, monkeypatch):
    release = updates.parse_release(payload(), "0.2.0")
    monkeypatch.setattr(updates, "open_url", lambda _: io.BytesIO(b"installer"))
    with pytest.raises(updates.UpdateError, match="cancelled"):
        updates.download_installer(release, tmp_path, lambda *_: None, lambda: True)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(updates.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})())
    with pytest.raises(updates.UpdateError, match="free space"):
        updates.download_installer(release, tmp_path, lambda *_: None, lambda: False)


def test_handoff_reverifies_installer_and_preserves_restart_arguments(tmp_path):
    release = updates.parse_release(payload(), "0.2.0")
    (tmp_path / release.filename).write_bytes(b"installer")
    install = tmp_path / "installed"
    install.mkdir()
    (install / "Audio Transcriber.exe").touch()
    arguments = ["--data-dir", str(tmp_path / "custom library")]
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "version": release.version,
                "size": release.size,
                "sha256": release.sha256,
                "install_dir": str(install),
                "arguments": arguments,
            }
        )
    )
    _, installer, actual_dir, actual_args = read_request(request)
    assert actual_args == arguments
    command = installer_command(installer, actual_dir, tmp_path / "log.txt")
    assert "/NORESTART" in command and "/NOCLOSEAPPLICATIONS" in command
    assert command[0] == str(installer)
    installer.write_bytes(b"tampering")
    with pytest.raises(updates.UpdateError):
        read_request(request)


def test_no_published_release_is_normal(monkeypatch):
    import urllib.error

    def missing(_):
        raise urllib.error.HTTPError(updates.LATEST_URL, 404, "No release", {}, None)

    monkeypatch.setattr(updates, "open_url", missing)
    assert updates.latest_release("0.2.0") is None
