"""Small independent Windows update runner, copied outside the app before use."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .updates import Release, UpdateError, verify_installer, version_tuple

APP_MUTEX = r"Local\AudioTranscriberRunning-v1"
UPDATE_MUTEX = r"Local\AudioTranscriberUpdating-v1"


class WindowsMutex:
    kernel: Any
    handle: Any
    owned: bool

    def __init__(self, name: str) -> None:
        from ctypes import wintypes

        if sys.platform != "win32":  # Also lets type checks on Linux skip Windows calls.
            raise OSError("The update runner works only on Windows.")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.handle = self.kernel.CreateMutexW(None, False, name)
        if not self.handle:
            raise OSError("Could not create the application lock.")
        self.owned = False

    def acquire(self, milliseconds: int = 0) -> bool:
        result = self.kernel.WaitForSingleObject(self.handle, milliseconds)
        self.owned = result in (0, 0x80)
        return self.owned

    def close(self) -> None:
        if self.handle:
            if self.owned:
                self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def installer_command(installer: Path, install_dir: Path, log: Path) -> list[str]:
    return [
        str(installer),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/SP-",
        "/NORESTART",
        "/NOCLOSEAPPLICATIONS",
        "/RESTARTEXITCODE=3010",
        f"/DIR={install_dir}",
        f"/LOG={log}",
    ]


def read_request(path: Path) -> tuple[Release, Path, Path, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    version = value["version"]
    version_tuple(version)
    release = Release(version, "", "", value["size"], value["sha256"])
    installer = path.parent / release.filename
    install_dir = Path(value["install_dir"])
    if not install_dir.is_absolute() or not (install_dir / "Audio Transcriber.exe").is_file():
        raise UpdateError("The existing app installation could not be found.")
    arguments = value["arguments"]
    if (
        not isinstance(arguments, list)
        or len(arguments) % 2
        or any(
            arguments[i] not in ("--data-dir", "--model-root", "--ffmpeg")
            or not isinstance(arguments[i + 1], str)
            for i in range(0, len(arguments), 2)
        )
    ):
        raise UpdateError("The app restart options are invalid.")
    verify_installer(installer, release)
    return release, installer, install_dir, arguments


def message(text: str, retry: bool = False) -> int:
    if sys.platform != "win32":
        return 0
    return int(
        ctypes.windll.user32.MessageBoxW(
            None, text, "Audio Transcriber update", 0x10 | (5 if retry else 0)
        )
    )


def main() -> int:
    if sys.platform != "win32" or len(sys.argv) != 2:
        return 2
    request = Path(sys.argv[1]).resolve()
    update_lock = WindowsMutex(UPDATE_MUTEX)
    app_lock = WindowsMutex(APP_MUTEX)
    try:
        if not update_lock.acquire():
            raise UpdateError("Another update is already running.")
        _, installer, install_dir, arguments = read_request(request)
        deadline = time.monotonic() + 300
        while not app_lock.acquire(1000):
            if time.monotonic() >= deadline:
                raise UpdateError("The app is still open. Close it and retry the update.")
        # The updating lock prevents another GUI from starting in this interval.
        app_lock.close()
        log = request.parent / "install.log"
        while True:
            code = subprocess.run(  # noqa: S603 - checksum-verified release installer
                installer_command(installer, install_dir, log),
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).returncode
            if code == 0:
                break
            if (
                message(
                    f"Installation did not finish (code {code}).\n\n"
                    f"Your library is preserved. Retry installation?\nLog: {log}",
                    True,
                )
                != 4
            ):
                return 1
        update_lock.close()
        reopened = subprocess.Popen(  # noqa: S603 - installed GUI and validated restart options
            [str(install_dir / "Audio Transcriber.exe"), *arguments],
            cwd=install_dir,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        (request.parent / "result.json").write_text(
            json.dumps({"installed": True, "reopened_pid": reopened.pid}), encoding="utf-8"
        )
        return 0
    except Exception as error:
        message(
            f"The update could not finish.\n\n{error}\n\n"
            f"You can run the downloaded installer from:\n{request.parent}"
        )
        return 1
    finally:
        app_lock.close()
        update_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
