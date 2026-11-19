from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .cli import default_data_dir


def worker_command(arguments: list[str]) -> tuple[str, list[str]]:
    """Use a console worker beside the frozen GUI, or this Python environment."""
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).with_name("audio-transcriber-worker.exe")), arguments
    return sys.executable, ["-m", "audio_transcriber.cli", *arguments]


def main() -> int:
    parser = argparse.ArgumentParser(prog="audio-transcriber-desktop")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args()
    data_dir = (args.data_dir or default_data_dir()).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PySide6.QtCore import QLockFile, QTimer
        from PySide6.QtWidgets import QApplication, QMessageBox

        from .desktop_ui import LectureWindow
    except ImportError as error:
        import traceback

        log = data_dir / "startup-error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        error_message = (
            "Desktop dependencies could not load"
            if getattr(sys, "frozen", False)
            else ("Install the 'desktop' extra to run the Windows application")
        )
        raise SystemExit(f"{error_message}. Details: {log}") from error

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Transcriber")
    app_mutex = None
    if os.name == "nt" and getattr(sys, "frozen", False):
        from .update_helper import APP_MUTEX, UPDATE_MUTEX, WindowsMutex

        updating = WindowsMutex(UPDATE_MUTEX)
        app_mutex = WindowsMutex(APP_MUTEX)
        allowed = updating.acquire() and app_mutex.acquire()
        updating.close()
        if not allowed:
            app_mutex.close()
            QMessageBox.information(
                None, "Audio Transcriber", "The app or its updater is already open."
            )
            return 1
    lock = QLockFile(str(data_dir / "desktop.lock"))
    if not lock.tryLock(0):
        if app_mutex:
            app_mutex.close()
        message = QMessageBox()
        message.setWindowTitle("Audio Transcriber")
        message.setIcon(QMessageBox.Icon.Warning)
        message.setText("This lecture library is already open in another window.")
        message.exec()
        return 1
    window = LectureWindow(data_dir, args.model_root, args.ffmpeg)
    window.show()
    QTimer.singleShot(0, window.first_launch)
    result = app.exec()
    lock.unlock()
    if app_mutex:
        app_mutex.close()
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
