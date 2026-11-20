"""The app's model process: started with the app, keeps the selected models loaded.

Minimizing changes nothing. Closing the app stops the process, which unloads
every model. Jobs run one at a time in it (see `resident.py` for the protocol);
pause and cancel still work through each job's stop file.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal

from .desktop import worker_command
from .resident import DONE, LOAD_FAILED, LOADED

type Output = Callable[[str], None]
type Finished = Callable[[int, bool], None]  # exit code, crashed


class ResidentWorker(QObject):
    status = Signal(str)  # A model loaded or failed to load.

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._exited)
        self.process.errorOccurred.connect(self._error)
        self.buffer = ""
        self.selection: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] | None = None
        # Requests awaiting their DONE line, oldest first; a job carries its callbacks.
        self.pending: deque[tuple[Output, Finished] | None] = deque()

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def error_text(self) -> str:
        return self.process.errorString()

    def load(self, selection: list[dict[str, Any]]) -> None:
        """Keep exactly these models loaded; unchanged selections send nothing."""
        self.selection = selection
        if selection == self.sent and self.running:
            return
        self._send({"load": selection})

    def run(self, arguments: list[str], output: Output, finished: Finished) -> None:
        self._send({"run": arguments}, (output, finished))

    def stop(self) -> None:
        """Exit the process, which unloads every model."""
        if not self.running:
            return
        self.pending.clear()  # Nothing waits on replies once the app is closing.
        self.process.write(b'{"quit": true}\n')
        self.process.closeWriteChannel()
        if not self.process.waitForFinished(5000):
            self.process.kill()
            self.process.waitForFinished(2000)
        self.sent = None

    def _send(self, request: dict[str, Any], job: tuple[Output, Finished] | None = None) -> None:
        if not self.running:
            program, arguments = worker_command(["serve"])
            self.buffer = ""
            self.sent = None
            self.process.start(program, arguments)
            if not self.process.waitForStarted(15_000):
                if job is not None:
                    job[0](f"error: the model process could not start: {self.error_text}\n")
                    job[1](1, True)
                return
            if "load" not in request and self.selection:
                # A restarted process loads the selection before the next job.
                self._write({"load": self.selection}, None)
        self._write(request, job)

    def _write(self, request: dict[str, Any], job: tuple[Output, Finished] | None) -> None:
        if "load" in request:
            self.sent = request["load"]
        self.pending.append(job)
        self.process.write((json.dumps(request) + "\n").encode("utf-8"))

    def _read(self) -> None:
        self.buffer += bytes(self.process.readAllStandardOutput().data()).decode(
            "utf-8", errors="replace"
        )
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            text = line.rstrip("\r")
            if text.startswith(DONE):
                job = self.pending.popleft() if self.pending else None
                if job is not None:
                    try:
                        code = int(text.removeprefix(DONE))
                    except ValueError:
                        code = 1
                    job[1](code, False)
            elif text.startswith(LOADED):
                self.status.emit(f"Loaded {text.removeprefix(LOADED)}.")
            elif text.startswith(LOAD_FAILED):
                self.status.emit(f"Could not load {text.removeprefix(LOAD_FAILED)}")
            elif self.pending and self.pending[0] is not None:
                self.pending[0][0](line + "\n")

    def _exited(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read()
        jobs = [job for job in self.pending if job is not None]
        self.pending.clear()
        self.sent = None
        for output, finished in jobs:
            if self.buffer:
                output(self.buffer)
            output(f"error: the model process stopped unexpectedly (exit {code})\n")
            finished(code or 1, True)
        self.buffer = ""

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._exited(1, QProcess.ExitStatus.CrashExit)
