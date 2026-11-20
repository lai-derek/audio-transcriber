"""The resident worker runs jobs in one process and reuses each loaded model."""

import io
import json

from audio_transcriber import resident
from audio_transcriber.ai_cleanup import DEFAULT_MODEL
from audio_transcriber.storage import SessionStore


def requests(*items):
    return io.StringIO("".join(json.dumps(item) + "\n" for item in items))


def test_jobs_reuse_one_loaded_model_and_report_exit_codes(
    tmp_path, three_second_wav, monkeypatch, capsys
):
    made = []

    class Fixture:
        device = "CPU"

        def generate(self, text, mode, style):
            return text.capitalize() + "."

    def factory(*args):
        made.append(args)
        return Fixture()

    monkeypatch.setattr("audio_transcriber.cli.make_model", factory)
    data = tmp_path / "library"
    common = ["--data-dir", str(data), "--model-root", str(tmp_path / "models")]
    transcribe = [*common, "transcribe", str(three_second_wav), "--model", "mock"]
    transcribe += ["--cleanup", "medium", "--cleanup-mode", "dictation", "--formatting", "prose"]
    assert (
        resident.serve(
            requests(
                {"load": []},
                {"run": transcribe},
                {"run": transcribe},
                {"run": [*common, "summarize", "missing-session"]},
                {"run": ["--no-such-flag"]},
                {"quit": True},
                {"run": transcribe},  # Never reached: quit ends the worker.
            )
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    codes = [line.removeprefix(resident.DONE) for line in lines if line.startswith(resident.DONE)]
    assert codes == ["0", "0", "0", "1", "2"]
    # One load served both jobs; a one-off CLI run would have loaded it twice.
    assert len(made) == 1 and made[0][0] == DEFAULT_MODEL
    assert len(list(SessionStore(data).lectures.iterdir())) == 2
    assert resident.CACHE is None  # Exiting unloads everything.


def test_load_skips_missing_models_and_unload_releases_them(tmp_path, capsys):
    resident.CACHE = {}
    try:
        resident.CACHE[("speech", "x", "old-model", "CPU")] = object()
        resident.load(
            [
                {
                    "kind": "cleanup",
                    "model": DEFAULT_MODEL,
                    "device": "AUTO",
                    "model_root": str(tmp_path / "models"),
                    "data_dir": str(tmp_path),
                }
            ]
        )
        # The new selection is not installed, so nothing loads and the old model is released.
        assert resident.CACHE == {}
        assert resident.LOADED not in capsys.readouterr().out
    finally:
        resident.CACHE = None
