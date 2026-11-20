# Experimental Windows recipe; see packaging/README.md before redistribution.
from pathlib import Path
import os
import runpy
import sys

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

root = Path(SPECPATH).parent
# Avoid resolving Qt dependencies from unrelated tools on the developer's PATH.
windows = Path(os.environ["WINDIR"])
os.environ["PATH"] = os.pathsep.join(
    str(path) for path in (Path(sys.executable).parent, Path(sys.base_prefix), windows / "System32", windows)
)
ov_data, ov_bins, ov_hidden = collect_all("openvino")
genai_data, genai_bins, genai_hidden = collect_all("openvino_genai")
# GenAI loads this extension dynamically; import analysis does not collect its DLL.
tokenizer_bins = collect_dynamic_libs("openvino_tokenizers")
# Parakeet runs through sherpa-onnx; its extension loads onnxruntime.dll and the
# sherpa-onnx DLLs from its own folder, which import analysis does not collect.
sherpa_bins = collect_dynamic_libs("sherpa_onnx")
assets = runpy.run_path(str(root / "packaging" / "bundle_assets.py"))
notice_data = assets["collect_notices"](root, Path(workpath) / "notices")
notice_data.append((str(root / "packaging" / "GettingStarted.html"), "help"))
decoder = root / "build/ffmpeg/ffmpeg.exe"
if not decoder.is_file():
    raise RuntimeError("Build the bundled decoder with packaging/build_ffmpeg.sh first.")
notice_data.append((str(decoder), "ffmpeg"))


a = Analysis(
    [str(root / "packaging" / "desktop_entry.py")],
    pathex=[str(root / "src")],
    hookspath=[str(root / "packaging" / "hooks")],
    binaries=ov_bins + genai_bins + tokenizer_bins + sherpa_bins,
    datas=ov_data + genai_data + notice_data,
    hiddenimports=ov_hidden + genai_hidden + ["sherpa_onnx"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Audio Transcriber", console=False)

worker = Analysis(
    [str(root / "packaging" / "worker_entry.py")],
    pathex=[str(root / "src")],
    hookspath=[str(root / "packaging" / "hooks")],
    binaries=ov_bins + genai_bins + tokenizer_bins + sherpa_bins,
    datas=ov_data + genai_data + notice_data,
    hiddenimports=ov_hidden + genai_hidden + ["sherpa_onnx"],
)
worker_pyz = PYZ(worker.pure)
worker_exe = EXE(
    worker_pyz, worker.scripts, [], exclude_binaries=True,
    name="audio-transcriber-worker", console=True,
)
# Widgets uses raster rendering; the optional Mesa software-OpenGL DLL is unused.
for analysis in (a, worker):
    analysis.binaries = [
        item for item in analysis.binaries
        if Path(item[0]).name.lower() != "opengl32sw.dll"
    ]
    allowed_qt = {
        "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "Qt6Network.dll",
        "Qt6Multimedia.dll", "Qt6MultimediaWidgets.dll", "Qt6Svg.dll",
    }
    unexpected = {
        Path(item[0]).name for item in analysis.binaries
        if Path(item[0]).name.startswith("Qt6") and Path(item[0]).name not in allowed_qt
    }
    if unexpected:
        raise RuntimeError(f"Review newly collected Qt libraries and their sources: {unexpected}")
coll = COLLECT(
    exe, worker_exe, a.binaries, a.datas, worker.binaries, worker.datas,
    name="Audio Transcriber",
)
# One-file helper must remain runnable while the onedir application is replaced.
updater = Analysis(
    [str(root / "packaging" / "updater_entry.py")],
    pathex=[str(root / "src")], binaries=[], datas=[], hiddenimports=[],
)
updater_pyz = PYZ(updater.pure)
updater_exe = EXE(
    updater_pyz, updater.scripts, updater.binaries, updater.datas, [],
    name="audio-transcriber-updater", console=False,
)
import shutil
shutil.copy2(str(Path(DISTPATH) / "audio-transcriber-updater.exe"),
             str(Path(DISTPATH) / "Audio Transcriber" / "audio-transcriber-updater.exe"))
