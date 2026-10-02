"""Pythonと実行時依存を同梱したCOEIROINK Engineを構築する。"""

# ruff: noqa: F821  PyInstallerがAnalysisなどのspec専用APIを実行時に注入する。

import json
import sys
from argparse import ArgumentParser
from importlib.util import find_spec
from pathlib import Path
from shutil import copy2, copytree

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    copy_metadata,
)

parser = ArgumentParser()
parser.add_argument(
    "--backend",
    choices=("cpu", "cuda", "directml", "opencl"),
    required=True,
)
backend = parser.parse_args().backend

datas = []
binaries = []
hiddenimports = [
    # Coreでは初回利用までimportを遅延しているため、PyInstallerへ入口を明示する。
    "espnet2.bin.tts_inference",
    "espnet2.text.phoneme_tokenizer",
    "espnet2.text.token_id_converter",
    "pyworld.pyworld",
]


def collect_package(package: str) -> None:
    """ネイティブ拡張や実行時データを持つパッケージをまとめて収集する。"""

    package_datas, package_binaries, package_imports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_imports)


collect_package("pyopenjtalk")
collect_package("pyworld")
# Typeguard 4はデコレータ適用時にESPnetのソースを読むため、凍結モジュールと一緒に.pyも配置する。
datas.extend(collect_data_files("espnet2", include_py_files=True))

for distribution in (
    "coeiroink-engine",
    "coeirocore",
    "espnet",
    "kaldiio",
    "pyopenjtalk",
    "pyworld",
):
    datas.extend(copy_metadata(distribution))

optional_packages = {
    "directml": ("torch_directml",),
    "opencl": ("pytorch_ocl", "pyopencl"),
}
for package in optional_packages.get(backend, ()):
    if find_spec(package) is None:
        raise ModuleNotFoundError(
            f"{package} is required for the {backend} standalone package"
        )
    collect_package(package)

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="engine_internal",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="run",
)

# engine_root()が実行ファイルの隣を参照するため、サーバー用データは内部ディレクトリへ入れない。
target_dir = Path(DISTPATH) / "run"
for source in (
    "engine_manifest.json",
    "default.csv",
    "default_setting.yml",
    "presets.yaml",
    "LICENSE",
    "README.md",
):
    copy2(source, target_dir)
for source in ("engine_manifest_assets", "ui_template"):
    copytree(source, target_dir / source, dirs_exist_ok=True)

# 配布元のマニフェストは共通のまま保ち、Windows成果物だけ実行ファイル名を調整する。
if sys.platform == "win32":
    manifest_path = target_dir / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"] = "run.exe"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
(target_dir / "speaker_info").mkdir(exist_ok=True)
