import sys

from . import engine_manifest as _engine_manifest_module
from . import engine_manifest_loader as _engine_manifest_loader_module
from .engine_manifest import EngineManifest
from .engine_manifest_loader import EngineManifestLoader

# 旧CamelCaseモジュールを参照する外部コード向けに、実ファイルを戻さずimport経路だけを維持する。
sys.modules[f"{__name__}.EngineManifest"] = _engine_manifest_module
sys.modules[f"{__name__}.EngineManifestLoader"] = _engine_manifest_loader_module

__all__ = [
    "EngineManifest",
    "EngineManifestLoader",
]
