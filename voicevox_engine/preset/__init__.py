import sys

from . import preset as _preset_module
from . import preset_error as _preset_error_module
from . import preset_manager as _preset_manager_module
from .preset import Preset
from .preset_error import PresetError
from .preset_manager import PresetManager

# 旧ファイル名へのimportは互換エイリアスで受け、内部実装はsnake_caseへ統一する。
sys.modules[f"{__name__}.Preset"] = _preset_module
sys.modules[f"{__name__}.PresetError"] = _preset_error_module
sys.modules[f"{__name__}.PresetManager"] = _preset_manager_module

__all__ = [
    "Preset",
    "PresetError",
    "PresetManager",
]
