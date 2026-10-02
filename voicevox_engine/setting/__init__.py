import sys

from . import setting as _setting_module
from . import setting_loader as _setting_loader_module
from .setting import CorsPolicyMode, Setting
from .setting_loader import USER_SETTING_PATH, SettingLoader

# 旧CamelCaseパスは外部互換性のためエイリアスし、実ファイルはsnake_caseだけを持つ。
sys.modules[f"{__name__}.Setting"] = _setting_module
sys.modules[f"{__name__}.SettingLoader"] = _setting_loader_module

__all__ = [
    "USER_SETTING_PATH",
    "CorsPolicyMode",
    "Setting",
    "SettingLoader",
]
