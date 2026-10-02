"""Engine設定ファイルの読み書き。"""

from pathlib import Path
from threading import RLock

import yaml

from ..utility import atomic_write_text, engine_root, get_save_dir
from .setting import Setting

DEFAULT_SETTING_PATH: Path = engine_root() / "default_setting.yml"
USER_SETTING_PATH: Path = get_save_dir() / "setting.yml"
_SETTING_LOCK = RLock()


def _atomic_write_yaml(path: Path, data: object) -> None:
    atomic_write_text(path, yaml.safe_dump(data))


class SettingLoader:
    def __init__(self, setting_file_path: Path) -> None:
        self.setting_file_path = Path(setting_file_path)

    def load_setting_file(self) -> Setting:
        with _SETTING_LOCK:
            if not self.setting_file_path.is_file():
                setting = yaml.safe_load(
                    DEFAULT_SETTING_PATH.read_text(encoding="utf-8")
                )
            else:
                setting = yaml.safe_load(
                    self.setting_file_path.read_text(encoding="utf-8")
                )

            return Setting(
                cors_policy_mode=setting["cors_policy_mode"],
                allow_origin=setting["allow_origin"],
            )

    def dump_setting_file(self, settings: Setting) -> None:
        with _SETTING_LOCK:
            _atomic_write_yaml(self.setting_file_path, settings.model_dump())
