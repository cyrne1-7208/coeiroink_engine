from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from unittest.mock import patch

import pytest

from voicevox_engine.setting import CorsPolicyMode, Setting, SettingLoader
from voicevox_engine.utility import path_utility


def _setting(allow_origin: str | None) -> Setting:
    return Setting(
        cors_policy_mode=CorsPolicyMode.localapps,
        allow_origin=allow_origin,
    )


def test_load_missing_setting_uses_default(tmp_path: Path):
    setting = SettingLoader(tmp_path / "setting.yml").load_setting_file()

    assert setting.cors_policy_mode == CorsPolicyMode.localapps
    assert setting.allow_origin is None


def test_dump_setting_file_round_trips_and_cleans_up_temp_file(tmp_path: Path):
    setting_path = tmp_path / "setting.yml"
    loader = SettingLoader(setting_path)
    setting = _setting("https://example.test")

    loader.dump_setting_file(setting)

    assert loader.load_setting_file() == setting
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))


def test_dump_setting_file_keeps_previous_file_when_replace_fails(tmp_path: Path):
    setting_path = tmp_path / "setting.yml"
    loader = SettingLoader(setting_path)
    loader.dump_setting_file(_setting("https://before.example"))
    previous_contents = setting_path.read_bytes()

    with (
        patch.object(
            path_utility.os,
            "replace",
            side_effect=OSError("replace failed"),
        ),
        pytest.raises(OSError, match="replace failed"),
    ):
        loader.dump_setting_file(_setting("https://after.example"))

    assert setting_path.read_bytes() == previous_contents
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))


def test_concurrent_setting_writes_are_serialized(tmp_path: Path):
    setting_path = tmp_path / "setting.yml"
    loaders = [SettingLoader(setting_path) for _ in range(4)]
    active_writes = 0
    maximum_active_writes = 0
    counter_lock = Lock()
    real_replace = path_utility.os.replace

    def replace_with_observation(source, destination):
        nonlocal active_writes, maximum_active_writes
        with counter_lock:
            active_writes += 1
            maximum_active_writes = max(maximum_active_writes, active_writes)
        sleep(0.01)
        try:
            return real_replace(source, destination)
        finally:
            with counter_lock:
                active_writes -= 1

    def dump_setting(args: tuple[SettingLoader, Setting]) -> None:
        loader, setting = args
        loader.dump_setting_file(setting)

    settings = [_setting(f"https://{index}.example") for index in range(4)]
    with (
        patch.object(
            path_utility.os,
            "replace",
            side_effect=replace_with_observation,
        ),
        ThreadPoolExecutor(max_workers=4) as executor,
    ):
        list(executor.map(dump_setting, zip(loaders, settings)))

    assert maximum_active_writes == 1
    assert SettingLoader(setting_path).load_setting_file().allow_origin in {
        setting.allow_origin for setting in settings
    }
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))
