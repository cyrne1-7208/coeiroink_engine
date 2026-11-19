"""CORSミドルウェアのOrigin制限とSettingLoaderの原子的・直列化保存を検証する。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from run import _add_cors_middleware
from voicevox_engine.setting import CorsPolicyMode, Setting, SettingLoader
from voicevox_engine.utility import path_utility


def _setting(allow_origin: str | None) -> Setting:
    return Setting(
        cors_policy_mode=CorsPolicyMode.localapps,
        allow_origin=allow_origin,
    )


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        (None, 200),
        ("app://.", 200),
        ("http://localhost:50021", 200),
        ("https://allowed.example", 200),
        ("http://localhost.evil.example", 403),
        ("https://blocked.example", 403),
    ],
)
def test_origin_policy_preserves_allowlist_and_cors_headers(
    origin: str | None, status: int
) -> None:
    app = FastAPI()
    _add_cors_middleware(app, CorsPolicyMode.localapps, ["https://allowed.example"])

    @app.get("/")
    async def root() -> dict[str, object]:
        return {}

    response = TestClient(app).get("/", headers={"Origin": origin} if origin else {})

    assert response.status_code == status
    assert response.headers.get("access-control-allow-origin") == (
        origin if status == 200 else None
    )


def test_load_missing_setting_uses_default(tmp_path: Path) -> None:
    setting = SettingLoader(tmp_path / "setting.yml").load_setting_file()

    assert setting.cors_policy_mode == CorsPolicyMode.localapps
    assert setting.allow_origin is None


def test_dump_setting_file_round_trips_and_cleans_up_temp_file(tmp_path: Path) -> None:
    setting_path = tmp_path / "setting.yml"
    loader = SettingLoader(setting_path)
    setting = _setting("https://example.test")

    loader.dump_setting_file(setting)

    assert loader.load_setting_file() == setting
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))


def test_dump_setting_file_keeps_previous_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting_path = tmp_path / "setting.yml"
    loader = SettingLoader(setting_path)
    loader.dump_setting_file(_setting("https://before.example"))
    previous_contents = setting_path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(path_utility.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        loader.dump_setting_file(_setting("https://after.example"))

    assert setting_path.read_bytes() == previous_contents
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))


def test_concurrent_setting_writes_are_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting_path = tmp_path / "setting.yml"
    loaders = [SettingLoader(setting_path) for _ in range(4)]
    active_writes = 0
    maximum_active_writes = 0
    counter_lock = Lock()
    real_replace = path_utility.os.replace

    def replace_with_observation(source: object, destination: object) -> None:
        nonlocal active_writes, maximum_active_writes
        with counter_lock:
            active_writes += 1
            maximum_active_writes = max(maximum_active_writes, active_writes)
        sleep(0.01)
        try:
            real_replace(source, destination)
        finally:
            with counter_lock:
                active_writes -= 1

    def dump_setting(args: tuple[SettingLoader, Setting]) -> None:
        loader, setting = args
        loader.dump_setting_file(setting)

    settings = [_setting(f"https://{index}.example") for index in range(4)]
    monkeypatch.setattr(path_utility.os, "replace", replace_with_observation)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(dump_setting, zip(loaders, settings, strict=True)))

    assert maximum_active_writes == 1
    assert SettingLoader(setting_path).load_setting_file().allow_origin in {
        setting.allow_origin for setting in settings
    }
    assert not list(tmp_path.glob(".setting.yml.*.tmp"))
