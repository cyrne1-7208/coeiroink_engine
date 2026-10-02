import importlib

import pytest


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [
        (
            "voicevox_engine.engine_manifest.EngineManifest",
            "voicevox_engine.engine_manifest.engine_manifest",
        ),
        (
            "voicevox_engine.engine_manifest.EngineManifestLoader",
            "voicevox_engine.engine_manifest.engine_manifest_loader",
        ),
        ("voicevox_engine.metas.Metas", "voicevox_engine.metas.metas"),
        ("voicevox_engine.metas.MetasStore", "voicevox_engine.metas.metas_store"),
        ("voicevox_engine.preset.Preset", "voicevox_engine.preset.preset"),
        (
            "voicevox_engine.preset.PresetError",
            "voicevox_engine.preset.preset_error",
        ),
        (
            "voicevox_engine.preset.PresetManager",
            "voicevox_engine.preset.preset_manager",
        ),
        ("voicevox_engine.setting.Setting", "voicevox_engine.setting.setting"),
        (
            "voicevox_engine.setting.SettingLoader",
            "voicevox_engine.setting.setting_loader",
        ),
    ],
)
def test_legacy_module_import_resolves_to_canonical_module(
    legacy_name: str,
    canonical_name: str,
) -> None:
    """旧CamelCase importを維持しながら、実装をsnake_caseへ集約する。"""

    assert importlib.import_module(legacy_name) is importlib.import_module(
        canonical_name
    )


def test_package_level_class_exports_remain_compatible() -> None:
    from voicevox_engine.engine_manifest import EngineManifest
    from voicevox_engine.preset import Preset, PresetError, PresetManager
    from voicevox_engine.setting import CorsPolicyMode, Setting, SettingLoader

    assert (
        EngineManifest
        is importlib.import_module(
            "voicevox_engine.engine_manifest.engine_manifest"
        ).EngineManifest
    )
    assert Preset is importlib.import_module("voicevox_engine.preset.preset").Preset
    assert (
        PresetError
        is importlib.import_module("voicevox_engine.preset.preset_error").PresetError
    )
    assert (
        PresetManager
        is importlib.import_module(
            "voicevox_engine.preset.preset_manager"
        ).PresetManager
    )
    setting_module = importlib.import_module("voicevox_engine.setting.setting")
    assert CorsPolicyMode is setting_module.CorsPolicyMode
    assert Setting is setting_module.Setting
    assert (
        SettingLoader
        is importlib.import_module(
            "voicevox_engine.setting.setting_loader"
        ).SettingLoader
    )
