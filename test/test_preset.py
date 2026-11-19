"""PresetManagerのYAML検証・CRUD・ロールバック・並行追加をCWD非依存で検証する。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from voicevox_engine.preset import Preset, PresetError, PresetManager

SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"


def _preset(**overrides: object) -> Preset:
    values: dict[str, object] = {
        "id": 10,
        "name": "test10",
        "speaker_uuid": SPEAKER_UUID,
        "style_id": 2,
        "speedScale": 1,
        "pitchScale": 1,
        "intonationScale": 0.5,
        "volumeScale": 1,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
    }
    values.update(overrides)
    return Preset(**values)


def _write_presets_yaml(path: Path, items: list[dict[str, object]]) -> Path:
    path.write_text(
        yaml.safe_dump(items, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _valid_preset_items() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "name": "test",
            "speaker_uuid": SPEAKER_UUID,
            "style_id": 0,
            "speedScale": 1,
            "pitchScale": 0,
            "intonationScale": 1,
            "volumeScale": 1,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
        },
        {
            "id": 2,
            "name": "test2",
            "speaker_uuid": SPEAKER_UUID,
            "style_id": 2,
            "speedScale": 1.5,
            "pitchScale": 0,
            "intonationScale": 1,
            "volumeScale": 0.7,
            "prePhonemeLength": 0.5,
            "postPhonemeLength": 0.5,
        },
    ]


@pytest.fixture
def valid_preset_file(tmp_path: Path) -> Path:
    return _write_presets_yaml(tmp_path / "presets.yaml", _valid_preset_items())


@pytest.fixture
def invalid_format_preset_file(tmp_path: Path) -> Path:
    items = _valid_preset_items()
    items[0]["style_id"] = "not_int"
    return _write_presets_yaml(tmp_path / "invalid.yaml", items)


def test_load_presets_returns_consistent_results(valid_preset_file: Path) -> None:
    manager = PresetManager(preset_path=valid_preset_file)
    first = manager.load_presets()
    second = manager.load_presets()

    assert len(first) == 2
    assert first == second


def test_load_presets_rejects_invalid_format(invalid_format_preset_file: Path) -> None:
    manager = PresetManager(preset_path=invalid_format_preset_file)

    with pytest.raises(
        PresetError, match="プリセットの設定ファイルの形式に誤りがあります"
    ):
        manager.load_presets()


def test_load_presets_rejects_duplicate_ids(tmp_path: Path) -> None:
    items = _valid_preset_items()
    items[1]["id"] = 1
    path = _write_presets_yaml(tmp_path / "duplicate.yaml", items)

    with pytest.raises(PresetError, match="プリセットIDに重複があります"):
        PresetManager(preset_path=path).load_presets()


def test_load_presets_rejects_empty_file(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(PresetError, match="プリセットの設定ファイルの内容が空です"):
        PresetManager(preset_path=empty_path).load_presets()


def test_load_presets_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PresetError, match="プリセットの設定ファイルが見つかりません"):
        PresetManager(preset_path=tmp_path / "missing.yaml").load_presets()


@pytest.mark.parametrize(
    ("requested_id", "expected_id"),
    [(10, 10), (2, 3), (-1, 3)],
)
def test_add_preset_assigns_id_and_persists(
    valid_preset_file: Path,
    requested_id: int,
    expected_id: int,
) -> None:
    manager = PresetManager(preset_path=valid_preset_file)
    preset = _preset(id=requested_id, name=f"added-{requested_id}")

    assigned_id = manager.add_preset(preset)

    assert assigned_id == expected_id
    assert len(manager.presets) == 3
    assert any(stored == preset for stored in manager.presets)


def test_add_preset_empty_list_starts_at_one(tmp_path: Path) -> None:
    path = tmp_path / "empty-list.yaml"
    path.write_text("[]\n", encoding="utf-8")
    manager = PresetManager(preset_path=path)
    preset = _preset(id=-1, name="test1", style_id=1, pitchScale=0, intonationScale=1)

    assert manager.add_preset(preset) == 1
    assert preset.id == 1
    assert manager.load_presets()[0].id == 1


def test_concurrent_add_preset_keeps_both_updates(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.yaml"
    path.write_text("[]\n", encoding="utf-8")
    managers = [PresetManager(preset_path=path) for _ in range(2)]

    def add(preset_manager: PresetManager, name: str) -> int:
        return preset_manager.add_preset(
            _preset(id=-1, name=name, style_id=1, pitchScale=0, intonationScale=1)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        ids = list(executor.map(add, managers, ("test1", "test2")))

    assert set(ids) == {1, 2}
    presets = PresetManager(preset_path=path).load_presets()
    assert {preset.id for preset in presets} == {1, 2}


def test_update_preset_replaces_matching_entry(valid_preset_file: Path) -> None:
    manager = PresetManager(preset_path=valid_preset_file)
    preset = _preset(id=1, name="test1 new")

    updated_id = manager.update_preset(preset)

    assert updated_id == 1
    assert len(manager.presets) == 2
    assert any(stored == preset for stored in manager.presets)


def test_update_preset_rejects_missing_target(valid_preset_file: Path) -> None:
    manager = PresetManager(preset_path=valid_preset_file)

    with pytest.raises(PresetError, match="更新先のプリセットが存在しません"):
        manager.update_preset(_preset(id=10, name="missing"))

    assert len(manager.presets) == 2


def test_delete_preset_removes_matching_entry(valid_preset_file: Path) -> None:
    manager = PresetManager(preset_path=valid_preset_file)

    deleted_id = manager.delete_preset(1)

    assert deleted_id == 1
    assert len(manager.presets) == 1


def test_delete_preset_rejects_missing_target(valid_preset_file: Path) -> None:
    manager = PresetManager(preset_path=valid_preset_file)

    with pytest.raises(PresetError, match="削除対象のプリセットが存在しません"):
        manager.delete_preset(10)

    assert len(manager.presets) == 2


@pytest.mark.parametrize("operation", ["add", "update", "delete"])
def test_mutations_reject_invalid_yaml_file(
    invalid_format_preset_file: Path,
    operation: str,
) -> None:
    manager = PresetManager(preset_path=invalid_format_preset_file)
    preset = _preset(id=1, name="", speaker_uuid="", style_id=0)

    with pytest.raises(
        PresetError, match="プリセットの設定ファイルの形式に誤りがあります"
    ):
        if operation == "add":
            manager.add_preset(preset)
        elif operation == "update":
            manager.update_preset(preset)
        else:
            manager.delete_preset(10)


@pytest.mark.parametrize("operation", ["add", "update", "delete"])
def test_write_failure_rolls_back_in_memory_presets(
    valid_preset_file: Path,
    tmp_path: Path,
    operation: str,
) -> None:
    manager = PresetManager(preset_path=valid_preset_file)
    manager.load_presets()
    manager.load_presets = list
    manager.preset_path = tmp_path / "missing" / "presets.yaml"

    with pytest.raises(
        PresetError, match="プリセットの設定ファイルへの書き込みに失敗しました"
    ):
        if operation == "add":
            manager.add_preset(_preset(id=10, name="test10"))
        elif operation == "update":
            manager.update_preset(_preset(id=1, name="test1 new"))
        else:
            manager.delete_preset(1)

    assert len(manager.presets) == 2
    assert manager.presets[0].name == "test"
