# ユーザー辞書のCRUD・インポート・Open JTalkへの反映・原子的永続化とロールバックのテスト

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import HTTPException
from pyopenjtalk import g2p

from voicevox_engine import user_dict
from voicevox_engine.coeiroink_v2.prosody import estimate_prosody
from voicevox_engine.model import UserDictWord, WordTypes
from voicevox_engine.part_of_speech_data import MAX_PRIORITY, part_of_speech_data
from voicevox_engine.user_dict import (
    apply_word,
    create_word,
    delete_word,
    import_user_dict,
    read_dict,
    reset_user_dict,
    rewrite_word,
    update_dict,
    write_to_json,
)

SEED_UUID = "aab7dda2-0d97-43c8-8cb7-3f440dab9b4e"
OTHER_UUID = "b1affe2a-d5f0-4050-926c-f28e0c1d9a98"
MISSING_UUID = "c2be4dc5-d07d-4767-8be1-04a1bb3f05a9"

VALID_DICT_JSON = {
    SEED_UUID: {
        "surface": "ｔｅｓｔ",
        "cost": part_of_speech_data[WordTypes.PROPER_NOUN].cost_candidates[5],
        "part_of_speech": "名詞",
        "part_of_speech_detail_1": "固有名詞",
        "part_of_speech_detail_2": "一般",
        "part_of_speech_detail_3": "*",
        "inflectional_type": "*",
        "inflectional_form": "*",
        "stem": "*",
        "yomi": "テスト",
        "pronunciation": "テスト",
        "accent_type": 1,
        "accent_associative_rule": "*",
    },
}

SEED_WORD = UserDictWord(
    **{k: v for k, v in VALID_DICT_JSON[SEED_UUID].items() if k != "cost"},
    priority=5,
)

IMPORT_WORD = UserDictWord(
    surface="ｔｅｓｔ２",
    priority=5,
    part_of_speech="名詞",
    part_of_speech_detail_1="固有名詞",
    part_of_speech_detail_2="一般",
    part_of_speech_detail_3="*",
    inflectional_type="*",
    inflectional_form="*",
    stem="*",
    yomi="テストツー",
    pronunciation="テストツー",
    accent_type=1,
    accent_associative_rule="*",
)


@pytest.fixture(autouse=True)
def _reset_openjtalk_user_dict() -> Iterator[None]:
    yield
    reset_user_dict()


def _write_seed_dict(path: Path) -> bytes:
    payload = json.dumps(VALID_DICT_JSON, ensure_ascii=False).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _extract_added_word(words: dict[str, UserDictWord]) -> UserDictWord:
    added = [word for word_uuid, word in words.items() if word_uuid != SEED_UUID]
    assert len(added) == 1
    return added[0]


def test_read_not_exist_json(tmp_path: Path) -> None:
    assert read_dict(user_dict_path=tmp_path / "not_exist.json") == {}


def test_update_dict_missing_default_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        update_dict(
            default_dict_path=tmp_path / "missing.csv",
            user_dict_path=tmp_path / "user_dict.json",
            compiled_dict_path=tmp_path / "user.dic",
        )


def test_create_word_defaults_to_proper_noun() -> None:
    assert (
        create_word(surface="test", pronunciation="テスト", accent_type=1) == SEED_WORD
    )


def test_priority_round_trips_for_every_part_of_speech() -> None:
    for pos in part_of_speech_data:
        for priority in range(MAX_PRIORITY + 1):
            created = create_word(
                surface="test",
                pronunciation="テスト",
                accent_type=1,
                word_type=pos,
                priority=priority,
            )
            assert created.priority == priority


def test_prosody_uses_current_user_dictionary(tmp_path: Path) -> None:
    surface = "音声確認語"
    previous = estimate_prosody(surface).plain

    apply_word(
        surface=surface,
        pronunciation="テストヨミ",
        accent_type=1,
        priority=MAX_PRIORITY,
        user_dict_path=tmp_path / "prosody.json",
        compiled_dict_path=tmp_path / "prosody.dic",
    )

    assert estimate_prosody(surface).plain != previous


@pytest.mark.parametrize("seeded", [False, True])
def test_apply_word_persists_new_entry(tmp_path: Path, seeded: bool) -> None:
    user_dict_path = tmp_path / "user.json"
    if seeded:
        _write_seed_dict(user_dict_path)

    apply_word(
        surface="test2" if seeded else "test",
        pronunciation="テストツー" if seeded else "テスト",
        accent_type=3 if seeded else 1,
        user_dict_path=user_dict_path,
        compiled_dict_path=tmp_path / "user.dic",
    )

    words = read_dict(user_dict_path=user_dict_path)
    assert len(words) == (2 if seeded else 1)
    added = _extract_added_word(words)
    assert (added.surface, added.pronunciation, added.accent_type) == (
        ("ｔｅｓｔ２", "テストツー", 3) if seeded else ("ｔｅｓｔ", "テスト", 1)
    )


def test_rewrite_word_updates_existing_and_rejects_missing_id(tmp_path: Path) -> None:
    user_dict_path = tmp_path / "rewrite.json"
    compiled_path = tmp_path / "rewrite.dic"
    _write_seed_dict(user_dict_path)

    with pytest.raises(HTTPException):
        rewrite_word(
            word_uuid=MISSING_UUID,
            surface="test2",
            pronunciation="テストツー",
            accent_type=2,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    rewrite_word(
        word_uuid=SEED_UUID,
        surface="test2",
        pronunciation="テストツー",
        accent_type=2,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    updated = read_dict(user_dict_path=user_dict_path)[SEED_UUID]
    assert (updated.surface, updated.pronunciation, updated.accent_type) == (
        "ｔｅｓｔ２",
        "テストツー",
        2,
    )


def test_delete_word_removes_existing_and_rejects_missing_id(tmp_path: Path) -> None:
    user_dict_path = tmp_path / "delete.json"
    compiled_path = tmp_path / "delete.dic"
    _write_seed_dict(user_dict_path)

    with pytest.raises(HTTPException):
        delete_word(
            word_uuid=MISSING_UUID,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    delete_word(
        word_uuid=SEED_UUID,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    assert read_dict(user_dict_path=user_dict_path) == {}


def test_import_dict_respects_override_flag(tmp_path: Path) -> None:
    user_dict_path = tmp_path / "import.json"
    compiled_path = tmp_path / "import.dic"
    _write_seed_dict(user_dict_path)

    import_user_dict(
        {OTHER_UUID: IMPORT_WORD},
        override=False,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    assert read_dict(user_dict_path) == {
        SEED_UUID: SEED_WORD,
        OTHER_UUID: IMPORT_WORD,
    }

    import_user_dict(
        {SEED_UUID: IMPORT_WORD},
        override=False,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    assert read_dict(user_dict_path)[SEED_UUID] == SEED_WORD

    import_user_dict(
        {SEED_UUID: IMPORT_WORD},
        override=True,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    assert read_dict(user_dict_path)[SEED_UUID] == IMPORT_WORD


def test_import_rejects_invalid_accent_rule_and_pos(tmp_path: Path) -> None:
    user_dict_path = tmp_path / "invalid.json"
    compiled_path = tmp_path / "invalid.dic"
    _write_seed_dict(user_dict_path)

    invalid_rule = deepcopy(IMPORT_WORD)
    invalid_rule.accent_associative_rule = "invalid"
    with pytest.raises(ValueError):
        import_user_dict(
            {SEED_UUID: invalid_rule},
            override=True,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    invalid_pos = deepcopy(IMPORT_WORD)
    invalid_pos.context_id = 2
    invalid_pos.part_of_speech = "フィラー"
    invalid_pos.part_of_speech_detail_1 = "*"
    invalid_pos.part_of_speech_detail_2 = "*"
    invalid_pos.part_of_speech_detail_3 = "*"
    with pytest.raises(ValueError):
        import_user_dict(
            {SEED_UUID: invalid_pos},
            override=True,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )


def test_update_dict_reloads_applied_word_after_reset(tmp_path: Path) -> None:
    user_dict_path = tmp_path / "reload.json"
    compiled_path = tmp_path / "reload.dic"
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_path)
    test_text = "テスト用の文字列"
    expected_pronunciation = "デフォルトノジショデハゼッタイニセイセイサレナイヨミ"

    assert g2p(text=test_text, kana=True) != expected_pronunciation

    apply_word(
        surface=test_text,
        pronunciation=expected_pronunciation,
        accent_type=1,
        priority=10,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_path,
    )
    assert g2p(text=test_text, kana=True) == expected_pronunciation

    # 疑似的にエンジンを再起動し、保存済み辞書から読みが復元されることを確認する。
    reset_user_dict()
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_path)
    assert g2p(text=test_text, kana=True) == expected_pronunciation


def test_write_to_json_uses_atomic_replace_in_target_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_dict_path = tmp_path / "nested" / "user_dict.json"
    user_dict_path.parent.mkdir()
    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def tracking_replace(source: str | os.PathLike, target: str | os.PathLike) -> None:
        calls.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(user_dict.os, "replace", tracking_replace)
    write_to_json({OTHER_UUID: IMPORT_WORD}, user_dict_path)

    assert len(calls) == 1
    source_path, target_path = calls[0]
    assert source_path.parent == user_dict_path.parent
    assert target_path == user_dict_path
    assert read_dict(user_dict_path) == {OTHER_UUID: IMPORT_WORD}
    assert list(user_dict_path.parent.iterdir()) == [user_dict_path]


def test_concurrent_apply_keeps_both_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_dict_path = tmp_path / "concurrent.json"
    compiled_path = tmp_path / "concurrent.dic"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    monkeypatch.setattr(
        user_dict,
        "_create_user_dict",
        lambda _src, dst: dst.write_bytes(b"compiled"),
    )
    monkeypatch.setattr(user_dict, "_set_user_dict", lambda _dst: None)

    def add_word(surface: str) -> None:
        try:
            barrier.wait(timeout=5)
            apply_word(
                surface=surface,
                pronunciation="テスト",
                accent_type=1,
                user_dict_path=user_dict_path,
                compiled_dict_path=compiled_path,
            )
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [
        threading.Thread(target=add_word, args=("同時追加一",)),
        threading.Thread(target=add_word, args=("同時追加二",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    assert len(read_dict(user_dict_path)) == 2


def test_compiled_dictionary_is_released_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_dict_path = tmp_path / "windows_replace.json"
    compiled_path = tmp_path / "windows_replace.dic"
    locked_path: Path | None = None
    real_replace = os.replace

    def apply_dictionary(path: Path) -> None:
        nonlocal locked_path
        locked_path = Path(path)

    def reset_dictionary() -> None:
        nonlocal locked_path
        locked_path = None

    def replace_unlocked(source: str | os.PathLike, target: str | os.PathLike) -> None:
        if locked_path in (Path(source), Path(target)):
            raise PermissionError("dictionary is still open")
        real_replace(source, target)

    monkeypatch.setattr(
        user_dict,
        "_create_user_dict",
        lambda _src, dst: dst.write_bytes(b"compiled"),
    )
    monkeypatch.setattr(user_dict, "_set_user_dict", apply_dictionary)
    monkeypatch.setattr(user_dict, "_reset_user_dict", reset_dictionary)
    monkeypatch.setattr(user_dict.os, "replace", replace_unlocked)

    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_path)

    assert compiled_path.read_bytes() == b"compiled"
    assert locked_path == compiled_path.resolve()


def test_compile_failure_keeps_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_dict_path = tmp_path / "compile_failure.json"
    compiled_path = tmp_path / "compile_failure.dic"
    previous_json = _write_seed_dict(user_dict_path)
    compiled_path.write_bytes(b"previous")

    def fail_compile(_src: Path, _dst: Path) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(user_dict, "_create_user_dict", fail_compile)

    with pytest.raises(RuntimeError, match="compile failed"):
        apply_word(
            surface="追加語",
            pronunciation="ツイカゴ",
            accent_type=1,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    assert user_dict_path.read_bytes() == previous_json
    assert compiled_path.read_bytes() == b"previous"


def test_apply_failure_rolls_back_persistent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_dict_path = tmp_path / "apply_failure.json"
    compiled_path = tmp_path / "apply_failure.dic"
    previous_json = _write_seed_dict(user_dict_path)
    compiled_path.write_bytes(b"previous")
    apply_count = 0

    def fail_first_apply(_compiled: Path) -> None:
        nonlocal apply_count
        apply_count += 1
        if apply_count == 1:
            raise RuntimeError("apply failed")

    monkeypatch.setattr(
        user_dict,
        "_create_user_dict",
        lambda _src, dst: dst.write_bytes(b"new"),
    )
    monkeypatch.setattr(user_dict, "_set_user_dict", fail_first_apply)

    with pytest.raises(RuntimeError, match="apply failed"):
        apply_word(
            surface="追加語",
            pronunciation="ツイカゴ",
            accent_type=1,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    assert apply_count == 2
    assert user_dict_path.read_bytes() == previous_json
    assert compiled_path.read_bytes() == b"previous"


@pytest.mark.parametrize("failing_target", ["json", "compiled"])
def test_persistence_failure_rolls_back_json_and_compiled_dictionary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_target: str,
) -> None:
    user_dict_path = tmp_path / f"{failing_target}_failure.json"
    compiled_path = tmp_path / f"{failing_target}_failure.dic"
    previous_json = _write_seed_dict(user_dict_path)
    compiled_path.write_bytes(b"previous")
    target_to_fail = user_dict_path if failing_target == "json" else compiled_path
    real_replace = os.replace
    failed = False

    def replace_once(source: str | os.PathLike, target: str | os.PathLike) -> None:
        nonlocal failed
        if Path(target) == target_to_fail and not failed:
            failed = True
            raise OSError(f"{failing_target} replace failed")
        real_replace(source, target)

    monkeypatch.setattr(
        user_dict,
        "_create_user_dict",
        lambda _src, dst: dst.write_bytes(b"new"),
    )
    monkeypatch.setattr(user_dict, "_set_user_dict", lambda _dst: None)
    monkeypatch.setattr(user_dict.os, "replace", replace_once)

    with pytest.raises(OSError, match=f"{failing_target} replace failed"):
        apply_word(
            surface="追加語",
            pronunciation="ツイカゴ",
            accent_type=1,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_path,
        )

    assert user_dict_path.read_bytes() == previous_json
    assert compiled_path.read_bytes() == b"previous"
