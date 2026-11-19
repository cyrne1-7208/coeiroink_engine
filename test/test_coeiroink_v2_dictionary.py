"""COEIROINK v2辞書APIから公開ユーザー辞書への変換・モーラ数検証・反映呼び出しを確認する。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from voicevox_engine.coeiroink_v2 import dictionary as v2_dictionary
from voicevox_engine.coeiroink_v2.dictionary import (
    DictionaryError,
    build_user_dictionary,
    set_dictionary,
)
from voicevox_engine.coeiroink_v2.models import DictionaryWords


def _payload(num_moras: int = 4) -> DictionaryWords:
    return DictionaryWords.model_validate(
        {
            "dictionaryWords": [
                {"word": "音声", "yomi": "オンセイ", "accent": 1, "numMoras": num_moras}
            ]
        }
    )


def test_builds_stable_public_engine_records() -> None:
    first = build_user_dictionary(_payload())
    second = build_user_dictionary(_payload())

    assert list(first) == list(second)
    word = next(iter(first.values()))
    assert word.surface == "音声"
    assert word.pronunciation == "オンセイ"
    assert word.accent_type == 1
    assert word.mora_count == 4


def test_rejects_inconsistent_mora_count() -> None:
    with pytest.raises(DictionaryError, match="numMoras"):
        build_user_dictionary(_payload(num_moras=3))


def test_set_dictionary_replaces_then_compiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_json = tmp_path / "user.json"
    compiled = tmp_path / "user.dic"
    bundled = tmp_path / "default.csv"
    bundled.write_text("", encoding="utf-8")
    replace_dictionary = Mock()
    monkeypatch.setattr(v2_dictionary, "replace_user_dict", replace_dictionary)

    set_dictionary(
        _payload(),
        user_dict_path=user_json,
        compiled_dict_path=compiled,
        default_dict_path=bundled,
    )

    records = replace_dictionary.call_args.args[0]
    assert len(records) == 1
    replace_dictionary.assert_called_once_with(
        records,
        default_dict_path=bundled,
        user_dict_path=user_json,
        compiled_dict_path=compiled,
    )
