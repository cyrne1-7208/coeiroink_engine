from pathlib import Path
from unittest.mock import patch

import pytest

from voicevox_engine.coeiroink_v2.dictionary import (
    DictionaryError,
    build_user_dictionary,
    set_dictionary,
)
from voicevox_engine.coeiroink_v2.models import DictionaryWords


def _payload(num_moras=4):
    return DictionaryWords.model_validate(
        {
            "dictionaryWords": [
                {"word": "音声", "yomi": "オンセイ", "accent": 1, "numMoras": num_moras}
            ]
        }
    )


def test_builds_stable_public_engine_records():
    first = build_user_dictionary(_payload())
    second = build_user_dictionary(_payload())

    assert list(first) == list(second)
    word = next(iter(first.values()))
    assert word.surface == "音声"
    assert word.pronunciation == "オンセイ"
    assert word.accent_type == 1
    assert word.mora_count == 4


def test_rejects_inconsistent_mora_count():
    with pytest.raises(DictionaryError, match="numMoras"):
        build_user_dictionary(_payload(num_moras=3))


def test_set_dictionary_replaces_then_compiles(tmp_path: Path):
    user_json = tmp_path / "user.json"
    compiled = tmp_path / "user.dic"
    bundled = tmp_path / "default.csv"
    bundled.write_text("", encoding="utf-8")

    with (
        patch("voicevox_engine.coeiroink_v2.dictionary.write_to_json") as write,
        patch(
            "voicevox_engine.coeiroink_v2.dictionary.update_dict"
        ) as compile_dictionary,
    ):
        set_dictionary(
            _payload(),
            user_dict_path=user_json,
            compiled_dict_path=compiled,
            default_dict_path=bundled,
        )

    records, target = write.call_args.args
    assert len(records) == 1
    assert target == user_json
    compile_dictionary.assert_called_once_with(
        default_dict_path=bundled,
        user_dict_path=user_json,
        compiled_dict_path=compiled,
    )
