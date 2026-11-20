# ユーザー辞書単語モデルのバリデーション・モーラ数計算・品詞定義の整合のテスト

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voicevox_engine.kana_parser import parse_kana
from voicevox_engine.model import UserDictWord, WordTypes
from voicevox_engine.part_of_speech_data import part_of_speech_data


def _word(**overrides: object) -> UserDictWord:
    values: dict[str, object] = {
        "surface": "テスト",
        "priority": 0,
        "part_of_speech": "名詞",
        "part_of_speech_detail_1": "固有名詞",
        "part_of_speech_detail_2": "一般",
        "part_of_speech_detail_3": "*",
        "inflectional_type": "*",
        "inflectional_form": "*",
        "stem": "*",
        "yomi": "テスト",
        "pronunciation": "テスト",
        "accent_type": 0,
        "accent_associative_rule": "*",
    }
    values.update(overrides)
    return UserDictWord(**values)


def test_word_types_match_part_of_speech_data() -> None:
    assert set(WordTypes) == set(part_of_speech_data.keys())


def test_valid_word_and_zenkaku_surface_conversion() -> None:
    word = _word(surface="test")
    assert word.surface == "ｔｅｓｔ"
    assert word.mora_count == 3


@pytest.mark.parametrize(
    ("pronunciation", "expected_mora_count"),
    [
        ("テスト", 3),
        ("ボイボ", 3),
        ("クヮンセイ", 4),
        ("キィクァヂェテェ", 4),
    ],
)
def test_mora_count_matches_expected_moras(
    pronunciation: str, expected_mora_count: int
) -> None:
    assert _word(pronunciation=pronunciation).mora_count == expected_mora_count
    parsed_count = sum(len(phrase.moras) for phrase in parse_kana(f"{pronunciation}'"))
    assert expected_mora_count == parsed_count


def test_mora_count_matches_kana_parser_for_all_sutegana_combinations() -> None:
    excluded = {"ァ", "ィ", "ゥ", "ェ", "ォ", "ッ", "ャ", "ュ", "ョ", "ヮ"}
    base_chars = [
        chr(code) for code in range(12449, 12533) if chr(code) not in excluded
    ]
    for base in base_chars:
        for small in "ァィゥェォャュョ":
            pronunciation = base + small
            expected = sum(
                len(phrase.moras) for phrase in parse_kana(f"{pronunciation}'")
            )
            assert _word(pronunciation=pronunciation).mora_count == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"pronunciation": "ぼいぼ"},
        {"pronunciation": "アィウェォ"},
        {"pronunciation": "アヮ"},
        {"pronunciation": 123},
        {"accent_type": 4},
        {"accent_type": -1},
        {"mora_count": "abc"},
    ],
)
def test_invalid_pronunciation_or_accent_type_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _word(**overrides)
