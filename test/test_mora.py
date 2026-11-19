"""モーラ対応表（OpenJTalk双方向変換・単射性）とモーラ表記変換を検証する。"""

from __future__ import annotations

import pytest

from voicevox_engine.mora_list import openjtalk_mora2text, openjtalk_text2mora
from voicevox_engine.text_analysis import mora_to_text


@pytest.mark.parametrize(
    ("mora", "expected"),
    [
        ("cl", "ッ"),
        ("ti", "ティ"),
        ("tu", "トゥ"),
        ("di", "ディ"),
        ("gye", "ギェ"),
        ("ye", "イェ"),
    ],
)
def test_openjtalk_mora2text_maps_special_moras(mora: str, expected: str) -> None:
    assert openjtalk_mora2text[mora] == expected


def test_openjtalk_mora2text_is_injective() -> None:
    """異なるモーラが同じ読みがなに対応しないことを確認する。"""

    values = list(openjtalk_mora2text.values())
    assert len(values) == len(set(values))


@pytest.mark.parametrize(
    ("text", "expected_mora", "expected_reverse"),
    [
        ("キィ", ("ky", "i"), "キィ"),
        ("クァ", ("kw", "a"), "クァ"),
        ("クォ", ("kw", "o"), "クォ"),
        ("グェ", ("gw", "e"), "グェ"),
        ("テェ", ("ty", "e"), "テェ"),
        ("ニィ", ("ny", "i"), "ニィ"),
        # ヂョはジョと同音モーラのため、逆引きでは代表表記ジョへ正規化される。
        ("ヂョ", ("j", "o"), "ジョ"),
    ],
)
def test_extended_foreign_moras_are_available_in_both_directions(
    text: str,
    expected_mora: tuple[str, str],
    expected_reverse: str,
) -> None:
    assert openjtalk_text2mora[text] == expected_mora
    assert openjtalk_mora2text["".join(expected_mora)] == expected_reverse


@pytest.mark.parametrize(
    ("mora", "expected"),
    [
        ("a", "ア"),
        ("i", "イ"),
        ("ka", "カ"),
        ("N", "ン"),
        ("cl", "ッ"),
        ("gye", "ギェ"),
        ("ye", "イェ"),
        ("wo", "ウォ"),
        ("A", "ア"),
        ("I", "イ"),
        ("kA", "カ"),
        ("gyE", "ギェ"),
        ("yE", "イェ"),
        ("wO", "ウォ"),
    ],
)
def test_mora_to_text_converts_voiced_and_unvoiced_moras(
    mora: str, expected: str
) -> None:
    assert mora_to_text(mora) == expected


@pytest.mark.parametrize("invalid", ["x", ""])
def test_mora_to_text_returns_unknown_mora_without_raising(invalid: str) -> None:
    assert mora_to_text(invalid) == invalid
