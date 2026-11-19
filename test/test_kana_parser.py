"""AquesTalk風記法（kana）のパース・生成・エラーコードを検証する。"""

from __future__ import annotations

import pytest

from voicevox_engine.kana_parser import create_kana, parse_kana
from voicevox_engine.model import AccentPhrase, Mora, ParseKanaError, ParseKanaErrorCode


def _mora(
    text: str,
    vowel: str,
    *,
    consonant: str | None = None,
    consonant_length: float | None = None,
    vowel_length: float = 0.0,
    pitch: float = 0.0,
) -> Mora:
    if consonant is not None and consonant_length is None:
        consonant_length = 0.0
    return Mora(
        text=text,
        consonant=consonant,
        consonant_length=consonant_length,
        vowel=vowel,
        vowel_length=vowel_length,
        pitch=pitch,
    )


def _pause_mora() -> Mora:
    return _mora("、", "pau")


def _phrase(
    moras: list[Mora],
    *,
    accent: int = 1,
    pause: bool = False,
    interrogative: bool = False,
) -> AccentPhrase:
    return AccentPhrase(
        moras=moras,
        accent=accent,
        pause_mora=_pause_mora() if pause else None,
        is_interrogative=interrogative,
    )


@pytest.mark.parametrize(
    ("text", "expected_accent"),
    [
        ("シャ'シシュシェショ", 1),
        ("シャ'_シシュシェショ", 1),
        ("シャシ'シュシェショ", 2),
        ("シャ_シ'シュシェショ", 2),
        ("シャシシュ'シェショ", 3),
        ("シャ_シシュ'シェショ", 3),
        ("シャシシュシェショ'", 5),
        ("シャ_シシュシェショ'", 5),
    ],
)
def test_accent_and_mora_length(text: str, expected_accent: int) -> None:
    phrase = parse_kana(text)[0]
    assert phrase.accent == expected_accent
    assert len(phrase.moras) == 5


@pytest.mark.parametrize(
    "text",
    [
        "コンニチワ'",
        "ワタシワ'/シャチョオデ'_ス",
        "トテモ'、エラ'インデス",
        "ヲ'",
        "ェ'",
    ],
)
def test_roundtrip(text: str) -> None:
    assert create_kana(parse_kana(text)) == text


def test_extended_foreign_moras_are_single_moras() -> None:
    expected = [
        "ウゥ",
        "キィ",
        "ギィ",
        "クァ",
        "クィ",
        "クゥ",
        "クェ",
        "クォ",
        "グァ",
        "グィ",
        "グゥ",
        "グェ",
        "グォ",
        "ヂェ",
        "ヂャ",
        "ヂュ",
        "ヂョ",
        "テェ",
        "ニィ",
        "ヒィ",
        "ビィ",
        "ピィ",
        "ミィ",
        "リィ",
    ]
    phrase = parse_kana("".join(expected) + "'")[0]

    assert [mora.text for mora in phrase.moras] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ア'/ア'", [_phrase([_mora("ア", "a")]), _phrase([_mora("ア", "a")])]),
        (
            "ア'、ア'",
            [_phrase([_mora("ア", "a")], pause=True), _phrase([_mora("ア", "a")])],
        ),
        ("ア'/ア'/ア'/ア'/ア'", [_phrase([_mora("ア", "a")]) for _ in range(5)]),
        ("ス'", [_phrase([_mora("ス", "u", consonant="s")])]),
        ("_ス'", [_phrase([_mora("ス", "U", consonant="s")])]),
        ("ギェ'", [_phrase([_mora("ギェ", "e", consonant="gy")])]),
        (
            "ギェ'、ギェ'/ギェ'",
            [
                _phrase([_mora("ギェ", "e", consonant="gy")], pause=True),
                _phrase([_mora("ギェ", "e", consonant="gy")]),
                _phrase([_mora("ギェ", "e", consonant="gy")]),
            ],
        ),
        ("ア'？", [_phrase([_mora("ア", "a")], interrogative=True)]),
        (
            "ギェ'、ギェ'/ギェ'？",
            [
                _phrase([_mora("ギェ", "e", consonant="gy")], pause=True),
                _phrase([_mora("ギェ", "e", consonant="gy")]),
                _phrase([_mora("ギェ", "e", consonant="gy")], interrogative=True),
            ],
        ),
        (
            "ア'、ア'？、ア'？/ア'？",
            [
                _phrase([_mora("ア", "a")], pause=True),
                _phrase([_mora("ア", "a")], pause=True, interrogative=True),
                _phrase([_mora("ア", "a")], interrogative=True),
                _phrase([_mora("ア", "a")], interrogative=True),
            ],
        ),
    ],
)
def test_accent_phrase_marks(text: str, expected: list[AccentPhrase]) -> None:
    assert parse_kana(text) == expected


@pytest.mark.parametrize(
    ("kana", "code", "expected_kwargs"),
    [
        ("アクセント", ParseKanaErrorCode.ACCENT_NOTFOUND, None),
        ("'アクセント", ParseKanaErrorCode.ACCENT_TOP, None),
        ("ア'ク'セント", ParseKanaErrorCode.ACCENT_TWICE, None),
        ("ひ'らがな", ParseKanaErrorCode.UNKNOWN_TEXT, None),
        ("__ス'", ParseKanaErrorCode.UNKNOWN_TEXT, None),
        ("ア'/", ParseKanaErrorCode.EMPTY_PHRASE, {"position": "2"}),
        ("/ア'", ParseKanaErrorCode.EMPTY_PHRASE, None),
        ("", ParseKanaErrorCode.EMPTY_PHRASE, None),
        (
            "ヒト'ツメ/フタツメ",
            ParseKanaErrorCode.ACCENT_NOTFOUND,
            {"text": "フタツメ"},
        ),
        ("ア？ア'", ParseKanaErrorCode.INTERROGATION_MARK_NOT_AT_END, None),
    ],
)
def test_parse_kana_rejects_invalid_syntax(
    kana: str,
    code: ParseKanaErrorCode,
    expected_kwargs: dict[str, str] | None,
) -> None:
    with pytest.raises(ParseKanaError) as raised:
        parse_kana(kana)

    assert raised.value.errcode == code
    if expected_kwargs is not None:
        assert raised.value.kwargs == expected_kwargs


def test_create_kana_handles_interrogative_and_devoiced_moras() -> None:
    def styled_mora(text: str, vowel: str, consonant: str | None = None) -> Mora:
        return _mora(
            text,
            vowel,
            consonant=consonant,
            consonant_length=2.5 if consonant is not None else None,
            vowel_length=2.5,
            pitch=2.5,
        )

    koreha_arimasuka = [
        _phrase(
            [
                styled_mora("コ", "o", "k"),
                styled_mora("レ", "e", "r"),
                styled_mora("ワ", "a", "w"),
            ],
            accent=3,
        ),
        _phrase(
            [
                styled_mora("ア", "a"),
                styled_mora("リ", "i", "r"),
                styled_mora("マ", "a", "m"),
                styled_mora("ス", "U", "s"),
                styled_mora("カ", "a", "k"),
            ],
            accent=3,
        ),
    ]
    assert create_kana(koreha_arimasuka) == "コレワ'/アリマ'_スカ"
    koreha_arimasuka[-1].is_interrogative = True
    assert create_kana(koreha_arimasuka) == "コレワ'/アリマ'_スカ？"

    kya = [
        _phrase(
            [
                styled_mora("キャ", "a", "ky"),
                _mora("ッ", "cl", vowel_length=0.1, pitch=0.0),
            ],
            accent=1,
        )
    ]
    assert create_kana(kya) == "キャ'ッ"
    kya[-1].is_interrogative = True
    assert create_kana(kya) == "キャ'ッ？"
