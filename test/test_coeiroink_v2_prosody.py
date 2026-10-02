import pytest

from voicevox_engine.coeiroink_v2.prosody import (
    ProsodyError,
    estimate_prosody,
    estimate_prosody_from_kana,
)

REFERENCE = {
    "plain": [
        "^",
        "k",
        "o",
        "[",
        "r",
        "e",
        "w",
        "a",
        "#",
        "o",
        "[",
        "N",
        "s",
        "e",
        "e",
        "g",
        "o",
        "]",
        "o",
        "s",
        "e",
        "e",
        "n",
        "o",
        "#",
        "t",
        "e",
        "]",
        "s",
        "u",
        "t",
        "o",
        "d",
        "e",
        "s",
        "u",
        "$",
    ],
    "detail": [
        [
            {"phoneme": "k-o", "hira": "こ", "accent": 0},
            {"phoneme": "r-e", "hira": "れ", "accent": 1},
            {"phoneme": "w-a", "hira": "わ", "accent": 1},
        ],
        [
            {"phoneme": "o", "hira": "お", "accent": 0},
            {"phoneme": "N", "hira": "ん", "accent": 1},
            {"phoneme": "s-e", "hira": "せ", "accent": 1},
            {"phoneme": "e", "hira": "え", "accent": 1},
            {"phoneme": "g-o", "hira": "ご", "accent": 1},
            {"phoneme": "o", "hira": "お", "accent": 0},
            {"phoneme": "s-e", "hira": "せ", "accent": 0},
            {"phoneme": "e", "hira": "え", "accent": 0},
            {"phoneme": "n-o", "hira": "の", "accent": 0},
        ],
        [
            {"phoneme": "t-e", "hira": "て", "accent": 1},
            {"phoneme": "s-u", "hira": "す", "accent": 0},
            {"phoneme": "t-o", "hira": "と", "accent": 0},
            {"phoneme": "d-e", "hira": "で", "accent": 0},
            {"phoneme": "s-u", "hira": "す", "accent": 0},
        ],
    ],
}


def _wire(result):
    return result.model_dump()


def test_estimate_prosody_matches_saved_black_box_reference():
    result = estimate_prosody("これは音声合成のテストです。")

    assert _wire(result) == REFERENCE


def test_estimate_prosody_from_kana_matches_text_analysis():
    kana = "コレワ'、オンセエゴ'オセエノ、テ'_ストデ_ス"
    result = _wire(estimate_prosody_from_kana(kana))

    assert result["detail"] == [
        REFERENCE["detail"][0],
        [{"phoneme": "_", "hira": "、", "accent": 0}],
        REFERENCE["detail"][1],
        [{"phoneme": "_", "hira": "、", "accent": 0}],
        REFERENCE["detail"][2],
    ]
    assert result["plain"] == [
        "_" if token == "#" else token for token in REFERENCE["plain"]
    ]


def test_kana_preserves_pause_and_interrogative_tokens():
    result = estimate_prosody_from_kana("ア'、カ'？")

    assert result.plain == ["^", "a", "_", "k", "a", "?"]
    assert result.detail == [
        [{"phoneme": "a", "hira": "あ", "accent": 1}],
        [{"phoneme": "_", "hira": "、", "accent": 0}],
        [{"phoneme": "k-a", "hira": "か", "accent": 1}],
        [{"phoneme": "?", "hira": "？", "accent": 0}],
    ]


def test_results_are_deterministic():
    first = _wire(estimate_prosody("これは音声合成のテストです。"))
    second = _wire(estimate_prosody("これは音声合成のテストです。"))

    assert first == second


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: estimate_prosody(None), "text must be a string"),
        (lambda: estimate_prosody_from_kana(None), "kana must be a string"),
        (
            lambda: estimate_prosody_from_kana("ア"),
            "invalid kana: ACCENT_NOTFOUND",
        ),
    ],
)
def test_invalid_input_has_stable_error(call, message):
    with pytest.raises(ProsodyError, match=message):
        call()
