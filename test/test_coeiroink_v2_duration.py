import pytest

from voicevox_engine.coeiroink_v2.duration import (
    DurationConversionError,
    convert_duration,
)
from voicevox_engine.coeiroink_v2.models import Mora


def _mora(phoneme, hira):
    return Mora(phoneme=phoneme, hira=hira, accent=0)


def _assert_contiguous(durations, expected_end):
    cursor = 0
    for mora in durations:
        assert mora.wav_range.start == cursor
        assert mora.wav_range.end >= mora.wav_range.start
        assert mora.phoneme_pitches[0].wav_range.start == mora.wav_range.start
        for phoneme in mora.phoneme_pitches:
            assert phoneme.wav_range.start == cursor
            assert phoneme.wav_range.end >= phoneme.wav_range.start
            cursor = phoneme.wav_range.end
        assert mora.wav_range.end == cursor
    assert cursor == expected_end


def test_converts_phonemes_pauses_and_zero_width_markers():
    plain = ["^", "k", "o", "[", "N", "#", "s", "a", "_", "t", "o", "$"]
    detail = [
        [_mora("k-o", "こ"), _mora("N", "ん")],
        [_mora("s-a", "さ")],
        [_mora("t-o", "と")],
    ]
    frames = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    result = convert_duration(plain, detail, frames, hop_length=256)

    assert [(m.mora, m.hira) for m in result] == [
        ("pau", ""),
        ("k-o", "こ"),
        ("N", "ん"),
        ("s-a", "さ"),
        ("pau", ""),
        ("t-o", "と"),
        ("pau", ""),
    ]
    assert [(p.phoneme, p.wav_range.start, p.wav_range.end)
            for p in result[1].phoneme_pitches] == [
        ("k", 512, 1280),
        ("o", 1280, 2304),
    ]
    # '['と'#'は次の意味単位へ割り当て、Nが5+6フレーム、sが7+8フレームを受け取ります。
    assert result[2].wav_range == {"start": 2304, "end": 5120}
    assert result[3].wav_range == {"start": 5120, "end": 11264}
    assert result[4].wav_range == {"start": 11264, "end": 13824}
    assert result[-1].wav_range == {"start": 19712, "end": 23040}
    _assert_contiguous(result, sum(frames) * 256)


def test_supports_nasal_consonant_cluster_and_question_pause():
    plain = ["^", "sh", "i", "[", "N", "cl", "t", "a", "?"]
    detail = [[
        _mora("sh-i", "し"),
        _mora("N", "ん"),
        _mora("cl", "っ"),
        _mora("t-a", "た"),
    ]]
    frames = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    result = convert_duration(plain, detail, frames, hop_length=10)

    assert [m.mora for m in result] == ["pau", "sh-i", "N", "cl", "t-a", "pau"]
    assert [p.phoneme for p in result[1].phoneme_pitches] == ["sh", "i"]
    assert result[-1].phoneme_pitches[0].phoneme == "pau"
    _assert_contiguous(result, sum(frames) * 10)


def test_accepts_mapping_moras_without_mutating_inputs():
    plain = ["^", "a", "$"]
    detail = [[{"phoneme": "a", "hira": "あ", "accent": 1}]]
    frames = [4, 5, 6]

    result = convert_duration(plain, detail, frames, hop_length=2)

    assert result[1].model_dump(by_alias=True) == {
        "mora": "a",
        "hira": "あ",
        "phonemePitches": [
            {"phoneme": "a", "wavRange": {"start": 8, "end": 18}}
        ],
        "wavRange": {"start": 8, "end": 18},
    }
    assert detail == [[{"phoneme": "a", "hira": "あ", "accent": 1}]]
    assert frames == [4, 5, 6]


@pytest.mark.parametrize(
    "plain, detail, frames, hop_length, message",
    [
        (["^", "a", "$"], [[_mora("i", "い")]], [1, 2, 3], 1, "does not match"),
        (["^", "a", "#"], [[_mora("a", "あ")]], [1, 2, 3], 1, "unallocated"),
        (["^", "a", "$"], [[_mora("a", "あ")]], [1, 2], 1, "same length"),
        (["^", "a", "$"], [[_mora("a", "あ")]], [1, 2, 3], 0, "hop_length"),
    ],
)
def test_rejects_misaligned_or_invalid_input(
    plain, detail, frames, hop_length, message
):
    with pytest.raises(DurationConversionError, match=message):
        convert_duration(plain, detail, frames, hop_length)


def test_zero_duration_markers_still_produce_contiguous_ranges():
    result = convert_duration(
        ["^", "[", "a", "]", "$"],
        [[_mora("a", "あ")]],
        [0, 0, 3, 0, 0],
        hop_length=256,
    )

    _assert_contiguous(result, 3 * 256)
    assert result[0].wav_range == {"start": 0, "end": 0}
    assert result[1].wav_range == {"start": 0, "end": 768}
    assert result[2].wav_range == {"start": 768, "end": 768}
