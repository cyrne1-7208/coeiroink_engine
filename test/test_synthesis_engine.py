"""旧Core互換経路の入力変換を、手計算できる短いクエリで検証する。"""

from unittest.mock import Mock

import numpy as np
import pytest

from voicevox_engine.model import AccentPhrase, AudioQuery, Mora
from voicevox_engine.synthesis_engine import SynthesisEngine


@pytest.fixture
def query():
    frame = 256 / 24000
    return AudioQuery(
        accent_phrases=[
            AccentPhrase(
                moras=[
                    Mora(
                        text="カ",
                        consonant="k",
                        consonant_length=2 * frame,
                        vowel="a",
                        vowel_length=4 * frame,
                        pitch=5,
                    ),
                    Mora(text="イ", vowel="i", vowel_length=4 * frame, pitch=6),
                ],
                accent=1,
                pause_mora=Mora(
                    text="、", vowel="pau", vowel_length=2 * frame, pitch=0
                ),
            ),
            AccentPhrase(
                moras=[
                    Mora(
                        text="ス",
                        consonant="s",
                        consonant_length=2 * frame,
                        vowel="U",
                        vowel_length=4 * frame,
                        pitch=0,
                    )
                ],
                accent=1,
            ),
        ],
        speedScale=1,
        pitchScale=0,
        intonationScale=1,
        volumeScale=1,
        prePhonemeLength=2 * frame,
        postPhonemeLength=2 * frame,
        outputSamplingRate=24000,
        outputStereo=False,
    )


@pytest.fixture
def engine():
    core = Mock()
    core.metas.return_value = "[]"
    core.supported_devices.return_value = "{}"
    core.is_model_loaded.return_value = True
    core.decode_forward.side_effect = lambda **kwargs: np.full(
        kwargs["length"] * 256, 0.25
    )
    return SynthesisEngine(core=core)


def test_phoneme_lengths_are_assigned_to_consonants_vowels_and_pauses(engine, query):
    engine.core.yukarin_s_forward.return_value = np.array(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    )
    first, second = engine.replace_phoneme_length(query.accent_phrases, speaker_id=1)
    assert [(m.consonant_length, m.vowel_length) for m in first.moras] == [
        (0.03, 0.04),
        (None, 0.05),
    ]
    assert first.pause_mora.vowel_length == 0.06
    assert (second.moras[0].consonant_length, second.moras[0].vowel_length) == (
        0.07,
        0.08,
    )
    assert engine.core.yukarin_s_forward.call_args.kwargs["speaker_id"].item() == 1


def test_accent_flags_and_unvoiced_pitch_are_passed_to_legacy_core(engine, query):
    engine.core.yukarin_sa_forward.return_value = np.array(
        [[10.0, 5.0, 6.0, 9.0, 8.0, 10.0]]
    )
    first, second = engine.replace_mora_pitch(query.accent_phrases, speaker_id=1)
    args = engine.core.yukarin_sa_forward.call_args.kwargs
    np.testing.assert_array_equal(args["vowel_phoneme_list"], [[0, 7, 21, 0, 6, 0]])
    np.testing.assert_array_equal(
        args["consonant_phoneme_list"], [[-1, 23, -1, -1, 35, -1]]
    )
    np.testing.assert_array_equal(args["start_accent_list"], [[0, 1, 0, 0, 1, 0]])
    np.testing.assert_array_equal(args["end_accent_list"], [[0, 1, 0, 0, 1, 0]])
    np.testing.assert_array_equal(
        args["start_accent_phrase_list"], [[0, 1, 0, 0, 1, 0]]
    )
    np.testing.assert_array_equal(args["end_accent_phrase_list"], [[0, 0, 1, 0, 1, 0]])
    assert [m.pitch for m in first.moras] == [5, 6]
    assert first.pause_mora.pitch == second.moras[0].pitch == 0


def test_synthesis_maps_every_frame_and_preserves_query(engine, query):
    original = query.model_dump()
    wave = engine.synthesis(query, speaker_id=1)
    args = engine.core.decode_forward.call_args.kwargs
    # 先頭無音・k・a・i・休止・s・U・末尾無音。境界を含む全フレームを照合する。
    np.testing.assert_array_equal(
        args["phoneme"].argmax(axis=1),
        [0, 0, 23, 23, 7, 7, 7, 7, 21, 21, 21, 21, 0, 0, 35, 35, 6, 6, 6, 6, 0, 0],
    )
    np.testing.assert_array_equal(args["phoneme"].sum(axis=1), np.ones(22))
    np.testing.assert_array_equal(
        args["f0"].ravel(),
        [0, 0, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert args["speaker_id"].item() == 1
    assert wave.shape == (22 * 256,)
    assert query.model_dump() == original


def test_speed_and_pitch_controls_reach_decoder(engine, query):
    query.speedScale = 2
    query.pitchScale = 1
    query.intonationScale = 0.5
    engine.synthesis(query, speaker_id=1)
    args = engine.core.decode_forward.call_args.kwargs
    np.testing.assert_array_equal(
        args["phoneme"].argmax(axis=1), [0, 23, 7, 7, 21, 21, 0, 35, 6, 6, 0]
    )
    np.testing.assert_array_equal(
        args["f0"].ravel(), [0, 10.5, 10.5, 10.5, 11.5, 11.5, 0, 0, 0, 0, 0]
    )


def test_volume_resampling_and_stereo_are_applied_to_decoded_wave(engine, query):
    query.volumeScale = 0.5
    query.outputSamplingRate = 48000
    query.outputStereo = True
    wave = engine.synthesis(query, speaker_id=1)
    assert wave.shape == (44 * 256, 2)
    np.testing.assert_allclose(wave, 0.125)


def test_unvoiced_query_does_not_compute_an_empty_pitch_mean(engine, query):
    query.accent_phrases = [query.accent_phrases[-1]]
    # RuntimeWarningを隠さず、無声音だけのクエリをそのまま合成できることを確認する。
    with np.errstate(invalid="raise", divide="raise"):
        engine.synthesis(query, speaker_id=1)
    assert not engine.core.decode_forward.call_args.kwargs["f0"].any()


@pytest.mark.parametrize(
    "enabled, vowel, pitch, expected_count",
    [
        (False, "a", 5, 1),
        (True, "a", 5, 2),
        (True, "U", 5, 2),
        (True, "cl", 0, 1),
    ],
)
def test_interrogative_adjustment_is_optional_and_keeps_input(
    engine, query, enabled, vowel, pitch, expected_count
):
    query.accent_phrases = [
        AccentPhrase(
            moras=[Mora(text="テスト", vowel=vowel, vowel_length=0.1, pitch=pitch)],
            accent=1,
            is_interrogative=True,
        )
    ]
    original = query.model_dump()
    engine._synthesis_impl = Mock()
    engine.synthesis(query, speaker_id=1, enable_interrogative_upspeak=enabled)
    adjusted = engine._synthesis_impl.call_args.args[0].accent_phrases[0].moras
    assert len(adjusted) == expected_count
    if expected_count == 2:
        assert adjusted[-1].vowel == vowel
        assert adjusted[-1].consonant is None
        assert adjusted[-1].vowel_length == 0.15
        assert adjusted[-1].pitch == 5.3
    assert query.model_dump() == original
