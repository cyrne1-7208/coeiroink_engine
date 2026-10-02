import base64
import io

import numpy as np
import pytest
import soundfile

from voicevox_engine.coeiroink_v2.audio import (
    MAX_PAUSE_LENGTH_SECONDS,
    MAX_SAMPLING_RATE,
    AudioValidationError,
    apply_trim_buffer,
    decode_pcm_wav,
    decode_pcm_wav_base64,
    encode_pcm_wav,
    encode_pcm_wav_base64,
    estimate_world_f0,
    pitch_shift_resampling,
    prepare_world_f0,
    process_wav,
    process_wave,
    replace_pause_segments,
)


def _sine(seconds=0.25, sampling_rate=16000, frequency=440.0):
    count = int(seconds * sampling_rate)
    time = np.arange(count, dtype=np.float32) / sampling_rate
    return (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _stereo_pcm_wav():
    wave = np.column_stack((_sine(), _sine(frequency=660.0)))
    output = io.BytesIO()
    soundfile.write(output, wave, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _mora_duration(mora, start, end):
    return {
        "mora": mora,
        "hira": "",
        "phonemePitches": [],
        "wavRange": {"start": start, "end": end},
    }


def test_pcm_wav_round_trip_has_mono_pcm_header_and_sampling_rate():
    wave = _sine()

    encoded = encode_pcm_wav(wave, 16000)
    decoded, sampling_rate = decode_pcm_wav(encoded, expected_sampling_rate=16000)

    assert encoded[:4] == b"RIFF"
    assert encoded[8:12] == b"WAVE"
    assert encoded[20:22] == b"\x01\x00"  # PCM format tag
    assert encoded[22:24] == b"\x01\x00"  # one channel
    assert sampling_rate == 16000
    assert decoded.dtype == np.float32
    assert decoded.ndim == 1
    np.testing.assert_allclose(decoded, wave, atol=1 / 32768)


def test_base64_round_trip_is_stable_and_validated():
    wave = _sine(seconds=0.1)

    encoded = encode_pcm_wav_base64(wave, 16000)
    decoded, sampling_rate = decode_pcm_wav_base64(encoded)

    assert encoded == base64.standard_b64encode(encode_pcm_wav(wave, 16000)).decode(
        "ascii"
    )
    assert sampling_rate == 16000
    np.testing.assert_allclose(decoded, wave, atol=1 / 32768)

    with pytest.raises(AudioValidationError, match="valid Base64"):
        decode_pcm_wav_base64(encoded[:-1] + "!")


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros((10, 1), dtype=np.float32), "mono 1-D"),
        (np.zeros(10, dtype=np.int16), "floating-point"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
    ],
)
def test_encoder_rejects_invalid_waveforms(value, message):
    with pytest.raises(AudioValidationError, match=message):
        encode_pcm_wav(value, 16000)


def test_decoder_rejects_stereo_rate_mismatch_and_non_wav():
    with pytest.raises(AudioValidationError, match="mono"):
        decode_pcm_wav(_stereo_pcm_wav())
    with pytest.raises(AudioValidationError, match="unexpected sampling rate"):
        decode_pcm_wav(encode_pcm_wav(_sine(), 16000), expected_sampling_rate=44100)
    with pytest.raises(AudioValidationError, match="RIFF/WAVE"):
        decode_pcm_wav(b"not-a-wav")


def test_trim_buffer_retains_context_outside_detected_sound():
    sampling_rate = 16000
    sound = _sine(seconds=0.2, sampling_rate=sampling_rate)
    wave = np.concatenate(
        [
            np.zeros(4000, dtype=np.float32),
            sound,
            np.zeros(4000, dtype=np.float32),
        ]
    )

    unbuffered = apply_trim_buffer(wave, sampling_rate)
    buffered = apply_trim_buffer(
        wave,
        sampling_rate,
        start_trim_buffer=0.05,
        end_trim_buffer=0.1,
    )

    assert buffered.size == unbuffered.size + 800 + 1600
    np.testing.assert_array_equal(buffered[800:-1600], unbuffered)


def test_legacy_process_wave_delegates_to_current_processing():
    wave = _sine(seconds=0.2, sampling_rate=16000)

    identity, identity_rate = process_wave(
        wave,
        16000,
        output_sampling_rate=16000,
    )
    np.testing.assert_array_equal(identity, wave)
    assert identity_rate == 16000

    processed, processed_rate = process_wave(
        wave,
        16000,
        volume_scale=0.5,
        pre_phoneme_length=0.01,
        post_phoneme_length=0.02,
        output_sampling_rate=8000,
    )
    assert processed_rate == 8000
    assert processed.size == int(0.2 * 8000) + 80 + 160
    assert np.isfinite(processed).all()


def test_legacy_pitch_and_intonation_processing_is_deterministic():
    wave = _sine(seconds=0.4, sampling_rate=16000)

    first, first_rate = process_wave(
        wave,
        16000,
        pitch_scale=0.1,
        intonation_scale=0.8,
    )
    second, second_rate = process_wave(
        wave,
        16000,
        pitch_scale=0.1,
        intonation_scale=0.8,
    )

    assert first_rate == second_rate == 16000
    assert first.size > 0
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_legacy_process_wav_keeps_pcm_contract():
    source = encode_pcm_wav(_sine(seconds=0.1, sampling_rate=16000), 16000)

    result = process_wav(source, sampling_rate=16000, volume_scale=0.5)
    wave, sampling_rate = decode_pcm_wav(result)

    assert sampling_rate == 16000
    assert wave.size == 1600
    assert np.isfinite(wave).all()


@pytest.mark.parametrize(
    "sampling_rate, processing, message",
    [
        (0, {}, "sampling_rate"),
        (
            16000,
            {"output_sampling_rate": MAX_SAMPLING_RATE + 1},
            "output_sampling_rate",
        ),
        (16000, {"pre_phoneme_length": -0.001}, "pre_phoneme_length"),
        (16000, {"post_phoneme_length": -0.001}, "post_phoneme_length"),
        (16000, {"volume_scale": -0.001}, "volume_scale"),
        (16000, {"intonation_scale": -0.001}, "intonation_scale"),
    ],
)
def test_legacy_process_wave_rejects_invalid_values(sampling_rate, processing, message):
    with pytest.raises(AudioValidationError, match=message):
        process_wave(
            _sine(seconds=0.02),
            sampling_rate,
            **processing,
        )


def test_resampling_pitch_shift_preserves_length_and_changes_pitch():
    wave = _sine(seconds=0.4, sampling_rate=16000, frequency=220.0)

    first = pitch_shift_resampling(wave, 16000, pitch_scale=1.0)
    second = pitch_shift_resampling(wave, 16000, pitch_scale=1.0)

    assert first.shape == wave.shape
    assert np.isfinite(first).all()
    assert not np.array_equal(first, wave)
    np.testing.assert_array_equal(first, second)


def test_internal_pause_is_replaced_and_edge_pauses_are_untouched():
    wave = np.arange(20, dtype=np.float32) / 20.0
    durations = [
        {
            "mora": mora,
            "hira": "",
            "phonemePitches": [],
            "wavRange": {"start": start, "end": end},
        }
        for mora, start, end in (
            ("pau", 0, 2),
            ("a", 2, 6),
            ("pau", 6, 10),
            ("i", 10, 14),
            ("pau", 14, 20),
        )
    ]

    result = replace_pause_segments(
        wave,
        sampling_rate=10,
        mora_durations=durations,
        pause_length=0.3,
        pause_start_trim_buffer=0.1,
        pause_end_trim_buffer=0.1,
    )

    expected = np.concatenate((wave[:7], np.zeros(3, dtype=np.float32), wave[9:]))
    np.testing.assert_array_equal(result, expected)


def test_pause_replacement_rejects_an_oversized_pause_length():
    durations = [
        _mora_duration("a", 0, 2),
        _mora_duration("pau", 2, 4),
        _mora_duration("a", 4, 8),
    ]

    with pytest.raises(AudioValidationError, match="no greater"):
        replace_pause_segments(
            np.ones(8, dtype=np.float32),
            sampling_rate=16000,
            mora_durations=durations,
            pause_length=MAX_PAUSE_LENGTH_SECONDS + 0.001,
        )


def test_pause_replacement_rejects_a_zero_length_internal_pau():
    durations = [
        _mora_duration("a", 0, 2),
        _mora_duration("pau", 2, 2),
        _mora_duration("a", 2, 8),
    ]

    with pytest.raises(AudioValidationError, match="non-empty"):
        replace_pause_segments(
            np.ones(8, dtype=np.float32),
            sampling_rate=16000,
            mora_durations=durations,
            pause_length=0.3,
        )


def test_world_f0_response_is_finite_and_keeps_mora_durations():
    wave = _sine(seconds=0.4, sampling_rate=16000)

    f0 = estimate_world_f0(wave, 16000)
    response = prepare_world_f0(wave, 16000, [])

    assert f0.ndim == 1
    assert f0.size > 0
    assert np.isfinite(f0).all()
    assert response.mora_durations == []
    assert response.f0 == [float(value) for value in f0]
