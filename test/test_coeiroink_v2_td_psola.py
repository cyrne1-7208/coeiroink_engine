import numpy as np
import pytest

from voicevox_engine.coeiroink_v2.td_psola import (
    PROCESSING_ALGORITHM,
    TDPSOLAValidationError,
    process_td_psola,
)


def _sine(
    frequency=220.0,
    seconds=0.5,
    sampling_rate=16000,
    amplitude=0.25,
):
    count = int(seconds * sampling_rate)
    time = np.arange(count, dtype=np.float64) / sampling_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _chirp(seconds=0.5, sampling_rate=16000):
    count = int(seconds * sampling_rate)
    time = np.arange(count, dtype=np.float64) / sampling_rate
    phase = 2.0 * np.pi * (180.0 * time + 100.0 * time * time)
    envelope = np.minimum(1.0, np.minimum(time * 40.0, (seconds - time) * 40.0))
    return (0.2 * envelope * np.sin(phase)).astype(np.float32)


def _speech_like(seconds=0.6, sampling_rate=16000):
    count = int(seconds * sampling_rate)
    time = np.arange(count, dtype=np.float64) / sampling_rate
    voiced = (time < 0.22) | ((time >= 0.32) & (time < 0.55))
    f0 = 155.0 + 20.0 * np.sin(2.0 * np.pi * 2.0 * time)
    phase = 2.0 * np.pi * (155.0 * time + 10.0 * time * time)
    harmonic = np.sin(phase) + 0.35 * np.sin(2.0 * phase)
    noise = 0.025 * np.sin(2.0 * np.pi * 3100.0 * time)
    wave = np.where(voiced, 0.18 * harmonic, noise)
    return wave.astype(np.float32), np.where(voiced, f0, 0.0).astype(np.float32)


def _dominant_frequency(wave, sampling_rate, start_fraction=0.2, end_fraction=0.8):
    start = int(wave.size * start_fraction)
    end = int(wave.size * end_fraction)
    segment = wave[start:end].astype(np.float64)
    segment = segment - np.mean(segment)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    frequencies = np.fft.rfftfreq(segment.size, 1.0 / sampling_rate)
    usable = (frequencies >= 50.0) & (frequencies <= sampling_rate / 2.0)
    return float(frequencies[usable][np.argmax(spectrum[usable])])


def test_processing_algorithm_name_and_identity_are_stable():
    wave = _sine()

    assert PROCESSING_ALGORITHM == "td-psola"
    result = process_td_psola(
        wave,
        16000,
        pitch_scale=0.0,
        intonation_scale=1.0,
        f0=np.full(wave.size, 220.0, dtype=np.float32),
    )

    assert result.dtype == np.float32
    assert result.shape == wave.shape
    np.testing.assert_array_equal(result, wave)


def test_sine_pitch_shift_changes_pitch_without_changing_duration():
    sampling_rate = 16000
    wave = _sine(frequency=220.0, sampling_rate=sampling_rate)
    result = process_td_psola(
        wave,
        sampling_rate,
        pitch_scale=1.0,
        f0=np.full(wave.size, 220.0, dtype=np.float32),
    )

    frequency = _dominant_frequency(result, sampling_rate)
    assert result.size == wave.size
    assert np.isfinite(result).all()
    assert abs(frequency - 440.0) <= 18.0


def test_adjusted_target_f0_is_applied_without_world_resynthesis():
    sampling_rate = 16000
    wave = _sine(frequency=220.0, sampling_rate=sampling_rate)
    result = process_td_psola(
        wave,
        sampling_rate,
        f0=np.full(wave.size, 220.0, dtype=np.float32),
        target_f0=np.full(wave.size, 330.0, dtype=np.float32),
    )

    frequency = _dominant_frequency(result, sampling_rate)
    assert result.size == wave.size
    assert np.isfinite(result).all()
    assert abs(frequency - 330.0) <= 18.0


def test_chirp_processing_is_deterministic_and_finite():
    sampling_rate = 16000
    wave = _chirp(sampling_rate=sampling_rate)
    f0 = np.linspace(180.0, 280.0, wave.size, dtype=np.float32)

    first = process_td_psola(
        wave,
        sampling_rate,
        pitch_scale=-0.25,
        intonation_scale=0.75,
        f0=f0,
    )
    second = process_td_psola(
        wave,
        sampling_rate,
        pitch_scale=-0.25,
        intonation_scale=0.75,
        f0=f0,
    )

    assert first.dtype == np.float32
    assert first.shape == wave.shape
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_speech_like_voiced_unvoiced_transitions_remain_finite_and_bounded():
    sampling_rate = 16000
    wave, f0 = _speech_like(sampling_rate=sampling_rate)
    result = process_td_psola(
        wave,
        sampling_rate,
        pitch_scale=0.35,
        intonation_scale=1.25,
        f0=f0,
    )

    assert result.shape == wave.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 2.0 * np.max(np.abs(wave)) + 1e-6
    # The gap is unvoiced and longer than a grain, so it must not be filled by
    # a voiced overlap-add operation.
    gap = slice(int(0.24 * sampling_rate), int(0.30 * sampling_rate))
    np.testing.assert_array_equal(result[gap], wave[gap])


@pytest.mark.parametrize(
    "wave, sampling_rate, pitch_scale, intonation_scale, message",
    [
        (np.zeros((10, 1), dtype=np.float32), 16000, 0.1, 1.0, "mono 1-D"),
        (np.zeros(10, dtype=np.int16), 16000, 0.1, 1.0, "floating-point"),
        (np.array([0.0, np.nan], dtype=np.float32), 16000, 0.1, 1.0, "finite"),
        (np.zeros(10, dtype=np.float32), 16000, np.nan, 1.0, "pitch_scale"),
        (np.zeros(10, dtype=np.float32), 16000, 0.1, np.inf, "intonation_scale"),
        (np.zeros(10, dtype=np.float32), 16000, 0.1, -1.0, "greater than or equal"),
    ],
)
def test_invalid_waveforms_and_scales_are_rejected(
    wave, sampling_rate, pitch_scale, intonation_scale, message
):
    with pytest.raises(TDPSOLAValidationError, match=message):
        process_td_psola(
            wave,
            sampling_rate,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            f0=np.ones(10, dtype=np.float32),
        )


def test_float64_wave_outside_float32_range_is_rejected_before_cast():
    wave = np.array([np.finfo(np.float64).max], dtype=np.float64)

    with pytest.raises(TDPSOLAValidationError, match="finite float32"):
        process_td_psola(
            wave,
            16000,
            pitch_scale=0.0,
            intonation_scale=1.0,
            f0=np.ones(1, dtype=np.float32),
        )


def test_invalid_f0_and_short_or_unvoiced_audio_are_safe():
    wave = _sine(seconds=0.2)
    with pytest.raises(TDPSOLAValidationError, match="negative"):
        process_td_psola(
            wave,
            16000,
            pitch_scale=0.2,
            f0=np.full(wave.size, -1.0, dtype=np.float32),
        )

    unvoiced = np.zeros_like(wave)
    unchanged = process_td_psola(
        unvoiced,
        16000,
        pitch_scale=0.4,
        f0=np.zeros(wave.size, dtype=np.float32),
    )
    np.testing.assert_array_equal(unchanged, unvoiced)

    short = np.array([0.1, -0.1], dtype=np.float32)
    np.testing.assert_array_equal(
        process_td_psola(short, 16000, pitch_scale=0.4), short
    )
