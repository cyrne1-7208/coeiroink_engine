# Base64エンコードされた複数WAVの連結・変換・入力上限チェックのテスト

from __future__ import annotations

import base64
import importlib
import io

import numpy as np
import pytest
import soundfile
from scipy.signal import resample

from voicevox_engine.utility import ConnectBase64WavesException, connect_base64_waves

connect_module = importlib.import_module("voicevox_engine.utility.connect_base64_waves")


def _sine(seconds: float, samplerate: int, frequency: float) -> np.ndarray:
    axis = np.linspace(0, seconds, int(seconds * samplerate), endpoint=False)
    return np.sin(2 * np.pi * frequency * axis).astype(np.float32)


def _encode_wav_bytes(wave: np.ndarray, samplerate: int) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(
        file=buffer,
        data=wave,
        samplerate=samplerate,
        format="WAV",
        subtype="FLOAT",
    )
    return buffer.getvalue()


def _encode_base64(wave_bytes: bytes) -> str:
    return base64.standard_b64encode(wave_bytes).decode("utf-8")


def _sine_base64(seconds: float, samplerate: int, frequency: float) -> str:
    return _encode_base64(
        _encode_wav_bytes(_sine(seconds, samplerate, frequency), samplerate)
    )


def test_connect_concatenates_identical_waveforms() -> None:
    samplerate = 1000
    wave = _sine(seconds=2, samplerate=samplerate, frequency=10)
    encoded = _encode_base64(_encode_wav_bytes(wave, samplerate))

    connected, _ = connect_base64_waves(waves=[encoded, encoded])

    np.testing.assert_array_equal(connected, np.concatenate([wave, wave]))


def test_empty_input_raises() -> None:
    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves(waves=[])


def test_invalid_base64_raises() -> None:
    broken = _sine_base64(seconds=2, samplerate=1000, frequency=10)[1:]

    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves(waves=[broken])


def test_invalid_wave_bytes_raises() -> None:
    broken_bytes = _encode_wav_bytes(
        _sine(seconds=2, samplerate=1000, frequency=10), 1000
    )[1:]

    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves(waves=[_encode_base64(broken_bytes)])


def test_different_sampling_rates_are_resampled_to_highest() -> None:
    wave_24k = _sine(seconds=1, samplerate=24000, frequency=10)
    wave_1k = _sine(seconds=2, samplerate=1000, frequency=10)
    encoded_24k = _encode_base64(_encode_wav_bytes(wave_24k, 24000))
    encoded_1k = _encode_base64(_encode_wav_bytes(wave_1k, 1000))

    resampled = resample(wave_1k, 24000 * len(wave_1k) // 1000)
    expected = np.concatenate([wave_24k, resampled])

    connected, _ = connect_base64_waves(waves=[encoded_24k, encoded_1k])

    assert connected.shape == expected.shape
    np.testing.assert_array_almost_equal(expected, connected)


def test_different_channel_counts_upmix_mono_to_stereo() -> None:
    wave_1ch = _sine(seconds=2, samplerate=1000, frequency=10)
    wave_2ch = np.array([wave_1ch, wave_1ch]).T
    encoded_1ch = _encode_base64(_encode_wav_bytes(wave_1ch, 1000))
    encoded_2ch = _encode_base64(_encode_wav_bytes(wave_2ch, 1000))

    connected, _ = connect_base64_waves(waves=[encoded_1ch, encoded_2ch])

    np.testing.assert_array_equal(connected, np.concatenate([wave_2ch, wave_2ch]))


def test_rejects_more_than_two_channels() -> None:
    encoded = _encode_base64(
        _encode_wav_bytes(np.zeros((8, 3), dtype=np.float32), 1000)
    )

    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves([encoded])


def test_rejects_aggregate_decoded_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _sine_base64(seconds=0.1, samplerate=1000, frequency=10)
    monkeypatch.setattr(connect_module, "MAX_CONNECTED_WAVE_BYTES", 8)

    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves([encoded])


def test_rejects_projected_resampling_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_rate = _sine_base64(seconds=1, samplerate=1000, frequency=10)
    high_rate = _sine_base64(seconds=0.1, samplerate=24000, frequency=10)
    monkeypatch.setattr(connect_module, "MAX_CONNECTED_WAVE_SAMPLES", 10_000)

    with pytest.raises(ConnectBase64WavesException):
        connect_base64_waves([low_rate, high_rate])
