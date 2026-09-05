"""Small, validated audio primitives for the COEIROINK v2 HTTP API.

This module deliberately contains no model or HTTP code.  Waveforms remain
one-dimensional floating-point NumPy arrays until they cross the HTTP
boundary, where they are encoded as mono PCM16 WAV data.  Numerical
post-processing delegates to the public Core ``AudioManager`` implementation
so the Engine and Core keep the same audio rules.
"""

import base64
import binascii
import io
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import soundfile

from .models import MoraDuration, WorldF0


class AudioValidationError(ValueError):
    """Raised when an audio value cannot be used by the v2 API."""


class AudioProcessingError(RuntimeError):
    """Raised when a public audio primitive cannot produce finite samples."""


Waveform = np.ndarray
WavInput = bytes | bytearray | memoryview
MAX_SAMPLING_RATE = 384000
MAX_PAUSE_LENGTH_SECONDS = 60.0
MAX_GENERATED_WAVE_SAMPLES = 30000000


def _require_sampling_rate(sampling_rate: object, name: str) -> int:
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, int)
        or sampling_rate <= 0
        or sampling_rate > MAX_SAMPLING_RATE
    ):
        raise AudioValidationError(
            f"{name} must be a positive integer no greater than {MAX_SAMPLING_RATE}"
        )
    return sampling_rate


def _require_waveform(wave: object, name: str = "wave") -> Waveform:
    if not isinstance(wave, np.ndarray):
        raise AudioValidationError(f"{name} must be a NumPy ndarray")
    if wave.ndim != 1:
        raise AudioValidationError(f"{name} must be a mono 1-D array")
    if not np.issubdtype(wave.dtype, np.floating):
        raise AudioValidationError(f"{name} must have a floating-point dtype")
    if wave.size == 0:
        raise AudioValidationError(f"{name} must not be empty")
    if not np.isfinite(wave).all():
        raise AudioValidationError(f"{name} must contain only finite samples")
    converted = wave.astype(np.float32, copy=False)
    if not np.isfinite(converted).all():
        raise AudioValidationError(
            f"{name} cannot be represented as finite float32 samples"
        )
    return converted


def _require_bounded_waveform_size(size: int, name: str = "wave") -> None:
    if size <= 0 or size > MAX_GENERATED_WAVE_SAMPLES:
        raise AudioValidationError(
            f"{name} must contain no more than {MAX_GENERATED_WAVE_SAMPLES} samples"
        )


def _validate_scale(
    value: object,
    name: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        if minimum is None:
            requirement = "a finite number"
        else:
            requirement = f"a finite number greater than or equal to {minimum}"
        raise AudioValidationError(f"{name} must be {requirement}")
    return float(value)


def _validate_duration(
    value: object,
    name: str,
    maximum: float | None = None,
) -> float:
    result = _validate_scale(value, name, minimum=0.0)
    if maximum is not None and result > maximum:
        raise AudioValidationError(f"{name} must be no greater than {maximum}")
    return result


def encode_pcm_wav(wave: np.ndarray, sampling_rate: int) -> bytes:
    """Encode a finite mono float waveform as deterministic PCM16 WAV bytes."""

    wave = _require_waveform(wave)
    _require_bounded_waveform_size(wave.size)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")

    # PCMの表現範囲へここでクリップし、後処理のゲインによる範囲外サンプルも決定的に変換する。
    pcm_ready = np.clip(wave, -1.0, 1.0).astype(np.float32, copy=False)
    output = io.BytesIO()
    try:
        soundfile.write(
            file=output,
            data=pcm_ready,
            samplerate=sampling_rate,
            format="WAV",
            subtype="PCM_16",
        )
    except Exception as error:
        raise AudioProcessingError("failed to encode PCM WAV") from error
    result = output.getvalue()
    if result[:4] != b"RIFF" or result[8:12] != b"WAVE":
        raise AudioProcessingError("encoder did not produce a RIFF/WAVE file")
    return result


def decode_pcm_wav(
    wav_bytes: WavInput,
    expected_sampling_rate: int | None = None,
) -> tuple[Waveform, int]:
    """Decode a mono PCM WAV into ``(float32_samples, sampling_rate)``.

    The v2 service accepts PCM WAV input only.  Stereo, compressed, floating
    point, malformed, and non-WAV containers are rejected before samples are
    returned.
    """

    if not isinstance(wav_bytes, (bytes, bytearray, memoryview)):
        raise AudioValidationError("wav_bytes must be bytes-like")
    raw = bytes(wav_bytes)
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioValidationError("wav_bytes must be a RIFF/WAVE file")
    if expected_sampling_rate is not None:
        expected_sampling_rate = _require_sampling_rate(
            expected_sampling_rate, "expected_sampling_rate"
        )

    try:
        info = soundfile.info(io.BytesIO(raw))
        if info.format.upper() != "WAV":
            raise AudioValidationError("WAV container is required")
        if not info.subtype.upper().startswith("PCM"):
            raise AudioValidationError("PCM WAV data is required")
        if info.channels != 1:
            raise AudioValidationError("only mono WAV data is supported")
        _require_bounded_waveform_size(int(info.frames), "WAV data")
        if (
            expected_sampling_rate is not None
            and info.samplerate != expected_sampling_rate
        ):
            raise AudioValidationError(f"unexpected sampling rate: {info.samplerate}")
        wave, sampling_rate = soundfile.read(
            io.BytesIO(raw), dtype="float32", always_2d=True
        )
    except AudioValidationError:
        raise
    except Exception as error:
        raise AudioValidationError("wav_bytes is not a readable PCM WAV") from error

    if wave.ndim != 2 or wave.shape[1] != 1:
        raise AudioValidationError("only mono WAV data is supported")
    result = wave[:, 0]
    if result.size == 0:
        raise AudioValidationError("WAV data must not be empty")
    if not np.isfinite(result).all():
        raise AudioValidationError("decoded WAV contains non-finite samples")
    return result.astype(np.float32, copy=False), int(sampling_rate)


def encode_pcm_wav_base64(wave: np.ndarray, sampling_rate: int) -> str:
    """Encode a waveform as standard Base64-wrapped PCM16 WAV."""

    return base64.standard_b64encode(encode_pcm_wav(wave, sampling_rate)).decode(
        "ascii"
    )


def decode_pcm_wav_base64(
    wav_base64: str,
    expected_sampling_rate: int | None = None,
) -> tuple[Waveform, int]:
    """Decode a Base64 PCM WAV string into ``(float32_samples, rate)``."""

    if not isinstance(wav_base64, str):
        raise AudioValidationError("wav_base64 must be a string")
    try:
        raw = base64.b64decode(wav_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise AudioValidationError("wav_base64 is not valid Base64") from error
    return decode_pcm_wav(raw, expected_sampling_rate=expected_sampling_rate)


def _core_audio_manager() -> Any:
    """Load the public Core adapter lazily so pure WAV helpers stay lightweight."""

    from coeirocore.coeiro_manager import AudioManager

    return AudioManager


class _CompatibilityWaveProcessor:
    """旧Python APIから、モデルを生成せずCoreの波形プリミティブだけを利用する。"""

    @staticmethod
    def get_world(wave: np.ndarray, sampling_rate: int):
        return _core_audio_manager().get_world(wave, sampling_rate)

    @staticmethod
    def trim(wave: np.ndarray) -> np.ndarray:
        return _core_audio_manager().trim(wave)

    @staticmethod
    def volume(wave: np.ndarray, volume_scale: float) -> np.ndarray:
        return _core_audio_manager().volume(wave, volume_scale)

    @staticmethod
    def pitch_intonation(
        wave: np.ndarray,
        sampling_rate: int,
        pitch_scale: float,
        intonation_scale: float,
    ) -> np.ndarray:
        return _core_audio_manager().pitch_intonation(
            wave,
            sampling_rate,
            pitch_scale,
            intonation_scale,
        )

    @staticmethod
    def sil(
        wave: np.ndarray,
        sampling_rate: int,
        pre_phoneme_length: float,
        post_phoneme_length: float,
    ) -> np.ndarray:
        return _core_audio_manager().sil(
            wave,
            sampling_rate,
            pre_phoneme_length,
            post_phoneme_length,
        )

    @staticmethod
    def resample_output(
        wave: np.ndarray,
        sampling_rate: int,
        output_sampling_rate: int,
    ) -> np.ndarray:
        return _core_audio_manager().resampling(
            wave,
            sampling_rate,
            output_sampling_rate,
        )


_COMPATIBILITY_WAVE_PROCESSOR = _CompatibilityWaveProcessor()


def trim_wave(
    wave: np.ndarray,
    sampling_rate: int,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Waveform:
    """Trim silence while retaining the requested context around both edges.

    Coreと同じRMS判定を使う。trim-buffer値は検出範囲の外側へ残す音声長であり、追加で破棄する長さではない。
    """

    wave = _require_waveform(wave)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    start_trim_buffer = _validate_duration(
        start_trim_buffer,
        "start_trim_buffer",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    end_trim_buffer = _validate_duration(
        end_trim_buffer,
        "end_trim_buffer",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    try:
        from coeirocore.waveform import detect_non_silent_range

        detected_range = detect_non_silent_range(wave, top_db=30)
    except Exception as error:
        raise AudioProcessingError("failed to trim waveform") from error
    start = max(
        0,
        int(detected_range[0]) - int(sampling_rate * start_trim_buffer),
    )
    end = min(
        wave.size,
        int(detected_range[1]) + int(sampling_rate * end_trim_buffer),
    )
    trimmed = np.asarray(wave[start:end], dtype=np.float32)
    if trimmed.ndim != 1 or not np.isfinite(trimmed).all():
        raise AudioProcessingError("trim produced invalid samples")
    return trimmed.copy()


def apply_trim_buffer(
    wave: np.ndarray,
    sampling_rate: int,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Waveform:
    """Compatibility alias for buffered silence trimming."""

    return trim_wave(
        wave,
        sampling_rate,
        start_trim_buffer=start_trim_buffer,
        end_trim_buffer=end_trim_buffer,
    )


def replace_pause_segments(
    wave: np.ndarray,
    sampling_rate: int,
    mora_durations: Sequence[MoraDuration | dict],
    pause_length: float,
    pause_start_trim_buffer: float = 0.0,
    pause_end_trim_buffer: float = 0.0,
) -> Waveform:
    """Replace internal ``pau`` ranges with a requested silence length.

    The leading and trailing ``pau`` records describe the utterance edges and
    are handled by :func:`trim_wave`.  Only internal punctuation pauses are
    replaced here.  The two pause buffers retain a small amount of the source
    pause on each side so a model whose duration boundary overlaps speech is
    not cut abruptly.
    """

    current = _require_waveform(wave)
    _require_bounded_waveform_size(current.size)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    pause_length = _validate_duration(
        pause_length,
        "pause_length",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    pause_start_trim_buffer = _validate_duration(
        pause_start_trim_buffer,
        "pause_start_trim_buffer",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    pause_end_trim_buffer = _validate_duration(
        pause_end_trim_buffer,
        "pause_end_trim_buffer",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    if isinstance(mora_durations, (str, bytes)):
        raise AudioValidationError("mora_durations must be a sequence")
    try:
        durations = [
            item
            if isinstance(item, MoraDuration)
            else MoraDuration.model_validate(item)
            for item in mora_durations
        ]
    except (TypeError, ValueError) as error:
        raise AudioValidationError("mora_durations contains an invalid item") from error

    pauses = []
    previous_end = 0
    for item in durations:
        start = int(item.wav_range.start)
        end = int(item.wav_range.end)
        if start < previous_end or end < start or end > current.size:
            raise AudioValidationError(
                "mora_durations contains an invalid or overlapping wavRange"
            )
        previous_end = end
        if item.mora.lower() == "pau" and start > 0 and end < current.size:
            if end <= start:
                raise AudioValidationError(
                    "an internal pause must have a non-empty wavRange"
                )
            pauses.append((start, end))

    if not pauses:
        return current.copy()

    left_requested = int(sampling_rate * pause_start_trim_buffer)
    right_requested = int(sampling_rate * pause_end_trim_buffer)
    inserted_size = int(sampling_rate * pause_length)
    result_size = current.size
    for start, end in pauses:
        segment_size = end - start
        left_size = min(left_requested, segment_size)
        right_size = min(right_requested, segment_size - left_size)
        result_size += inserted_size + left_size + right_size - segment_size
    if result_size <= 0 or result_size > MAX_GENERATED_WAVE_SAMPLES:
        raise AudioValidationError(
            "pause processing would create an excessively large waveform"
        )
    inserted = np.zeros(inserted_size, dtype=np.float32)
    chunks = []
    cursor = 0
    for start, end in pauses:
        chunks.append(current[cursor:start])
        segment_size = end - start
        left_size = min(left_requested, segment_size)
        right_size = min(right_requested, segment_size - left_size)
        if left_size:
            chunks.append(current[start : start + left_size])
        chunks.append(inserted)
        if right_size:
            chunks.append(current[end - right_size : end])
        cursor = end
    chunks.append(current[cursor:])
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if result.size == 0 or not np.isfinite(result).all():
        raise AudioProcessingError("pause processing produced invalid samples")
    return result.copy()


def pitch_shift_resampling(
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float,
) -> Waveform:
    """Apply the v2 RESAMPLING-style global pitch shift at fixed duration."""

    current = _require_waveform(wave)
    _require_bounded_waveform_size(current.size)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    pitch_scale = _validate_scale(pitch_scale, "pitch_scale")
    if pitch_scale == 0.0:
        return current.copy()
    try:
        import librosa

        shifted = librosa.effects.pitch_shift(
            current,
            sr=sampling_rate,
            n_steps=12.0 * pitch_scale,
            bins_per_octave=12,
            res_type="kaiser_fast",
        )
    except Exception as error:
        raise AudioProcessingError("failed to apply resampling pitch shift") from error
    result = np.asarray(shifted, dtype=np.float32).reshape(-1)
    if result.size < current.size:
        result = np.pad(result, (0, current.size - result.size))
    elif result.size > current.size:
        result = result[: current.size]
    if not np.isfinite(result).all():
        raise AudioProcessingError("resampling pitch shift produced non-finite samples")
    return result.copy()


def process_wave(
    wave: np.ndarray,
    sampling_rate: int,
    *,
    volume_scale: float = 1.0,
    pitch_scale: float = 0.0,
    intonation_scale: float = 1.0,
    pre_phoneme_length: float = 0.0,
    post_phoneme_length: float = 0.0,
    output_sampling_rate: int | None = None,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> tuple[Waveform, int]:
    """旧Python APIの引数を、現行の一元化された波形後処理へ変換する。"""

    from .wave_processing import WaveProcessingOptions
    from .wave_processing import process_wave as process_current_wave

    resolved_output_rate = (
        sampling_rate if output_sampling_rate is None else output_sampling_rate
    )
    return process_current_wave(
        _COMPATIBILITY_WAVE_PROCESSOR,
        wave,
        sampling_rate,
        WaveProcessingOptions(
            volume_scale=volume_scale,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            pre_phoneme_length=pre_phoneme_length,
            post_phoneme_length=post_phoneme_length,
            output_sampling_rate=resolved_output_rate,
            start_trim_buffer=start_trim_buffer,
            end_trim_buffer=end_trim_buffer,
            processing_algorithm="world",
        ),
    )


def process_wav(
    wav_bytes: WavInput,
    sampling_rate: int | None = None,
    **processing: Any,
) -> bytes:
    """旧Python APIとしてPCM WAVをデコードし、現行処理後に再エンコードする。"""

    wave, input_sampling_rate = decode_pcm_wav(
        wav_bytes,
        expected_sampling_rate=sampling_rate,
    )
    output_wave, output_sampling_rate = process_wave(
        wave,
        input_sampling_rate,
        **processing,
    )
    return encode_pcm_wav(output_wave, output_sampling_rate)


def estimate_world_f0(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Estimate a finite WORLD F0 vector with the public Core implementation."""

    wave = _require_waveform(wave)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    try:
        f0, _, _ = _core_audio_manager().get_world(
            wave.astype(np.float64), sampling_rate
        )
    except Exception as error:
        raise AudioProcessingError("failed to estimate WORLD F0") from error
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    if not np.isfinite(f0).all():
        raise AudioProcessingError("WORLD F0 contains non-finite values")
    return f0.copy()


def prepare_world_f0(
    wave: np.ndarray,
    sampling_rate: int,
    mora_durations: Sequence[MoraDuration | dict],
) -> WorldF0:
    """Prepare the JSON-ready WORLD F0 response used by ``/v1/estimate_f0``."""

    if isinstance(mora_durations, (str, bytes)):
        raise AudioValidationError("mora_durations must be a sequence")
    try:
        durations = [
            item
            if isinstance(item, MoraDuration)
            else MoraDuration.model_validate(item)
            for item in mora_durations
        ]
    except (TypeError, ValueError) as error:
        raise AudioValidationError("mora_durations contains an invalid item") from error
    f0 = estimate_world_f0(wave, sampling_rate)
    return WorldF0(f0=[float(value) for value in f0], moraDurations=durations)


__all__ = [
    "MAX_GENERATED_WAVE_SAMPLES",
    "MAX_PAUSE_LENGTH_SECONDS",
    "MAX_SAMPLING_RATE",
    "AudioProcessingError",
    "AudioValidationError",
    "apply_trim_buffer",
    "decode_pcm_wav",
    "decode_pcm_wav_base64",
    "encode_pcm_wav",
    "encode_pcm_wav_base64",
    "estimate_world_f0",
    "pitch_shift_resampling",
    "prepare_world_f0",
    "process_wav",
    "process_wave",
    "replace_pause_segments",
    "trim_wave",
]
