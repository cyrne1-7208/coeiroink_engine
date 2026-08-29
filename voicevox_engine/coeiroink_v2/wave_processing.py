"""COEIROINK v2の波形後処理。

HTTPルーターはリクエスト値の解決だけを担当し、このモジュールはCoreの公開波形プリミティブを一定の順序で適用する。
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import audio as audio_helpers
from .td_psola import process_td_psola

_PROCESSING_ALGORITHM_ALIASES = {"coeiroink": "td-psola"}
_PROCESSING_ALGORITHMS = frozenset(("td-psola", "world", "resampling"))


@dataclass(frozen=True, slots=True)
class WaveProcessingOptions:
    """HTTPモデルから解決済みの波形後処理オプション。"""

    volume_scale: float
    pitch_scale: float
    intonation_scale: float
    pre_phoneme_length: float
    post_phoneme_length: float
    output_sampling_rate: int
    start_trim_buffer: float
    end_trim_buffer: float
    processing_algorithm: str | None = None
    adjusted_f0: Sequence[float] | None = None
    sampled_interval_value: int | None = None
    pause_length: float | None = None
    pause_start_trim_buffer: float | None = None
    pause_end_trim_buffer: float | None = None
    mora_durations: Sequence[Any] | None = None


def as_waveform(value: Any) -> np.ndarray:
    wave = np.asarray(value, dtype=np.float32)
    if wave.ndim != 1:
        wave = wave.reshape(-1)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise audio_helpers.AudioValidationError(
            "audio manager returned an invalid waveform"
        )
    return wave


def normalize_processing_algorithm(value: str | None) -> str:
    algorithm = (value or "td-psola").lower()
    algorithm = _PROCESSING_ALGORITHM_ALIASES.get(algorithm, algorithm)
    if algorithm not in _PROCESSING_ALGORITHMS:
        supported = ", ".join(sorted(_PROCESSING_ALGORITHMS))
        raise audio_helpers.AudioValidationError(
            f"processing_algorithm must be one of: {supported}"
        )
    return algorithm


def manager_audio_method(audio_manager: Any, name: str) -> Callable[..., Any] | None:
    method = getattr(audio_manager, name, None)
    return method if callable(method) else None


def _validated_f0(value: Sequence[float]) -> np.ndarray:
    try:
        f0 = np.asarray(value, dtype=np.float64)
    except Exception as error:
        raise audio_helpers.AudioValidationError(
            "adjusted_f0 must be numeric"
        ) from error
    if f0.ndim != 1 or f0.size == 0 or not np.isfinite(f0).all():
        raise audio_helpers.AudioValidationError(
            "adjusted_f0 must be a non-empty finite one-dimensional array"
        )
    if np.any(f0 < 0.0):
        raise audio_helpers.AudioValidationError("adjusted_f0 must not be negative")
    return f0


def _manager_world_f0(
    audio_manager: Any, wave: np.ndarray, sampling_rate: int
) -> np.ndarray | None:
    get_world = manager_audio_method(audio_manager, "get_world")
    if get_world is None:
        return None
    result = get_world(wave.astype(np.float64), sampling_rate)
    if not isinstance(result, (tuple, list)) or not result:
        raise audio_helpers.AudioProcessingError(
            "audio manager returned an invalid WORLD result"
        )
    return _validated_f0(result[0])


def _fit_f0_track(track: np.ndarray, size: int) -> np.ndarray:
    if track.size == size:
        return track.astype(np.float64, copy=True)
    positions = np.linspace(0.0, 1.0, track.size)
    target_positions = np.linspace(0.0, 1.0, size)
    return np.interp(target_positions, positions, track).astype(np.float64)


def _world_process(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float,
    intonation_scale: float,
    adjusted_f0: Sequence[float] | None,
) -> np.ndarray:
    """CoreのWORLD処理を使い、必要なら呼出元が指定したF0軌跡を適用する。"""

    if adjusted_f0 is None:
        pitch = manager_audio_method(audio_manager, "pitch_intonation")
        if pitch is not None:
            return as_waveform(
                pitch(wave, sampling_rate, pitch_scale, intonation_scale)
            )

    get_world = manager_audio_method(audio_manager, "get_world")
    if get_world is None:
        raise audio_helpers.AudioProcessingError(
            "WORLD processing with adjustedF0 requires get_world"
        )
    result = get_world(wave.astype(np.float64), sampling_rate)
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise audio_helpers.AudioProcessingError(
            "audio manager returned an invalid WORLD result"
        )
    base_f0, spectral_envelope, aperiodicity = result[:3]
    f0 = (
        _validated_f0(adjusted_f0)
        if adjusted_f0 is not None
        else _validated_f0(base_f0)
    )
    f0 = _fit_f0_track(f0, len(np.asarray(base_f0).reshape(-1)))
    if pitch_scale != 0.0:
        f0 *= 2.0 ** float(pitch_scale)
    if intonation_scale != 1.0:
        # WORLDは無声音をF0=0で表すため、有声音だけへ抑揚を適用して休止や無声子音に音高を作らない。
        voiced = f0 > 0.0
        if np.any(voiced):
            voiced_f0 = f0[voiced]
            mean = float(np.mean(voiced_f0))
            f0[voiced] = (voiced_f0 - mean) * float(intonation_scale) + mean

    try:
        from coeirocore.pyworld_compat import load_pyworld

        pw = load_pyworld()
        processed = pw.synthesize(
            f0.astype(np.float64),
            spectral_envelope,
            aperiodicity,
            sampling_rate,
        )
    except Exception as error:
        raise audio_helpers.AudioProcessingError(
            "failed to synthesize WORLD-processed audio"
        ) from error
    return as_waveform(processed)


def _processing_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise audio_helpers.AudioValidationError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise audio_helpers.AudioValidationError(
            f"{name} must be greater than or equal to {minimum}"
        )
    if maximum is not None and result > maximum:
        raise audio_helpers.AudioValidationError(
            f"{name} must be no greater than {maximum}"
        )
    return result


def _processing_duration(value: object, name: str) -> float:
    return _processing_number(
        value,
        name,
        minimum=0.0,
        maximum=audio_helpers.MAX_PAUSE_LENGTH_SECONDS,
    )


def _processing_sampling_rate(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        or int(value) > audio_helpers.MAX_SAMPLING_RATE
    ):
        raise audio_helpers.AudioValidationError(
            f"{name} must be a positive integer no greater than {audio_helpers.MAX_SAMPLING_RATE}"
        )
    return int(value)


def _require_processing_size(size: int, name: str = "processed wave") -> None:
    if size <= 0 or size > audio_helpers.MAX_GENERATED_WAVE_SAMPLES:
        raise audio_helpers.AudioValidationError(
            f"{name} would exceed the maximum of {audio_helpers.MAX_GENERATED_WAVE_SAMPLES} samples"
        )


@dataclass(frozen=True, slots=True)
class _ResolvedWaveProcessingOptions:
    volume_scale: float
    pitch_scale: float
    intonation_scale: float
    pre_phoneme_length: float
    post_phoneme_length: float
    output_sampling_rate: int
    start_trim_buffer: float
    end_trim_buffer: float
    processing_algorithm: str
    adjusted_f0: np.ndarray | None
    pause_length: float | None
    pause_start_trim_buffer: float
    pause_end_trim_buffer: float
    mora_durations: Sequence[Any] | None


def _optional_adjusted_f0(value: Sequence[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if len(value) == 0:
            # 公式リクエスト例の空配列は、呼出元指定のF0軌跡がないことを表す。
            return None
    except TypeError:
        pass
    return _validated_f0(value)


def _resolve_processing_options(
    options: WaveProcessingOptions,
) -> _ResolvedWaveProcessingOptions:
    pause_length = (
        _processing_duration(options.pause_length, "pause_length")
        if options.pause_length is not None
        else None
    )
    return _ResolvedWaveProcessingOptions(
        volume_scale=_processing_number(
            options.volume_scale, "volume_scale", minimum=0.0
        ),
        pitch_scale=_processing_number(
            options.pitch_scale, "pitch_scale", minimum=-32.0, maximum=32.0
        ),
        intonation_scale=_processing_number(
            options.intonation_scale, "intonation_scale", minimum=0.0
        ),
        pre_phoneme_length=_processing_duration(
            options.pre_phoneme_length, "pre_phoneme_length"
        ),
        post_phoneme_length=_processing_duration(
            options.post_phoneme_length, "post_phoneme_length"
        ),
        output_sampling_rate=_processing_sampling_rate(
            options.output_sampling_rate, "output_sampling_rate"
        ),
        start_trim_buffer=_processing_duration(
            options.start_trim_buffer, "start_trim_buffer"
        ),
        end_trim_buffer=_processing_duration(
            options.end_trim_buffer, "end_trim_buffer"
        ),
        processing_algorithm=normalize_processing_algorithm(
            options.processing_algorithm
        ),
        adjusted_f0=_optional_adjusted_f0(options.adjusted_f0),
        pause_length=pause_length,
        pause_start_trim_buffer=_processing_duration(
            options.pause_start_trim_buffer
            if options.pause_start_trim_buffer is not None
            else 0.0,
            "pause_start_trim_buffer",
        ),
        pause_end_trim_buffer=_processing_duration(
            options.pause_end_trim_buffer
            if options.pause_end_trim_buffer is not None
            else 0.0,
            "pause_end_trim_buffer",
        ),
        mora_durations=options.mora_durations,
    )


def _apply_pitch_processing(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    options: _ResolvedWaveProcessingOptions,
) -> np.ndarray:
    if (
        options.pitch_scale == 0.0
        and options.intonation_scale == 1.0
        and options.adjusted_f0 is None
    ):
        return wave

    if options.processing_algorithm == "world":
        return _world_process(
            audio_manager,
            wave,
            sampling_rate,
            options.pitch_scale,
            options.intonation_scale,
            options.adjusted_f0,
        )

    source_f0 = None
    if options.adjusted_f0 is not None or options.intonation_scale != 1.0:
        source_f0 = _manager_world_f0(audio_manager, wave, sampling_rate)

    if options.processing_algorithm == "resampling":
        if options.adjusted_f0 is not None or options.intonation_scale != 1.0:
            wave = as_waveform(
                process_td_psola(
                    wave,
                    sampling_rate,
                    pitch_scale=0.0,
                    intonation_scale=(
                        1.0
                        if options.adjusted_f0 is not None
                        else options.intonation_scale
                    ),
                    f0=source_f0,
                    target_f0=options.adjusted_f0,
                )
            )
        return audio_helpers.pitch_shift_resampling(
            wave,
            sampling_rate,
            options.pitch_scale,
        )

    if source_f0 is None:
        source_f0 = _manager_world_f0(audio_manager, wave, sampling_rate)
    return as_waveform(
        process_td_psola(
            wave,
            sampling_rate,
            pitch_scale=(
                0.0 if options.adjusted_f0 is not None else options.pitch_scale
            ),
            intonation_scale=(
                1.0 if options.adjusted_f0 is not None else options.intonation_scale
            ),
            f0=source_f0,
            target_f0=options.adjusted_f0,
        )
    )


def _apply_pause_and_trim(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    options: _ResolvedWaveProcessingOptions,
) -> np.ndarray:
    if options.pause_length is not None and options.mora_durations:
        wave = audio_helpers.replace_pause_segments(
            wave,
            sampling_rate,
            options.mora_durations,
            pause_length=options.pause_length,
            pause_start_trim_buffer=options.pause_start_trim_buffer,
            pause_end_trim_buffer=options.pause_end_trim_buffer,
        )
    if options.start_trim_buffer or options.end_trim_buffer:
        return audio_helpers.trim_wave(
            wave,
            sampling_rate,
            start_trim_buffer=options.start_trim_buffer,
            end_trim_buffer=options.end_trim_buffer,
        )
    trim = manager_audio_method(audio_manager, "trim")
    return as_waveform(trim(wave)) if trim is not None else wave


def _apply_volume_and_silence(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    options: _ResolvedWaveProcessingOptions,
) -> np.ndarray:
    volume = manager_audio_method(audio_manager, "volume")
    if options.volume_scale != 1.0:
        wave = as_waveform(
            volume(wave, options.volume_scale)
            if volume is not None
            else wave * options.volume_scale
        )

    if options.pre_phoneme_length == 0.0 and options.post_phoneme_length == 0.0:
        return wave
    projected_size = (
        wave.size
        + int(sampling_rate * options.pre_phoneme_length)
        + int(sampling_rate * options.post_phoneme_length)
    )
    _require_processing_size(projected_size)
    silence = manager_audio_method(audio_manager, "sil")
    if silence is not None:
        return as_waveform(
            silence(
                wave,
                sampling_rate,
                options.pre_phoneme_length,
                options.post_phoneme_length,
            )
        )
    pre = np.zeros(int(sampling_rate * options.pre_phoneme_length), dtype=np.float32)
    post = np.zeros(int(sampling_rate * options.post_phoneme_length), dtype=np.float32)
    return np.concatenate((pre, wave, post)).astype(np.float32)


def _resample_output(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    output_sampling_rate: int,
) -> np.ndarray:
    if output_sampling_rate == sampling_rate:
        return wave
    projected_size = math.ceil(wave.size * output_sampling_rate / sampling_rate)
    _require_processing_size(projected_size, "resampled wave")
    resampling = manager_audio_method(audio_manager, "resample_output")
    if resampling is None:
        resampling = manager_audio_method(audio_manager, "resampling")
    if resampling is not None:
        return as_waveform(resampling(wave, sampling_rate, output_sampling_rate))

    try:
        import resampy

        return as_waveform(
            resampy.resample(
                wave,
                sampling_rate,
                output_sampling_rate,
                filter="kaiser_fast",
            )
        )
    except Exception as error:
        raise audio_helpers.AudioProcessingError(
            "failed to resample waveform"
        ) from error


def process_wave(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    options: WaveProcessingOptions,
) -> tuple[np.ndarray, int]:
    """Coreの公開プリミティブを使い、v2の波形後処理を定められた順序で適用する。"""

    current = as_waveform(wave)
    _require_processing_size(current.size, "input wave")
    sampling_rate = _processing_sampling_rate(sampling_rate, "sampling_rate")
    resolved = _resolve_processing_options(options)

    # F0軌跡は未トリムのモデル波形を基準にするため、休止長変更や端部トリムより先に音高処理を行う。
    current = _apply_pitch_processing(audio_manager, current, sampling_rate, resolved)
    current = _apply_pause_and_trim(audio_manager, current, sampling_rate, resolved)
    current = _apply_volume_and_silence(audio_manager, current, sampling_rate, resolved)
    current = _resample_output(
        audio_manager,
        current,
        sampling_rate,
        resolved.output_sampling_rate,
    )
    current = as_waveform(current)
    _require_processing_size(current.size)

    # 旧sampling intervalは通信互換性のため受理するが、公開処理器は安全な解析間隔を内部で決める。
    return current, resolved.output_sampling_rate


__all__ = [
    "WaveProcessingOptions",
    "as_waveform",
    "manager_audio_method",
    "normalize_processing_algorithm",
    "process_wave",
]
