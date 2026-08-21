"""A small, deterministic time-domain PSOLA processor.

The processor is intentionally independent from the packaged COEIROINK
application.  It only needs NumPy and, when an F0 track is not supplied, the
public WORLD estimator exposed by :mod:`voicevox_engine.coeiroink_v2.audio`.
"""

import math
import numbers
from collections.abc import Sequence

import numpy as np

PROCESSING_ALGORITHM = "td-psola"
DEFAULT_FRAME_PERIOD = 0.005

# 不正または過大なリクエストによるmark/grainの無制限確保を防ぎ、音高変化は隣接grainが重なる1オクターブ以内へ制限する。
MIN_F0_HZ = 20.0
MAX_F0_HZ = 2000.0
MIN_PITCH_RATIO = 0.5
MAX_PITCH_RATIO = 2.0
MAX_MARKS = 1000000
FLOAT32_MAX = float(np.finfo(np.float32).max)


class TDPSOLAValidationError(ValueError):
    """Raised when an input to the TD-PSOLA processor is invalid."""


class TDPSOLAProcessingError(RuntimeError):
    """Raised when the processor cannot produce a finite waveform."""


def _require_waveform(wave: object) -> np.ndarray:
    if not isinstance(wave, np.ndarray):
        raise TDPSOLAValidationError("wave must be a NumPy ndarray")
    if wave.ndim != 1:
        raise TDPSOLAValidationError("wave must be a mono 1-D array")
    if not np.issubdtype(wave.dtype, np.floating):
        raise TDPSOLAValidationError("wave must have a floating-point dtype")
    if wave.size == 0:
        raise TDPSOLAValidationError("wave must not be empty")
    if not np.isfinite(wave).all():
        raise TDPSOLAValidationError("wave must contain only finite samples")

    if np.any(np.abs(wave) > FLOAT32_MAX):
        raise TDPSOLAValidationError(
            "wave cannot be represented as finite float32 samples"
        )
    converted = wave.astype(np.float32, copy=False)
    if not np.isfinite(converted).all():
        raise TDPSOLAValidationError(
            "wave cannot be represented as finite float32 samples"
        )
    return converted


def _require_rate(sampling_rate: object) -> int:
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, numbers.Integral)
        or int(sampling_rate) <= 0
    ):
        raise TDPSOLAValidationError("sampling_rate must be a positive integer")
    return int(sampling_rate)


def _require_scale(value: object, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TDPSOLAValidationError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise TDPSOLAValidationError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise TDPSOLAValidationError(
            f"{name} must be greater than or equal to {minimum}"
        )
    return value


def _require_frame_period(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TDPSOLAValidationError("frame_period must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise TDPSOLAValidationError("frame_period must be a finite positive number")
    return value


def _as_f0_array(f0: object) -> np.ndarray:
    try:
        values = np.asarray(f0)
    except Exception as error:
        raise TDPSOLAValidationError(
            "f0 must be a one-dimensional numeric array"
        ) from error
    if values.ndim != 1 or values.size == 0:
        raise TDPSOLAValidationError("f0 must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TDPSOLAValidationError("f0 must be a one-dimensional numeric array")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise TDPSOLAValidationError("f0 must contain only finite values")
    if np.any(values < 0.0):
        raise TDPSOLAValidationError("f0 must not contain negative values")
    return values


def _sample_f0_track(
    f0: np.ndarray,
    sample_count: int,
    sampling_rate: int,
    frame_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a sample- or frame-aligned F0 track to the waveform length."""

    if f0.size == sample_count:
        samples = f0.copy()
        voiced = samples > 0.0
    else:
        frame_step = frame_period * sampling_rate
        frame_positions = np.arange(f0.size, dtype=np.float64) * frame_step
        sample_positions = np.arange(sample_count, dtype=np.float64)
        nearest = np.rint(sample_positions / frame_step).astype(np.int64)
        nearest = np.clip(nearest, 0, f0.size - 1)
        voiced = f0[nearest] > 0.0

        positive = np.flatnonzero(f0 > 0.0)
        samples = np.zeros(sample_count, dtype=np.float64)
        if positive.size:
            samples[voiced] = np.interp(
                sample_positions[voiced],
                frame_positions[positive],
                f0[positive],
            )

    # WORLDが強い雑音から実用的な音声範囲外の値を返しても、巨大grainやゼロ幅stepを作らないよう周期を制限する。
    positive = samples > 0.0
    samples[positive] = np.clip(samples[positive], MIN_F0_HZ, MAX_F0_HZ)
    return samples, voiced


def _f0_from_public_world(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
    # F0軌跡を指定する呼出元がCoreアダプターやWORLDを初期化せず使えるよう、必要時だけ読み込む。
    from .audio import estimate_world_f0

    try:
        return _as_f0_array(estimate_world_f0(wave, sampling_rate))
    except TDPSOLAValidationError:
        raise
    except Exception as error:
        raise TDPSOLAProcessingError("failed to estimate WORLD F0") from error


def _voiced_runs(mask: np.ndarray, minimum_samples: int) -> list[tuple[int, int]]:
    padded = np.concatenate(
        (
            np.zeros(1, dtype=bool),
            mask.astype(bool, copy=False),
            np.zeros(1, dtype=bool),
        )
    )
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if int(end) - int(start) >= minimum_samples
    ]


def _period_at(f0: np.ndarray, sample: int, sampling_rate: int) -> float:
    index = min(max(int(sample), 0), f0.size - 1)
    frequency = float(f0[index])
    if not math.isfinite(frequency) or frequency <= 0.0:
        frequency = MIN_F0_HZ
    frequency = min(max(frequency, MIN_F0_HZ), MAX_F0_HZ)
    return max(2.0, float(sampling_rate) / frequency)


def _source_marks(
    wave: np.ndarray,
    f0: np.ndarray,
    start: int,
    end: int,
    sampling_rate: int,
) -> np.ndarray:
    """Find deterministic, local high-energy marks in one voiced run."""

    marks: list[int] = []
    expected = float(start)
    previous = start - 1
    while expected < end:
        period = _period_at(f0, round(expected), sampling_rate)
        center = round(expected)
        radius = max(2, round(0.45 * period))
        left = max(start, center - radius)
        right = min(end, center + radius + 1)
        if right - left < 3:
            break

        candidate = left + int(np.argmax(np.abs(wave[left:right])))
        if candidate <= previous:
            candidate = max(previous + 1, center)
        if candidate >= end:
            break
        marks.append(candidate)
        previous = candidate
        expected = float(candidate) + period
        if len(marks) > MAX_MARKS:
            raise TDPSOLAProcessingError("too many PSOLA analysis marks")

    if len(marks) < 2:
        return np.empty(0, dtype=np.int64)
    return np.asarray(marks, dtype=np.int64)


def _nearest_mark(marks: np.ndarray, sample: int) -> int:
    position = int(np.searchsorted(marks, sample, side="left"))
    if position <= 0:
        return 0
    if position >= marks.size:
        return int(marks.size - 1)
    before = position - 1
    after = position
    if abs(int(marks[before]) - sample) <= abs(int(marks[after]) - sample):
        return before
    return after


def _target_marks(
    source_marks: np.ndarray,
    target_f0: np.ndarray,
    start: int,
    end: int,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """目標F0の周期で合成markを進め、各markに最も近い入力grainの索引を対応付ける。"""

    marks: list[int] = []
    source_indices: list[int] = []
    expected = float(source_marks[0])
    previous = start - 1
    while expected < end:
        target = round(expected)
        if target <= previous:
            target = previous + 1
        if target >= end:
            break
        source_index = _nearest_mark(source_marks, target)
        marks.append(target)
        source_indices.append(source_index)
        previous = target
        period = _period_at(target_f0, target, sampling_rate)
        expected = float(target) + period
        if len(marks) > MAX_MARKS:
            raise TDPSOLAProcessingError("too many PSOLA synthesis marks")
    return np.asarray(marks, dtype=np.int64), np.asarray(source_indices, dtype=np.int64)


def _edge_gate(
    indices: np.ndarray, start: int, end: int, fade_samples: int
) -> np.ndarray:
    """Return a raised-cosine gate that is zero outside a voiced run."""

    values = np.asarray(indices, dtype=np.float64)
    gate = np.ones(values.shape, dtype=np.float64)
    gate[(values < start) | (values >= end)] = 0.0
    if fade_samples <= 0 or end <= start:
        return gate

    left = (values - float(start)) / float(fade_samples)
    left_gate = 0.5 - 0.5 * np.cos(np.pi * np.clip(left, 0.0, 1.0))
    right = (float(end - 1) - values) / float(fade_samples)
    right_gate = 0.5 - 0.5 * np.cos(np.pi * np.clip(right, 0.0, 1.0))
    inside = (values >= start) & (values < end)
    gate[inside] = np.minimum(left_gate[inside], right_gate[inside])
    return gate


def _bounded_target_f0(
    source_f0: np.ndarray,
    voiced: np.ndarray,
    pitch_scale: float,
    intonation_scale: float,
) -> np.ndarray:
    target = source_f0.copy()
    voiced_values = source_f0[voiced]
    if voiced_values.size == 0:
        return target

    if intonation_scale != 1.0:
        mean = float(np.mean(voiced_values))
        deviation = voiced_values - mean
        target[voiced] = mean + deviation * intonation_scale

    # exp2の指数を制限して有限なAPI入力を有限値に保ち、実際のPSOLA安全範囲は直後の相対倍率で制限する。
    ratio = 2.0 ** min(max(pitch_scale, -32.0), 32.0)
    target[voiced] *= ratio
    lower = source_f0[voiced] * MIN_PITCH_RATIO
    upper = source_f0[voiced] * MAX_PITCH_RATIO
    target[voiced] = np.clip(target[voiced], lower, upper)
    target[voiced] = np.clip(target[voiced], MIN_F0_HZ, MAX_F0_HZ)
    return target


def process_td_psola(
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float = 0.0,
    intonation_scale: float = 1.0,
    *,
    f0: Sequence[float] | None = None,
    target_f0: Sequence[float] | None = None,
    frame_period: float = DEFAULT_FRAME_PERIOD,
) -> np.ndarray:
    """Pitch/intonation-process a mono waveform while retaining its length.

    ``pitch_scale`` follows the COEIROINK/VOICEVOX convention: one unit is an
    octave and the frequency multiplier is ``2 ** pitch_scale``.
    ``intonation_scale`` expands or contracts the voiced F0 deviations around
    their voiced mean.  ``f0`` is the analysed source track and
    ``target_f0`` is an optional caller-adjusted destination track; either may
    be sample-aligned or WORLD-style.  When ``f0`` is omitted, the existing
    public WORLD estimator supplies the source track.

    The implementation is a bounded PSOLA adapter: marks are generated only
    inside voiced runs, each grain is at most two source periods wide, and the
    output has exactly the input sample count.  Unvoiced samples are retained
    and all overlap-add divisions are guarded.

    Algorithm basis: E. Moulines and F. Charpentier, "Pitch-synchronous
    waveform processing techniques for text-to-speech synthesis using
    diphones", Speech Communication 9(5-6), 453-467 (1990),
    https://doi.org/10.1016/0167-6393(90)90021-Z .
    """

    validated_wave = _require_waveform(wave)
    sampling_rate = _require_rate(sampling_rate)
    pitch_scale = _require_scale(pitch_scale, "pitch_scale")
    intonation_scale = _require_scale(intonation_scale, "intonation_scale", minimum=0.0)
    frame_period = _require_frame_period(frame_period)

    # 無変換時は推定器を呼ばず、サンプル値を変えない厳密な恒等処理にする。
    if target_f0 is None and pitch_scale == 0.0 and intonation_scale == 1.0:
        return validated_wave.copy()

    if f0 is None:
        # WORLDの解析窓より短い波形には有効な有声音変換を行えないため、そのまま保持する。
        if validated_wave.size < max(64, round(sampling_rate * 0.02)):
            return validated_wave.copy()
        track = _f0_from_public_world(validated_wave, sampling_rate)
    else:
        track = _as_f0_array(f0)

    source_f0, voiced = _sample_f0_track(
        track, validated_wave.size, sampling_rate, frame_period
    )
    if not np.any(voiced):
        return validated_wave.copy()

    if target_f0 is None:
        target_samples = _bounded_target_f0(
            source_f0, voiced, pitch_scale, intonation_scale
        )
    else:
        requested_track = _as_f0_array(target_f0)
        requested_samples, requested_voiced = _sample_f0_track(
            requested_track,
            validated_wave.size,
            sampling_rate,
            frame_period,
        )
        voiced &= requested_voiced
        if not np.any(voiced):
            return validated_wave.copy()
        target_samples = source_f0.copy()
        lower = source_f0[voiced] * MIN_PITCH_RATIO
        upper = source_f0[voiced] * MAX_PITCH_RATIO
        target_samples[voiced] = np.clip(requested_samples[voiced], lower, upper)
        target_samples[voiced] = np.clip(target_samples[voiced], MIN_F0_HZ, MAX_F0_HZ)
    source = validated_wave.astype(np.float64, copy=False)
    output_size = source.size
    accumulated = np.zeros(output_size, dtype=np.float64)
    weights = np.zeros(output_size, dtype=np.float64)
    minimum_run = max(8, round(sampling_rate * 0.005))
    runs = _voiced_runs(voiced, minimum_run)

    window_cache = {}
    for start, end in runs:
        source_marks = _source_marks(source, source_f0, start, end, sampling_rate)
        if source_marks.size < 2:
            continue
        synthesis_marks, source_indices = _target_marks(
            source_marks, target_samples, start, end, sampling_rate
        )
        if synthesis_marks.size == 0:
            continue

        max_period = max(
            _period_at(source_f0, int(source_marks[0]), sampling_rate),
            _period_at(source_f0, int(source_marks[-1]), sampling_rate),
        )
        fade_samples = max(1, math.ceil(max_period))
        for synthesis_mark, source_mark_index in zip(
            synthesis_marks, source_indices, strict=True
        ):
            source_mark = int(source_marks[int(source_mark_index)])
            half_window = max(
                2,
                math.ceil(_period_at(source_f0, source_mark, sampling_rate)),
            )
            window = window_cache.get(half_window)
            if window is None:
                window = np.hanning(2 * half_window + 1).astype(np.float64, copy=False)
                window_cache[half_window] = window

            offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
            source_positions = source_mark + offsets
            output_positions = int(synthesis_mark) + offsets
            valid = (
                (source_positions >= start)
                & (source_positions < end)
                & (output_positions >= start)
                & (output_positions < end)
            )
            if not np.any(valid):
                continue

            source_positions = source_positions[valid]
            output_positions = output_positions[valid]
            local_window = window[valid]
            source_gate = _edge_gate(source_positions, start, end, fade_samples)
            accumulated[output_positions] += (
                source[source_positions] * local_window * source_gate
            )
            # 正規化でgateを相殺せず滑らかな遷移を残すため、窓の重みはsource gateから独立させる。
            weights[output_positions] += local_window

    covered = weights > np.finfo(np.float64).eps
    if not np.any(covered):
        return validated_wave.copy()

    processed = source.copy()
    processed[covered] = accumulated[covered] / weights[covered]
    result = source.copy()
    for start, end in runs:
        region = np.arange(start, end, dtype=np.int64)
        fade = min(
            max(1, math.ceil(sampling_rate * 0.005)),
            max(1, (end - start) // 2),
        )
        blend = _edge_gate(region, start, end, fade)
        active = covered[region]
        if np.any(active):
            positions = region[active]
            result[positions] = (
                source[positions] * (1.0 - blend[active])
                + processed[positions] * blend[active]
            )

    if (
        result.shape != validated_wave.shape
        or not np.isfinite(result).all()
        or np.any(np.abs(result) > FLOAT32_MAX)
    ):
        raise TDPSOLAProcessingError(
            "TD-PSOLA produced a non-finite or malformed waveform"
        )
    result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise TDPSOLAProcessingError(
            "TD-PSOLA produced a non-finite or malformed waveform"
        )
    return result.copy()


def apply_td_psola(
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float = 0.0,
    intonation_scale: float = 1.0,
    *,
    f0: Sequence[float] | None = None,
    target_f0: Sequence[float] | None = None,
    frame_period: float = DEFAULT_FRAME_PERIOD,
) -> np.ndarray:
    """Compatibility name for the v2 waveform-processing adapter."""

    return process_td_psola(
        wave,
        sampling_rate,
        pitch_scale=pitch_scale,
        intonation_scale=intonation_scale,
        f0=f0,
        target_f0=target_f0,
        frame_period=frame_period,
    )


__all__ = [
    "DEFAULT_FRAME_PERIOD",
    "MAX_F0_HZ",
    "MIN_F0_HZ",
    "PROCESSING_ALGORITHM",
    "TDPSOLAProcessingError",
    "TDPSOLAValidationError",
    "apply_td_psola",
    "process_td_psola",
]
