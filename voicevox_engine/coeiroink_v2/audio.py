"""COEIROINK v2 HTTP API向けの小さく検証可能な音声処理です。
このモジュールにはモデル処理とHTTP処理を含めません。
波形はHTTP境界でモノラルPCM16 WAVへ変換するまで1次元の浮動小数点NumPy配列として扱います。
数値的な後処理は公開CoreのAudioManagerへ委譲し、EngineとCoreの音声処理規則を揃えます。
"""

import base64
import binascii
import io
import math
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile

from .models import MoraDuration, WorldF0


class AudioValidationError(ValueError):
    """v2 APIで音声値を使用できない場合に発生します。"""


class AudioProcessingError(RuntimeError):
    """公開音声処理が有限なサンプルを生成できない場合に発生します。"""


Waveform = np.ndarray
WavInput = Union[bytes, bytearray, memoryview]
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
            "{} must be a positive integer no greater than {}".format(
                name, MAX_SAMPLING_RATE
            )
        )
    return sampling_rate


def _require_waveform(wave: object, name: str = "wave") -> Waveform:
    if not isinstance(wave, np.ndarray):
        raise AudioValidationError("{} must be a NumPy ndarray".format(name))
    if wave.ndim != 1:
        raise AudioValidationError("{} must be a mono 1-D array".format(name))
    if not np.issubdtype(wave.dtype, np.floating):
        raise AudioValidationError("{} must have a floating-point dtype".format(name))
    if wave.size == 0:
        raise AudioValidationError("{} must not be empty".format(name))
    if not np.isfinite(wave).all():
        raise AudioValidationError("{} must contain only finite samples".format(name))
    converted = wave.astype(np.float32, copy=False)
    if not np.isfinite(converted).all():
        raise AudioValidationError(
            "{} cannot be represented as finite float32 samples".format(name)
        )
    return converted


def _require_bounded_waveform_size(size: int, name: str = "wave") -> None:
    if size <= 0 or size > MAX_GENERATED_WAVE_SAMPLES:
        raise AudioValidationError(
            "{} must contain no more than {} samples".format(
                name, MAX_GENERATED_WAVE_SAMPLES
            )
        )


def _validate_scale(
    value: object,
    name: str,
    minimum: Optional[float] = None,
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
            requirement = "a finite number greater than or equal to {}".format(
                minimum
            )
        raise AudioValidationError("{} must be {}".format(name, requirement))
    return float(value)


def _validate_duration(
    value: object,
    name: str,
    maximum: Optional[float] = None,
) -> float:
    result = _validate_scale(value, name, minimum=0.0)
    if maximum is not None and result > maximum:
        raise AudioValidationError(
            "{} must be no greater than {}".format(name, maximum)
        )
    return result


def encode_pcm_wav(wave: np.ndarray, sampling_rate: int) -> bytes:
    """有限なモノラル浮動小数点波形を決定的なPCM16 WAVバイト列へ変換します。"""

    wave = _require_waveform(wave)
    _require_bounded_waveform_size(wave.size)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")

    # PCMの値域に収めてから変換することで結果を決定的にし、後処理による不正なPCM値を防ぎます。
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
    expected_sampling_rate: Optional[int] = None,
) -> Tuple[Waveform, int]:
    """モノラルPCM WAVを``(float32_samples, sampling_rate)``へ変換します。
    v2サービスはPCM WAVだけを受け付け、ステレオ・圧縮・浮動小数点・破損・非WAVコンテナはサンプル返却前に拒否します。
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
            raise AudioValidationError(
                "unexpected sampling rate: {}".format(info.samplerate)
            )
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
    """波形を標準Base64で包んだPCM16 WAVへ変換します。"""

    return base64.standard_b64encode(
        encode_pcm_wav(wave, sampling_rate)
    ).decode("ascii")


def decode_pcm_wav_base64(
    wav_base64: str,
    expected_sampling_rate: Optional[int] = None,
) -> Tuple[Waveform, int]:
    """Base64 PCM WAV文字列を``(float32_samples, rate)``へ変換します。"""

    if not isinstance(wav_base64, str):
        raise AudioValidationError("wav_base64 must be a string")
    try:
        raw = base64.b64decode(wav_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise AudioValidationError("wav_base64 is not valid Base64") from error
    return decode_pcm_wav(raw, expected_sampling_rate=expected_sampling_rate)


def _core_audio_manager() -> Any:
    """単純なWAV処理を軽量に保つため、公開Coreアダプターを遅延読込します。"""

    from coeirocore.coeiro_manager import AudioManager

    return AudioManager


def trim_wave(
    wave: np.ndarray,
    sampling_rate: int,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Waveform:
    """無音を削り、両端に指定された余白を残します。
    凍結版の公開Coreは``librosa.effects.trim(top_db=30)``を使います。
    v2のtrim-bufferは検出範囲の外側に残す音声量であり、追加で削るサンプル数ではありません。
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
        import librosa

        _, detected_range = librosa.effects.trim(wave, top_db=30)
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
    """バッファ付き無音トリミングの互換エイリアスです。"""

    return trim_wave(
        wave,
        sampling_rate,
        start_trim_buffer=start_trim_buffer,
        end_trim_buffer=end_trim_buffer,
    )


def replace_pause_segments(
    wave: np.ndarray,
    sampling_rate: int,
    mora_durations: Sequence[Union[MoraDuration, dict]],
    pause_length: float,
    pause_start_trim_buffer: float = 0.0,
    pause_end_trim_buffer: float = 0.0,
) -> Waveform:
    """内部の``pau``範囲を指定された無音長へ置き換えます。
    先頭と末尾の``pau``は発話端を表すため、:func:`trim_wave`で処理します。
    ここでは内部の句読点休止だけを置き換え、両側のバッファで境界付近の音声が急に切れないようにします。
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
        raise AudioValidationError(
            "mora_durations contains an invalid item"
        ) from error

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
        if (
            item.mora.lower() == "pau"
            and start > 0
            and end < current.size
        ):
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
    """v2のRESAMPLING方式で長さを保った全体ピッチシフトを適用します。"""

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
        raise AudioProcessingError(
            "resampling pitch shift produced non-finite samples"
        )
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
    output_sampling_rate: Optional[int] = None,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Tuple[Waveform, int]:
    """決定的なv2波形処理を行い``(wave, rate)``を返します。
    処理順は公開Coreの``AudioManager.synthesis``と同じく、バッファ付きトリミング、音量、WORLDのピッチ・抑揚、前後無音、リサンプリングです。
    変更のない処理は省略し、入力波形がサンプル単位で同一になる場合を保ちます。
    """

    current = _require_waveform(wave)
    _require_bounded_waveform_size(current.size)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    volume_scale = _validate_scale(volume_scale, "volume_scale", minimum=0.0)
    pitch_scale = _validate_scale(pitch_scale, "pitch_scale")
    intonation_scale = _validate_scale(
        intonation_scale, "intonation_scale", minimum=0.0
    )
    pre_phoneme_length = _validate_duration(
        pre_phoneme_length,
        "pre_phoneme_length",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
    post_phoneme_length = _validate_duration(
        post_phoneme_length,
        "post_phoneme_length",
        maximum=MAX_PAUSE_LENGTH_SECONDS,
    )
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
    if output_sampling_rate is None:
        output_sampling_rate = sampling_rate
    output_sampling_rate = _require_sampling_rate(
        output_sampling_rate, "output_sampling_rate"
    )

    current = trim_wave(
        current,
        sampling_rate,
        start_trim_buffer=start_trim_buffer,
        end_trim_buffer=end_trim_buffer,
    )

    core_audio_manager = None
    if (
        volume_scale != 1.0
        or pitch_scale != 0.0
        or intonation_scale != 1.0
        or pre_phoneme_length != 0.0
        or post_phoneme_length != 0.0
        or output_sampling_rate != sampling_rate
    ):
        core_audio_manager = _core_audio_manager()
    try:
        if volume_scale != 1.0:
            current = core_audio_manager.volume(current, volume_scale)
        if (pitch_scale != 0.0 or intonation_scale != 1.0) and current.size:
            current = core_audio_manager.pitch_intonation(
                current,
                sampling_rate,
                pitch_scale,
                intonation_scale,
            )
        if pre_phoneme_length != 0.0 or post_phoneme_length != 0.0:
            projected_size = (
                current.size
                + int(sampling_rate * pre_phoneme_length)
                + int(sampling_rate * post_phoneme_length)
            )
            _require_bounded_waveform_size(projected_size, "processed wave")
            current = core_audio_manager.sil(
                current,
                sampling_rate,
                pre_phoneme_length,
                post_phoneme_length,
            )
        if output_sampling_rate != sampling_rate and current.size:
            projected_size = int(
                math.ceil(current.size * output_sampling_rate / sampling_rate)
            )
            _require_bounded_waveform_size(projected_size, "resampled wave")
            current = core_audio_manager.resampling(
                current, sampling_rate, output_sampling_rate
            )
    except Exception as error:
        raise AudioProcessingError("failed to process waveform") from error

    current = np.asarray(current, dtype=np.float32).reshape(-1)
    _require_bounded_waveform_size(current.size, "processed wave")
    if not np.isfinite(current).all():
        raise AudioProcessingError(
            "waveform processing produced non-finite samples"
        )
    return current.copy(), output_sampling_rate


def process_wav(
    wav_bytes: WavInput,
    sampling_rate: Optional[int] = None,
    **processing: Any,
) -> bytes:
    """PCM WAVをデコードし、処理して再エンコードします。"""

    wave, input_sampling_rate = decode_pcm_wav(
        wav_bytes, expected_sampling_rate=sampling_rate
    )
    output_wave, output_sampling_rate = process_wave(
        wave, input_sampling_rate, **processing
    )
    return encode_pcm_wav(output_wave, output_sampling_rate)


def estimate_world_f0(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
    """公開Coreの実装で有限なWORLD F0ベクトルを推定します。"""

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
    mora_durations: Sequence[Union[MoraDuration, dict]],
) -> WorldF0:
    """``/v1/estimate_f0``が返すJSON用のWORLD F0値を準備します。"""

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
        raise AudioValidationError(
            "mora_durations contains an invalid item"
        ) from error
    f0 = estimate_world_f0(wave, sampling_rate)
    return WorldF0(f0=[float(value) for value in f0], moraDurations=durations)


__all__ = [
    "AudioProcessingError",
    "AudioValidationError",
    "MAX_GENERATED_WAVE_SAMPLES",
    "MAX_PAUSE_LENGTH_SECONDS",
    "MAX_SAMPLING_RATE",
    "apply_trim_buffer",
    "decode_pcm_wav",
    "decode_pcm_wav_base64",
    "encode_pcm_wav",
    "encode_pcm_wav_base64",
    "estimate_world_f0",
    "prepare_world_f0",
    "pitch_shift_resampling",
    "process_wave",
    "process_wav",
    "replace_pause_segments",
    "trim_wave",
]
