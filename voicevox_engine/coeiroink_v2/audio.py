# COEIROINK v2 HTTP APIで使う、入力を検証する小さな音声プリミティブ
#
# このモジュールにはモデルやHTTPのコードを意図的に含めない。
# 波形はHTTPの境界を越えるまで一次元の浮動小数点NumPy配列のまま扱い、境界でモノラルのPCM16 WAVデータへエンコードする。
# 数値的な後処理は公開Coreの`AudioManager`の実装に委譲し、EngineとCoreで同じ音声処理の規則を保つ。

import base64
import binascii
import io
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import soundfile

from .models import MoraDuration, WorldF0


# v2 APIで使えない音声の値が渡された場合のエラー
class AudioValidationError(ValueError):
    pass


# 公開の音声プリミティブが有限のサンプルを生成できない場合のエラー
class AudioProcessingError(RuntimeError):
    pass


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


# モーラごとの継続長の列を検証し、MoraDurationの一覧に揃える。
def _validate_mora_durations(
    mora_durations: Sequence[MoraDuration | dict],
) -> list[MoraDuration]:
    if isinstance(mora_durations, (str, bytes)):
        raise AudioValidationError("mora_durations must be a sequence")
    try:
        return [
            item
            if isinstance(item, MoraDuration)
            else MoraDuration.model_validate(item)
            for item in mora_durations
        ]
    except (TypeError, ValueError) as error:
        raise AudioValidationError("mora_durations contains an invalid item") from error


# 有限値のモノラル浮動小数点波形を決定的なPCM16 WAVのバイト列へエンコードする。
def encode_pcm_wav(wave: np.ndarray, sampling_rate: int) -> bytes:
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


# モノラルのPCM WAVをデコードし、float32のサンプルとサンプリングレートの組を返す。
#
# v2のサービスはPCM WAVの入力だけを受け付ける。
# ステレオ、圧縮形式、浮動小数点形式、壊れたデータ、WAV以外のコンテナは、サンプルを返す前に拒否する。
def decode_pcm_wav(
    wav_bytes: WavInput,
    expected_sampling_rate: int | None = None,
) -> tuple[Waveform, int]:
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
        with soundfile.SoundFile(io.BytesIO(raw)) as source:
            if source.format.upper() != "WAV":
                raise AudioValidationError("WAV container is required")
            if not source.subtype.upper().startswith("PCM"):
                raise AudioValidationError("PCM WAV data is required")
            if source.channels != 1:
                raise AudioValidationError("only mono WAV data is supported")
            _require_bounded_waveform_size(source.frames, "WAV data")
            sampling_rate = source.samplerate
            if (
                expected_sampling_rate is not None
                and sampling_rate != expected_sampling_rate
            ):
                raise AudioValidationError(f"unexpected sampling rate: {sampling_rate}")
            result = source.read(dtype="float32")
    except AudioValidationError:
        raise
    except Exception as error:
        raise AudioValidationError("wav_bytes is not a readable PCM WAV") from error

    if result.size == 0:
        raise AudioValidationError("WAV data must not be empty")
    if not np.isfinite(result).all():
        raise AudioValidationError("decoded WAV contains non-finite samples")
    return result, sampling_rate


# 波形をPCM16 WAVにし、標準のBase64でエンコードする。
def encode_pcm_wav_base64(wave: np.ndarray, sampling_rate: int) -> str:
    return base64.standard_b64encode(encode_pcm_wav(wave, sampling_rate)).decode(
        "ascii"
    )


# Base64のPCM WAV文字列をデコードし、float32のサンプルとサンプリングレートの組を返す。
def decode_pcm_wav_base64(
    wav_base64: str,
    expected_sampling_rate: int | None = None,
) -> tuple[Waveform, int]:
    if not isinstance(wav_base64, str):
        raise AudioValidationError("wav_base64 must be a string")
    try:
        raw = base64.b64decode(wav_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise AudioValidationError("wav_base64 is not valid Base64") from error
    return decode_pcm_wav(raw, expected_sampling_rate=expected_sampling_rate)


# WAVだけを扱う補助関数を軽量に保つため、公開Coreのアダプターを遅延読み込みする。
def _core_audio_manager() -> Any:
    from coeirocore.coeiro_manager import AudioManager

    return AudioManager


# 旧Python APIから、モデルを生成せずにCoreの波形プリミティブだけを利用するためのクラス
class _CompatibilityWaveProcessor:
    @staticmethod
    def get_world_f0(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
        return _core_audio_manager().get_world_f0(wave, sampling_rate)

    @staticmethod
    def get_world(
        wave: np.ndarray, sampling_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# 前後に指定された長さの音声を残しながら無音を取り除く。
#
# Coreと同じRMS判定を使う。trim-buffer値は検出範囲の外側へ残す音声長であり、追加で破棄する長さではない。
def trim_wave(
    wave: np.ndarray,
    sampling_rate: int,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Waveform:
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


# 互換用の別名で、trim_waveと同じく前後の音声を残して無音を取り除く。
def apply_trim_buffer(
    wave: np.ndarray,
    sampling_rate: int,
    start_trim_buffer: float = 0.0,
    end_trim_buffer: float = 0.0,
) -> Waveform:
    return trim_wave(
        wave,
        sampling_rate,
        start_trim_buffer=start_trim_buffer,
        end_trim_buffer=end_trim_buffer,
    )


# 文中の`pau`の区間を、指定された長さの無音に置き換える。
#
# 先頭と末尾の`pau`は発話の端を表し、`trim_wave`で処理する。ここでは文中の句読点による休止だけを置き換える。
# 2つの休止バッファーは元の休止を両側に少しずつ残し、継続長の境界が音声と重なるモデルでも音声が急に途切れないようにする。
def replace_pause_segments(
    wave: np.ndarray,
    sampling_rate: int,
    mora_durations: Sequence[MoraDuration | dict],
    pause_length: float,
    pause_start_trim_buffer: float = 0.0,
    pause_end_trim_buffer: float = 0.0,
) -> Waveform:
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
    durations = _validate_mora_durations(mora_durations)

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
    return result


# v2のRESAMPLING方式で長さを変えずに全体の音高を変える。
def pitch_shift_resampling(
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float,
) -> Waveform:
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


# 旧Python APIの引数を現行の一元化された波形後処理へ変換する。
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


# 旧Python APIとしてPCM WAVをデコードし、現行処理後に再エンコードする。
def process_wav(
    wav_bytes: WavInput,
    sampling_rate: int | None = None,
    **processing: Any,
) -> bytes:
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


# 公開Coreの実装で有限値のWORLD F0ベクトルを推定する。
def estimate_world_f0(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
    wave = _require_waveform(wave)
    sampling_rate = _require_sampling_rate(sampling_rate, "sampling_rate")
    try:
        f0 = _core_audio_manager().get_world_f0(wave.astype(np.float64), sampling_rate)
    except Exception as error:
        raise AudioProcessingError("failed to estimate WORLD F0") from error
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    if not np.isfinite(f0).all():
        raise AudioProcessingError("WORLD F0 contains non-finite values")
    return f0.copy()


# `/v1/estimate_f0`が返す、JSONに変換できるWORLD F0の応答を作る。
def prepare_world_f0(
    wave: np.ndarray,
    sampling_rate: int,
    mora_durations: Sequence[MoraDuration | dict],
) -> WorldF0:
    durations = _validate_mora_durations(mora_durations)
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
