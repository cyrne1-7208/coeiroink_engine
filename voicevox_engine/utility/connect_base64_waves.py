import base64
import binascii
import io

import numpy as np
import soundfile
from scipy.signal import resample

MAX_CONNECTED_WAVE_BYTES = 128 * 1024 * 1024
MAX_CONNECTED_WAVE_SAMPLES = 30_000_000
MAX_CONNECTED_WAVE_SECONDS = 600.0
MAX_CONNECTED_SAMPLING_RATE = 384_000


class ConnectBase64WavesException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _channel_count(wave: np.ndarray) -> int:
    if wave.ndim == 1:
        return 1
    if wave.ndim == 2:
        channels = int(wave.shape[1])
        if channels in (1, 2):
            return channels
    raise ConnectBase64WavesException(
        "wavファイルはモノラルまたはステレオにしてください"
    )


def decode_base64_waves(waves: list[str]) -> list[tuple[np.ndarray, int]]:
    """
    base64エンコードされた複数のwavデータをデコードする
    Parameters
    ----------
    waves: list[str]
        base64エンコードされたwavデータのリスト
    Returns
    -------
    waves_nparray_sr: List[Tuple[np.ndarray, int]]
        (NumPy配列の音声波形データ, サンプリングレート) 形式のタプルのリスト
    """
    if len(waves) == 0:
        raise ConnectBase64WavesException("wavファイルが含まれていません")

    waves_nparray_sr = []
    decoded_bytes = 0
    decoded_samples = 0
    total_seconds = 0.0
    for wave in waves:
        if not isinstance(wave, str):
            raise ConnectBase64WavesException("base64データは文字列で指定してください")
        # 上限確認を文字列の複製前に行い、巨大入力を拒否する段階で同量のASCII bytesを確保しない。
        estimated_bytes = (len(wave) + 3) // 4 * 3
        if decoded_bytes + estimated_bytes > MAX_CONNECTED_WAVE_BYTES:
            raise ConnectBase64WavesException(
                "wavファイルの合計データ量が上限を超えています"
            )
        try:
            encoded = wave.encode("ascii")
            wav_bin = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, TypeError, ValueError) as error:
            raise ConnectBase64WavesException("base64デコードに失敗しました") from error
        decoded_bytes += len(wav_bin)
        if decoded_bytes > MAX_CONNECTED_WAVE_BYTES:
            raise ConnectBase64WavesException(
                "wavファイルの合計データ量が上限を超えています"
            )
        try:
            info = soundfile.info(io.BytesIO(wav_bin))
            if (
                info.frames <= 0
                or info.samplerate <= 0
                or info.samplerate > MAX_CONNECTED_SAMPLING_RATE
                or info.channels not in (1, 2)
            ):
                raise ConnectBase64WavesException(
                    "wavファイルの形式またはサイズが対応範囲外です"
                )
            decoded_samples += int(info.frames) * int(info.channels)
            total_seconds += float(info.frames) / float(info.samplerate)
            if decoded_samples > MAX_CONNECTED_WAVE_SAMPLES:
                raise ConnectBase64WavesException(
                    "wavファイルの合計サンプル数が上限を超えています"
                )
            if total_seconds > MAX_CONNECTED_WAVE_SECONDS:
                raise ConnectBase64WavesException(
                    "wavファイルの合計時間が上限を超えています"
                )
            _data = soundfile.read(io.BytesIO(wav_bin))
        except ConnectBase64WavesException:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ConnectBase64WavesException(
                "wavファイルを読み込めませんでした"
            ) from error
        waves_nparray_sr.append(_data)

    return waves_nparray_sr


def connect_base64_waves(waves: list[str]) -> tuple[np.ndarray, int]:
    """検証済みWAVを最大サンプリングレートとチャンネル数へ揃え、時間方向に連結する。

    モノラルとステレオが混在する場合はモノラルをステレオへ複製し、変換後もモジュール定数の合計サンプル数上限を適用する。
    """

    waves_nparray_sr = decode_base64_waves(waves)

    max_sampling_rate = max(sr for _, sr in waves_nparray_sr)
    max_channels = max(_channel_count(wave) for wave, _ in waves_nparray_sr)
    output_samples = sum(
        (max_sampling_rate * len(wave) // sampling_rate) * max_channels
        for wave, sampling_rate in waves_nparray_sr
    )
    if output_samples > MAX_CONNECTED_WAVE_SAMPLES:
        raise ConnectBase64WavesException(
            "結合後のwavファイルがサンプル数上限を超えています"
        )

    waves_nparray_list = []
    for nparray, sr in waves_nparray_sr:
        if sr != max_sampling_rate:
            nparray = resample(nparray, max_sampling_rate * len(nparray) // sr)
        channels = _channel_count(nparray)
        if channels < max_channels:
            nparray = np.array([nparray, nparray]).T
        waves_nparray_list.append(nparray)

    return np.concatenate(waves_nparray_list), max_sampling_rate
