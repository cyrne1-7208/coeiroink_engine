import json
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from coeirocore.coeiro_manager import MetaManager
from pyopenjtalk import tts
from scipy.signal import resample

DUMMY_TEXT = "これはダミーのテキストです"


def initialize(path: str, use_gpu: bool, *args: list[Any]) -> None:
    pass


def yukarin_s_forward(length: int, **kwargs: dict[str, Any]) -> np.ndarray:
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, yukarin_s_forward() is a mock. Return values are incorrect.",
    )
    return np.ones(length) / 5


def yukarin_sa_forward(length: int, **kwargs: dict[str, Any]) -> np.ndarray:
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, yukarin_sa_forward() is a mock. Return values are incorrect.",
    )
    return np.ones((1, length)) * 5


def decode_forward(length: int, **kwargs: dict[str, Any]) -> np.ndarray:
    """
    合成音声の波形データをNumPy配列で返します。ただし、常に固定の文言を読み上げます（DUMMY_TEXT）
    参照→SynthesisEngine のdocstring [Mock]

    Parameters
    ----------
    length : int
        フレームの長さ

    Returns
    -------
    wave : np.ndarray
        音声合成した波形データ

    Note
    -------
        ここで行う音声合成では、調声（ピッチ等）を反映しない
        また、入力内容によらず常に固定の文言を読み上げる

        # pyopenjtalk.tts()の出力仕様
        dtype=np.float64, 16 bit, mono 48000 Hz

        # resampleの説明
        非モックdecode_forwardと合わせるために、出力を24kHzに変換した。
    """
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, decode_forward() is a mock. Return values are incorrect.",
    )
    wave, _sr = tts(DUMMY_TEXT)
    return resample(
        wave.astype("int16"),
        24000 * len(wave) // 48000,
    )


def metas(speaker_info_dir: Path = Path("speaker_info")) -> str:
    return json.dumps(
        MetaManager(speaker_info_dir=speaker_info_dir).get_metas_dict(),
        ensure_ascii=False,
    )


def supported_devices() -> str:
    # OpenCLは内部拡張として保持し、VOICEVOX互換APIでは既存schemaにより3項目へ絞られる。
    from coeirocore.devices import (
        DeviceBackend,
        get_supported_device_capabilities,
    )

    capabilities = get_supported_device_capabilities()
    return json.dumps(
        {
            "cpu": capabilities[DeviceBackend.CPU].available,
            "cuda": capabilities[DeviceBackend.CUDA].available,
            "dml": capabilities[DeviceBackend.DIRECTML].available,
            "opencl": capabilities[DeviceBackend.OPENCL].available,
        },
        ensure_ascii=False,
    )
