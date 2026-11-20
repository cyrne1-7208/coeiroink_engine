import json
from pathlib import Path
from typing import Literal, cast

from coeirocore import __version__ as coeirocore_version
from coeirocore.coeiro_manager import AudioManager
from coeirocore.devices import DeviceBackend, get_supported_device_capabilities
from packaging.version import Version

from ..utility import engine_root
from .coeiroink_adapter import CoeiroinkVoicevoxAdapter
from .synthesis_engine_base import SynthesisEngineBase

Device = Literal["cpu", "cuda", "directml", "opencl"]
SUPPORTED_DEVICES = ("cpu", "cuda", "directml", "opencl")


# 旧GPUフラグを含む指定を、Coreへ渡すデバイス名へ正規化する。
def resolve_device(
    device: str | None = None,
    use_gpu: bool | None = None,
) -> Device:
    # 旧use_gpuと新deviceの併用は優先順位を設けず、曖昧な設定として拒否する。
    if device is not None and use_gpu is not None:
        raise ValueError("--deviceと--use_gpuは同時に指定できません。")

    if use_gpu is not None:
        return "cuda" if use_gpu else "cpu"

    if device is None:
        return "cpu"
    if device not in SUPPORTED_DEVICES:
        supported = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(
            f"未対応のデバイスです: {device} ({supported}から指定してください)"
        )
    return cast(Device, device)


def _resolve_speaker_info_dir(
    speaker_info_dir: Path | None,
    voicevox_dir: Path | None,
) -> Path:
    if speaker_info_dir is None:
        speaker_info_dir = (
            voicevox_dir if voicevox_dir is not None else engine_root()
        ) / "speaker_info"
    return speaker_info_dir.expanduser().resolve()


# 起動設定を整理し、ネイティブAPIと互換APIで共有するCoreを作る。
def make_audio_manager(
    *,
    speaker_info_dir: Path,
    device: str | None = None,
    use_gpu: bool | None = None,
    device_index: int = 0,
    opencl_platform_index: int = 0,
    cpu_num_threads: int | None = None,
    resampler: str = "resampy",
    max_loaded_models: int | None = 1,
    generator_only: bool = True,
    voice_smoothing: bool = False,
    preload_models: bool = True,
    dangerous_skip_check: bool = False,
) -> AudioManager:
    selected_device = resolve_device(device=device, use_gpu=use_gpu)
    return AudioManager(
        fs=44100,
        device=selected_device,
        device_index=device_index,
        opencl_platform_index=opencl_platform_index,
        use_gpu=None,
        speaker_info_dir=speaker_info_dir.expanduser().resolve(),
        cpu_num_threads=0 if cpu_num_threads in (None, 0) else cpu_num_threads,
        resampler=resampler,
        max_loaded_models=max_loaded_models,
        generator_only=generator_only,
        voice_smoothing=voice_smoothing,
        preload_models=preload_models,
        dangerous_skip_check=dangerous_skip_check,
    )


# Coreで検証済みのメタデータを再走査せず、VOICEVOX互換のJSONに変換する。
def _core_metas(audio_manager: AudioManager) -> str:
    return json.dumps(
        audio_manager.meta_manager.get_metas_dict(),
        ensure_ascii=False,
    )


def _core_supported_devices() -> str:
    capabilities = get_supported_device_capabilities()
    return json.dumps(
        {
            "cpu": capabilities[DeviceBackend.CPU].available,
            "cuda": capabilities[DeviceBackend.CUDA].available,
            "dml": capabilities[DeviceBackend.DIRECTML].available,
            # OpenCLはCOEIROINK拡張として保持し、VOICEVOXの既存DTOでは読み飛ばされる。
            "opencl": capabilities[DeviceBackend.OPENCL].available,
        },
        ensure_ascii=False,
    )


def make_synthesis_engines(
    use_gpu: bool | None = None,
    voicelib_dirs: list[Path] | None = None,
    voicevox_dir: Path | None = None,
    runtime_dirs: list[Path] | None = None,
    cpu_num_threads: int | None = None,
    enable_mock: bool = True,
    max_loaded_models: int | None = 1,
    generator_only: bool = True,
    speaker_info_dir: Path | None = None,
    device: str | None = None,
    device_index: int = 0,
    opencl_platform_index: int = 0,
    resampler: str = "resampy",
    audio_manager: AudioManager | None = None,
    voice_smoothing: bool = False,
    dangerous_skip_check: bool = False,
) -> dict[str, SynthesisEngineBase]:
    """
    Coreを準備し、音声合成エンジンを生成する。

    Parameters
    ----------
    use_gpu : bool, optional, default=None
        旧互換の引数で、Trueはcuda、Falseはcpuに変換される。
    voicelib_dirs : list[Path], optional, default=None
        旧Engineとの呼び出し互換性のために受け取る。Python版Coreでは使用しない。
    voicevox_dir : Path, optional, default=None
        コンパイル済みのvoicevox、またはvoicevox_engineがあるディレクトリ
        speaker_info_dirを省略した場合は、このディレクトリのspeaker_infoを使う。
    runtime_dirs : list[Path], optional, default=None
        旧Engineとの呼び出し互換性のために受け取る。Python版Coreでは使用しない。
    cpu_num_threads : int, optional, default=None
        音声ライブラリが推論に用いるCPUスレッド数
        Noneまたは0のとき、ライブラリ側の挙動により論理コア数の半分、または物理コア数が指定される。
    enable_mock : bool, optional, default=True
        旧Engineとの互換用で、指定しても動作に影響しない。
    max_loaded_models : int | None, optional, default=1
        直近に使ったMYCOEIROINKモデルの保持上限で、Noneでは全モデルを起動時に読み込む。
    generator_only : bool, optional, default=True
        VITSの推論用generatorだけを読み込むか
        非対応モデルではFalseを指定する。
    speaker_info_dir : Path, optional, default=None
        MYCOEIROINKを展開したspeaker_infoディレクトリ
    device : str, optional, default=None
        Coreで使用するデバイスで、cpu、cuda、directml、openclから指定する。
    device_index : int, optional, default=0
        Coreで使用するデバイス番号
    opencl_platform_index : int, optional, default=0
        OpenCLで使用するプラットフォーム番号
    resampler : str, optional, default="resampy"
        出力サンプリングレート変換に使う実装
    audio_manager : AudioManager, optional, default=None
        ネイティブAPIと共有するCoreで、未指定の場合はこの関数で生成する。
    voice_smoothing : bool, optional, default=False
        Coreの実験的な音声補正を有効化するか
    dangerous_skip_check : bool, optional, default=False
        推定メモリ不足を警告だけにし、メモリ不足による拒否と自動解放を省略するか

    Returns
    -------
    dict[str, SynthesisEngineBase]
        Coreのバージョンをキーにした音声合成エンジン
    """
    if audio_manager is None:
        speaker_info_dir = _resolve_speaker_info_dir(speaker_info_dir, voicevox_dir)
        audio_manager = make_audio_manager(
            speaker_info_dir=speaker_info_dir,
            device=device,
            use_gpu=use_gpu,
            device_index=device_index,
            opencl_platform_index=opencl_platform_index,
            cpu_num_threads=cpu_num_threads,
            resampler=resampler,
            max_loaded_models=max_loaded_models,
            generator_only=generator_only,
            voice_smoothing=voice_smoothing,
            dangerous_skip_check=dangerous_skip_check,
        )
    elif max_loaded_models is None:
        audio_manager.initialize_all_speakers()

    # デバイス初期化とモデル管理はCoreに集約し、VOICEVOX互換層には生成済みのCoreだけを渡す。
    return {
        coeirocore_version: CoeiroinkVoicevoxAdapter(
            speakers=_core_metas(audio_manager),
            supported_devices=_core_supported_devices(),
            audio_manager=audio_manager,
        )
    }


# `+cpu`などのローカル接尾辞を含むCoreバージョンを比較可能な値へ変換する。
def _version_key(version: str) -> Version:
    return Version(version)


# 音声合成エンジンが1つ以上あることを確かめ、最新のCoreバージョンを返す。
def get_latest_core_version(synthesis_engines: dict[str, SynthesisEngineBase]) -> str:
    if not synthesis_engines:
        raise RuntimeError("音声合成エンジンがありません。")
    return max(synthesis_engines, key=_version_key)
