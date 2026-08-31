import json
from pathlib import Path
from typing import Literal, cast

from coeirocore import __version__ as coeirocore_version
from coeirocore.coeiro_manager import AudioManager
from coeirocore.devices import DeviceBackend, get_supported_device_capabilities

from ..utility import engine_root
from .coeiroink_adapter import CoeiroinkVoicevoxAdapter
from .synthesis_engine_base import SynthesisEngineBase

Device = Literal["cpu", "cuda", "directml", "opencl"]
SUPPORTED_DEVICES = ("cpu", "cuda", "directml", "opencl")


def resolve_device(
    device: str | None = None,
    use_gpu: bool | None = None,
) -> Device:
    """旧GPUフラグを含む指定を、Coreへ渡すデバイス名へ正規化する。"""

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


def make_audio_manager(
    *,
    speaker_info_dir: Path,
    device: str | None = None,
    use_gpu: bool | None = None,
    device_index: int = 0,
    opencl_platform_index: int = 0,
    cpu_num_threads: int | None = None,
    resampler: str = "resampy",
    load_all_models: bool = False,
) -> AudioManager:
    """起動設定を正規化し、ネイティブAPIと互換APIが共有するCoreを生成する。"""

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
        load_all_models=load_all_models,
    )


def _core_metas(audio_manager: AudioManager) -> str:
    """Coreが検証済みのメタデータを再走査せずVOICEVOX形式へ渡す。"""

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
    load_all_models: bool = False,
    speaker_info_dir: Path | None = None,
    device: str | None = None,
    device_index: int = 0,
    opencl_platform_index: int = 0,
    resampler: str = "resampy",
    audio_manager: AudioManager | None = None,
) -> dict[str, SynthesisEngineBase]:
    """
    音声ライブラリをロードして、音声合成エンジンを生成

    Parameters
    ----------
    use_gpu: bool, optional
        旧互換引数。Trueはcuda、Falseはcpuに変換される。
    voicelib_dirs: List[Path], optional, default=None
        旧Engineとの呼び出し互換性のために受け取る。Python版Coreでは使用しない。
    voicevox_dir: Path, optional, default=None
        コンパイル済みのvoicevox、またはvoicevox_engineがあるディレクトリ
    runtime_dirs: List[Path], optional, default=None
        旧Engineとの呼び出し互換性のために受け取る。Python版Coreでは使用しない。
    cpu_num_threads: int, optional, default=None
        音声ライブラリが、推論に用いるCPUスレッド数を設定する
        Noneのとき、ライブラリ側の挙動により論理コア数の半分か、物理コア数が指定される
    enable_mock: bool, optional, default=True
        旧Engineとの呼び出し互換性のために受け取る。
    load_all_models: bool, optional, default=False
        全MYCOEIROINKモデルを起動時に読み込み、モデル切替後も保持する。
    speaker_info_dir: Path, optional, default=None
        MYCOEIROINKを展開したspeaker_infoディレクトリ
    device: str, optional, default=None
        Coreで使用するデバイス。cpu、cuda、directml、openclから指定する。
    device_index: int, optional, default=0
        Coreで使用するデバイス番号。
    opencl_platform_index: int, optional, default=0
        OpenCLで使用するプラットフォーム番号。
    resampler: str, optional, default="resampy"
        出力サンプリングレート変換に使う実装。
    audio_manager: AudioManager, optional, default=None
        ネイティブAPIと共有する生成済みCore。未指定時は旧呼び出し互換のためこのfactoryで生成する。
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
            load_all_models=load_all_models,
        )
    elif load_all_models:
        audio_manager.initialize_all_speakers()

    # VOICEVOX互換層には生成済みCoreだけを渡し、デバイス初期化やモデル管理を持たせない。
    return {
        coeirocore_version: CoeiroinkVoicevoxAdapter(
            speakers=_core_metas(audio_manager),
            supported_devices=_core_supported_devices(),
            audio_manager=audio_manager,
        )
    }
