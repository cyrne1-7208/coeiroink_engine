from pathlib import Path
from typing import Literal, cast

from coeirocore import __version__ as coeirocore_version

from ..utility import engine_root
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
        旧Engineとの呼び出し互換性のために受け取る。モデルは要求時に読み込む。
    speaker_info_dir: Path, optional, default=None
        MYCOEIROINKを展開したspeaker_infoディレクトリ
    device: str, optional, default=None
        Coreで使用するデバイス。cpu、cuda、directml、openclから指定する。
    device_index: int, optional, default=0
        Coreで使用するデバイス番号。
    opencl_platform_index: int, optional, default=0
        OpenCLで使用するプラットフォーム番号。
    """
    selected_device = resolve_device(device=device, use_gpu=use_gpu)

    if cpu_num_threads == 0 or cpu_num_threads is None:
        cpu_num_threads = 0

    if speaker_info_dir is None:
        speaker_info_dir = (
            voicevox_dir if voicevox_dir is not None else engine_root()
        ) / "speaker_info"
    speaker_info_dir = speaker_info_dir.expanduser().resolve()

    # dev.coreという旧モジュール名は維持するが、ここでは実際のCoreからメタデータとデバイス能力だけを取得する。
    from ..dev.core import metas as mock_metas
    from ..dev.core import supported_devices as mock_supported_devices
    from .coeiroink_adapter import CoeiroinkVoicevoxAdapter

    # デバイス名と番号を分離したままCoreへ渡し、バックエンド自体を利用できない場合はEngine側でCPUへ置き換えない。
    return {
        coeirocore_version: CoeiroinkVoicevoxAdapter(
            speakers=mock_metas(speaker_info_dir=speaker_info_dir),
            supported_devices=mock_supported_devices(),
            speaker_info_dir=speaker_info_dir,
            cpu_num_threads=cpu_num_threads,
            device=selected_device,
            device_index=device_index,
            opencl_platform_index=opencl_platform_index,
        )
    }
