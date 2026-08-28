"""VOICEVOX形式をCOEIROINKネイティブ合成へ変換するEngineアダプター。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from coeirocore.coeiro_manager import AudioManager
from coeirocore.query_manager import query2tokens_prosody

from ..model import AccentPhrase, AudioQuery
from .synthesis_engine_base import SynthesisEngineBase


class CoeiroinkVoicevoxAdapter(SynthesisEngineBase):
    """VOICEVOXのDTOだけを変換し、解析・推論・後処理は公開COEIROINK Coreへ委譲する。"""

    def __init__(
        self,
        speakers: str,
        supported_devices: str | None = None,
        speaker_info_dir: Path = Path("speaker_info"),
        cpu_num_threads: int | None = None,
        audio_manager: AudioManager | None = None,
        device: str | None = None,
        use_gpu: bool | None = None,
        device_index: int = 0,
        opencl_platform_index: int = 0,
        resampler: str = "resampy",
    ) -> None:
        super().__init__()

        # 循環importを避けつつ、旧use_gpuを含むデバイス解決規則は一箇所に保つ。
        from .make_synthesis_engines import resolve_device

        self._speakers = speakers
        self._supported_devices = supported_devices
        self.default_sampling_rate = 44100
        self.device = resolve_device(device=device, use_gpu=use_gpu)
        self.audio_manager = (
            audio_manager
            if audio_manager is not None
            else AudioManager(
                fs=self.default_sampling_rate,
                device=self.device,
                device_index=device_index,
                opencl_platform_index=opencl_platform_index,
                use_gpu=None,
                speaker_info_dir=speaker_info_dir,
                cpu_num_threads=cpu_num_threads,
                resampler=resampler,
            )
        )

    @property
    def speakers(self) -> str:
        return self._speakers

    @property
    def supported_devices(self) -> str | None:
        return self._supported_devices

    def replace_phoneme_length(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        return accent_phrases

    def replace_mora_pitch(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        return accent_phrases

    def initialize_speaker_synthesis(self, speaker_id: int, skip_reinit: bool) -> None:
        self.audio_manager.initialize_speaker(
            style_id=speaker_id,
            skip_reinit=skip_reinit,
        )

    def is_initialized_speaker_synthesis(self, speaker_id: int) -> bool:
        return self.audio_manager.is_speaker_initialized(style_id=speaker_id)

    def _synthesis_impl(self, query: AudioQuery, speaker_id: int) -> np.ndarray:
        tokens = query2tokens_prosody(query)
        wave = self.audio_manager.synthesis(
            text=tokens,
            style_id=speaker_id,
            speed_scale=query.speedScale,
            volume_scale=query.volumeScale,
            pitch_scale=query.pitchScale,
            intonation_scale=query.intonationScale,
            pre_phoneme_length=query.prePhonemeLength,
            post_phoneme_length=query.postPhonemeLength,
            output_sampling_rate=query.outputSamplingRate,
            pause_length=query.pauseLength,
            pause_length_scale=query.pauseLengthScale,
        )
        if query.outputStereo and wave.ndim == 1:
            wave = np.column_stack((wave, wave))
        return wave


__all__ = ["CoeiroinkVoicevoxAdapter"]
