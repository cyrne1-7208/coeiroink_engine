from pathlib import Path
from typing import List, Optional

import numpy as np
from coeirocore.coeiro_manager import AudioManager
from coeirocore.query_manager import query2tokens_prosody

from ...model import AccentPhrase, AudioQuery
from ...synthesis_engine import SynthesisEngineBase


class MockSynthesisEngine(SynthesisEngineBase):
    def __init__(
        self,
        speakers: str,
        supported_devices: Optional[str] = None,
        speaker_info_dir: Path = Path("speaker_info"),
        cpu_num_threads: Optional[int] = None,
        audio_manager: Optional[AudioManager] = None,
    ):
        super().__init__()

        self._speakers = speakers
        self._supported_devices = supported_devices
        self.default_sampling_rate = 44100

        # 名前はMockですが、実際の推論と音声後処理は公開CoreのAudioManagerへ委譲します。
        self.audio_manager = (
            audio_manager
            if audio_manager is not None
            else AudioManager(
                fs=self.default_sampling_rate,
                use_gpu=False,
                speaker_info_dir=speaker_info_dir,
                cpu_num_threads=cpu_num_threads,
            )
        )

    @property
    def speakers(self) -> str:
        return self._speakers

    @property
    def supported_devices(self) -> Optional[str]:
        return self._supported_devices

    def replace_phoneme_length(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        return accent_phrases

    def replace_mora_pitch(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        return accent_phrases

    def initialize_speaker_synthesis(self, speaker_id: int, skip_reinit: bool):
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
        )
        if query.outputStereo and wave.ndim == 1:
            wave = np.column_stack((wave, wave))
        return wave
