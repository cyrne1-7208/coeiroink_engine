# Engineテストで共有するアプリ構築フィクスチャとv2用フェイク

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from coeirocore.coeiro_manager import PredictionResult
from fastapi.testclient import TestClient

from run import generate_app
from voicevox_engine.setting import SettingLoader
from voicevox_engine.synthesis_engine.coeiroink_adapter import CoeiroinkVoicevoxAdapter

ENGINE_ROOT = Path(__file__).resolve().parents[1]
SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"
SECOND_SPEAKER_UUID = "00000000-0000-4000-8000-000000000002"
STYLE_ID = 1234567890


# 実モデルに依存せず統合FastAPIアプリとTestClientを構築するクラス
@dataclass(frozen=True, slots=True)
class EngineClientFactory:
    tmp_path: Path
    speaker_uuid: str = SPEAKER_UUID
    second_speaker_uuid: str = SECOND_SPEAKER_UUID
    style_id: int = STYLE_ID

    def create_speaker_fixture(
        self,
        speaker_info_dir: Path,
        *,
        folder_name: str | None = None,
        speaker_uuid: str | None = None,
        speaker_name: str = "テスト話者",
        style_id: int | None = None,
        style_name: str = "テスト",
        metadata_version: object = "0.0.1",
    ) -> Path:
        resolved_uuid = speaker_uuid or self.speaker_uuid
        resolved_style_id = self.style_id if style_id is None else style_id
        speaker_dir = speaker_info_dir / (folder_name or resolved_uuid)
        (speaker_dir / "icons").mkdir(parents=True)
        (speaker_dir / "voice_samples").mkdir()
        (speaker_dir / "metas.json").write_text(
            json.dumps(
                {
                    "speakerName": speaker_name,
                    "speakerUuid": resolved_uuid,
                    "styles": [{"styleName": style_name, "styleId": resolved_style_id}],
                    "version": metadata_version,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (speaker_dir / "policy.md").write_text("test policy", encoding="utf-8")
        (speaker_dir / "portrait.png").write_bytes(b"portrait")
        (speaker_dir / "icons" / f"{resolved_style_id}.png").write_bytes(b"icon")
        for index in range(1, 4):
            (
                speaker_dir / "voice_samples" / f"{resolved_style_id}_{index:03}.wav"
            ).write_bytes(f"sample {index}".encode())
        return speaker_dir

    def create_client(
        self,
        *,
        folder_name: str | None = None,
        cancellable_engine: object = None,
        device: str = "cpu",
        duplicate_style: bool = False,
        disable_mutable_api: bool = False,
        metadata_version: object = "0.0.1",
    ) -> tuple[TestClient, Mock]:
        speaker_info_dir = self.tmp_path / "speaker_info"
        self.create_speaker_fixture(
            speaker_info_dir,
            folder_name=folder_name,
            metadata_version=metadata_version,
        )
        if duplicate_style:
            self.create_speaker_fixture(
                speaker_info_dir,
                folder_name="duplicate-speaker",
                speaker_uuid=self.second_speaker_uuid,
                speaker_name="重複話者",
                metadata_version=metadata_version,
            )

        core_metas = json.dumps(
            [
                {
                    "name": "テスト話者",
                    "speaker_uuid": self.speaker_uuid,
                    "styles": [{"name": "テスト", "id": self.style_id}],
                    "version": str(metadata_version),
                }
            ],
            ensure_ascii=False,
        )
        audio_manager = Mock()
        audio_manager.synthesis.return_value = np.linspace(
            -0.1, 0.1, 4410, dtype=np.float32
        )
        audio_manager.fs = 44100
        audio_manager.device = device
        audio_manager.predict.return_value = np.linspace(
            -0.1, 0.1, 4410, dtype=np.float32
        )
        engine = CoeiroinkVoicevoxAdapter(
            speakers=core_metas,
            supported_devices=json.dumps({"cpu": True, "cuda": False}),
            audio_manager=audio_manager,
        )
        app = generate_app(
            synthesis_engines={"0.0.0": engine},
            latest_core_version="0.0.0",
            setting_loader=SettingLoader(self.tmp_path / "setting.yml"),
            audio_manager=audio_manager,
            root_dir=ENGINE_ROOT,
            speaker_info_dir=speaker_info_dir,
            cancellable_engine=cancellable_engine,
            disable_mutable_api=disable_mutable_api,
        )
        return TestClient(app), audio_manager

    def post_audio_query(self, client: TestClient) -> dict[str, object]:
        response = client.post(
            "/voicevox/audio_query",
            params={
                "text": "テストです",
                "speaker": self.style_id,
                "enable_katakana_english": True,
            },
        )
        assert response.status_code == 200
        return response.json()


@pytest.fixture
def engine_client_factory(tmp_path: Path) -> EngineClientFactory:
    return EngineClientFactory(tmp_path=tmp_path)


# COEIROINK v2ルーター・波形後処理テストで共有するCore AudioManager代替
class V2FakeAudioManager:
    fs = 16000
    device = "cpu"

    def __init__(
        self,
        *,
        voice_smoothing: bool = False,
        skip_trailing_silence: bool = False,
        expected_speaker_uuid: str = "00000000-0000-0000-0000-000000000001",
        expected_style_id: int = 7,
    ) -> None:
        self.voice_smoothing = voice_smoothing
        self.skip_trailing_silence = skip_trailing_silence
        self._expected_speaker_uuid = expected_speaker_uuid
        self._expected_style_id = expected_style_id
        self.prediction_calls: list[dict[str, object]] = []
        self.smoothing_calls: list[dict[str, object]] = []

    def predict(
        self,
        text: list[str],
        style_id: int,
        speed_scale: float,
        speaker_uuid: str,
        **options: object,
    ) -> np.ndarray:
        self.prediction_calls.append(
            {
                "kind": "predict",
                "text": list(text),
                "style_id": style_id,
                "speed_scale": speed_scale,
                "speaker_uuid": speaker_uuid,
                **options,
            }
        )
        return np.linspace(-0.25, 0.25, 8, dtype=np.float32)

    def predict_with_duration(
        self,
        text: list[str],
        style_id: int,
        speed_scale: float,
        speaker_uuid: str,
        **options: object,
    ) -> PredictionResult:
        self.prediction_calls.append(
            {
                "kind": "predict_with_duration",
                "text": list(text),
                "style_id": style_id,
                "speed_scale": speed_scale,
                "speaker_uuid": speaker_uuid,
                **options,
            }
        )
        return PredictionResult(
            wav=np.linspace(-0.25, 0.25, 8, dtype=np.float32),
            duration_frames=[1, 2, 1],
        )

    def smooth_voice(
        self,
        wave: np.ndarray,
        tokens: list[str],
        duration_frames: list[int],
        *,
        hop_length: int,
    ) -> np.ndarray:
        self.smoothing_calls.append(
            {
                "wave": np.array(wave, copy=True),
                "tokens": list(tokens),
                "duration_frames": list(duration_frames),
                "hop_length": hop_length,
            }
        )
        return wave + np.float32(0.5)

    def get_hop_length(self, style_id: int, speaker_uuid: str) -> int:
        assert (style_id, speaker_uuid) == (
            self._expected_style_id,
            self._expected_speaker_uuid,
        )
        return 2

    @staticmethod
    def get_world_f0(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == 16000
        return np.array([110.0, 120.0], dtype=np.float64)

    @staticmethod
    def trim(wave: np.ndarray) -> np.ndarray:
        return wave

    @staticmethod
    def volume(wave: np.ndarray, scale: float) -> np.ndarray:
        return wave * scale

    @staticmethod
    def pitch_intonation(
        wave: np.ndarray,
        sampling_rate: int,
        pitch_scale: float,
        intonation_scale: float,
    ) -> np.ndarray:
        return wave + np.float32(pitch_scale + intonation_scale - 1.0)

    @staticmethod
    def sil(
        wave: np.ndarray, sampling_rate: int, pre: float, post: float
    ) -> np.ndarray:
        return np.concatenate(
            (
                np.zeros(int(sampling_rate * pre), dtype=np.float32),
                wave,
                np.zeros(int(sampling_rate * post), dtype=np.float32),
            )
        )

    @staticmethod
    def resample_output(
        wave: np.ndarray, sampling_rate: int, output_sampling_rate: int
    ) -> np.ndarray:
        return np.repeat(wave, output_sampling_rate // sampling_rate)


@pytest.fixture
def v2_fake_audio_manager_cls() -> type[V2FakeAudioManager]:
    return V2FakeAudioManager
