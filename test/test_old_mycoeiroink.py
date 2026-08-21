import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    ModelLoadError,
    StyleNotFoundError,
)
from fastapi.testclient import TestClient

from run import generate_app
from voicevox_engine.dev.synthesis_engine import MockSynthesisEngine
from voicevox_engine.setting import SettingLoader

# 実モデルに依存せず、旧MYCOEIROINKのメタデータ形式だけを検証する。
SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"
STYLE_ID = 1234567890


def create_old_mycoeiroink_fixture(
    speaker_info_dir: Path, folder_name: str = SPEAKER_UUID
) -> Path:
    speaker_dir = speaker_info_dir / folder_name
    (speaker_dir / "icons").mkdir(parents=True)
    (speaker_dir / "voice_samples").mkdir()
    (speaker_dir / "metas.json").write_text(
        json.dumps(
            {
                "speakerName": "テスト話者",
                "speakerUuid": SPEAKER_UUID,
                "styles": [{"styleName": "テスト", "styleId": STYLE_ID}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (speaker_dir / "policy.md").write_text("test policy", encoding="utf-8")
    (speaker_dir / "portrait.png").write_bytes(b"portrait")
    (speaker_dir / "icons" / f"{STYLE_ID}.png").write_bytes(b"icon")
    for index in range(1, 4):
        (speaker_dir / "voice_samples" / f"{STYLE_ID}_{index:03}.wav").write_bytes(
            f"sample {index}".encode()
        )
    return speaker_dir


def create_test_client(
    tmp_path: Path,
    folder_name: str = SPEAKER_UUID,
    cancellable_engine=None,
    device: str = "cpu",
    duplicate_style: bool = False,
):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir, folder_name=folder_name)
    if duplicate_style:
        duplicate_dir = create_old_mycoeiroink_fixture(
            speaker_info_dir, folder_name="duplicate-speaker"
        )
        duplicate_meta = json.loads((duplicate_dir / "metas.json").read_text())
        duplicate_meta["speakerName"] = "重複話者"
        duplicate_meta["speakerUuid"] = "00000000-0000-0000-0000-000000000002"
        (duplicate_dir / "metas.json").write_text(
            json.dumps(duplicate_meta, ensure_ascii=False), encoding="utf-8"
        )

    core_metas = json.dumps(
        [
            {
                "name": "テスト話者",
                "speaker_uuid": SPEAKER_UUID,
                "styles": [{"name": "テスト", "id": STYLE_ID}],
                "version": "0.0.1",
            }
        ],
        ensure_ascii=False,
    )
    audio_manager = Mock()
    audio_manager.synthesis.return_value = np.linspace(
        -0.1, 0.1, 4410, dtype=np.float32
    )
    audio_manager.fs = 44100
    audio_manager.predict.return_value = np.linspace(-0.1, 0.1, 4410, dtype=np.float32)
    engine = MockSynthesisEngine(
        speakers=core_metas,
        supported_devices=json.dumps({"cpu": True, "cuda": False}),
        audio_manager=audio_manager,
    )
    app = generate_app(
        synthesis_engines={"0.0.0": engine},
        latest_core_version="0.0.0",
        setting_loader=SettingLoader(tmp_path / "setting.yml"),
        root_dir=Path(__file__).parents[1],
        speaker_info_dir=speaker_info_dir,
        cancellable_engine=cancellable_engine,
        device=device,
    )
    return TestClient(app), audio_manager


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/voicevox/initialize_speaker"),
        ("get", "/voicevox/is_initialized_speaker"),
    ],
)
def test_ambiguous_style_initialization_returns_422(
    tmp_path: Path, method: str, path: str
):
    client, audio_manager = create_test_client(tmp_path, duplicate_style=True)

    response = client.request(method, path, params={"speaker": STYLE_ID})

    assert response.status_code == 422
    audio_manager.initialize_speaker.assert_not_called()
    audio_manager.is_speaker_initialized.assert_not_called()


def test_old_mycoeiroink_metadata_endpoints(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    speakers_response = client.get("/voicevox/speakers")
    assert speakers_response.status_code == 200
    assert speakers_response.json() == [
        {
            "name": "テスト話者",
            "speaker_uuid": SPEAKER_UUID,
            "styles": [{"name": "テスト", "id": STYLE_ID, "type": "talk"}],
            "version": "0.0.1",
            "supported_features": {"permitted_synthesis_morphing": "ALL"},
        }
    ]

    info_response = client.get(
        "/voicevox/speaker_info", params={"speaker_uuid": SPEAKER_UUID}
    )
    assert info_response.status_code == 200
    info = info_response.json()
    assert info["policy"] == "test policy"
    assert info["style_infos"][0]["id"] == STYLE_ID
    assert len(info["style_infos"][0]["voice_samples"]) == 3


def test_speaker_info_supports_non_uuid_folder_name(tmp_path: Path):
    client, _ = create_test_client(tmp_path, folder_name="speaker_ver1.0")

    response = client.get(
        "/voicevox/speaker_info", params={"speaker_uuid": SPEAKER_UUID}
    )

    assert response.status_code == 200
    assert response.json()["style_infos"][0]["id"] == STYLE_ID


def test_old_mycoeiroink_audio_query_and_synthesis(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)

    query_response = client.post(
        "/voicevox/audio_query",
        params={"text": "テストです", "speaker": STYLE_ID},
    )
    assert query_response.status_code == 200

    synthesis_response = client.post(
        "/voicevox/synthesis",
        params={"speaker": STYLE_ID},
        json=query_response.json(),
    )
    assert synthesis_response.status_code == 200
    assert synthesis_response.headers["content-type"].startswith("audio/wav")
    assert synthesis_response.content.startswith(b"RIFF")
    audio_manager.synthesis.assert_called_once()


def test_unknown_legacy_style_is_rejected_by_audio_query(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    response = client.post(
        "/voicevox/audio_query",
        params={"text": "テストです", "speaker": 999999999},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "MYCOEIROINK style is not installed: 999999999"
    }


def test_v1_unknown_style_is_rejected_as_not_found(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    audio_manager.predict.side_effect = StyleNotFoundError(
        "MYCOEIROINK style is not installed for speakerUuid unknown: 999999999"
    )

    response = client.post(
        "/v1/predict",
        json={
            "speakerUuid": SPEAKER_UUID,
            "styleId": 999999999,
            "text": "テストです",
            "speedScale": 1.0,
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "MYCOEIROINK style is not installed for speakerUuid unknown: 999999999"
        )
    }


def test_v2_routes_use_the_public_core_audio_manager(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path, folder_name="speaker_ver1.0")

    assert client.get("/").json() == {"status": "start"}
    assert client.get("/openapi.json").json()["info"]["version"] == "0.1.0"
    assert client.get("/voicevox/version").json() == "0.1.0"
    assert client.get("/voicevox/engine_manifest").json()["version"] == "0.1.0"
    assert client.get("/v1/engine_info").json() == {
        "device": "cpu",
        "version": "0.1.0",
    }
    assert client.get("/v1/speakers").json()[0]["speakerUuid"] == SPEAKER_UUID

    response = client.post(
        "/v1/predict",
        json={
            "speakerUuid": SPEAKER_UUID,
            "styleId": STYLE_ID,
            "text": "ignored when prosodyDetail is supplied",
            "prosodyDetail": [[{"phoneme": "a", "hira": "あ", "accent": 0}]],
            "speedScale": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    audio_manager.predict.assert_called_once_with(
        text=["^", "a", "$"],
        style_id=STYLE_ID,
        speaker_uuid=SPEAKER_UUID,
        speed_scale=1.0,
    )


def test_v2_engine_info_uses_selected_device(tmp_path: Path):
    client, _ = create_test_client(tmp_path, device="directml")

    assert client.get("/v1/engine_info").json()["device"] == "directml"


def test_corrupt_old_mycoeiroink_returns_explicit_error(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    audio_manager.synthesis.side_effect = ModelLoadError(
        f"Failed to load MYCOEIROINK style {STYLE_ID}"
    )
    query = client.post(
        "/voicevox/audio_query",
        params={"text": "テストです", "speaker": STYLE_ID},
    ).json()

    response = client.post(
        "/voicevox/synthesis", params={"speaker": STYLE_ID}, json=query
    )

    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to load MYCOEIROINK style {STYLE_ID}"}


def test_invalid_synthesis_parameter_returns_422(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    audio_manager.synthesis.side_effect = InvalidSynthesisParameterError(
        "speed_scale must be a positive finite number"
    )
    query = client.post(
        "/voicevox/audio_query",
        params={"text": "テストです", "speaker": STYLE_ID},
    ).json()
    query["speedScale"] = 0

    response = client.post(
        "/voicevox/synthesis", params={"speaker": STYLE_ID}, json=query
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "speed_scale must be a positive finite number"}
