# run.pyの統合Engineアプリにおけるルート共存・環境変数・変更系API制御・話者配信とエラー変換のテスト

from __future__ import annotations

from base64 import b64decode
from typing import TYPE_CHECKING

import pytest
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    ModelLoadError,
    StyleNotFoundError,
)

import run as engine_run
import voicevox_engine

if TYPE_CHECKING:
    from conftest import EngineClientFactory

LEGACY_VOICEVOX_PATHS = {
    "/accent_phrases",
    "/add_preset",
    "/audio_query",
    "/audio_query_from_preset",
    "/cancellable_synthesis",
    "/connect_waves",
    "/core_versions",
    "/delete_preset",
    "/engine_manifest",
    "/import_user_dict",
    "/initialize_speaker",
    "/is_initialized_speaker",
    "/mora_data",
    "/mora_length",
    "/mora_pitch",
    "/morphable_targets",
    "/multi_synthesis",
    "/presets",
    "/setting",
    "/speaker_info",
    "/speakers",
    "/supported_devices",
    "/synthesis",
    "/synthesis_morphing",
    "/update_preset",
    "/user_dict",
    "/user_dict_word",
    "/user_dict_word/{word_uuid}",
    "/version",
}

SINGING_PATHS = {
    "/voicevox/singers",
    "/voicevox/singer_info",
    "/voicevox/sing_frame_audio_query",
    "/voicevox/sing_frame_f0",
    "/voicevox/sing_frame_volume",
    "/voicevox/frame_synthesis",
}


def test_voicevox_routes_are_prefixed_without_moving_coeiroink_v1(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()
    paths = set(client.app.openapi()["paths"])

    for legacy_path in LEGACY_VOICEVOX_PATHS:
        assert legacy_path not in paths
        assert f"/voicevox{legacy_path}" in paths

    assert "/voicevox/validate_kana" in paths
    assert "/" in paths
    assert "/v1/engine_info" in paths
    assert "/v1/download_info" in paths
    assert "/voicevox/download_infos" not in paths
    assert not any(path.startswith("/voicevox/v1/") for path in paths)
    assert paths.isdisjoint(SINGING_PATHS)

    assert client.get("/").json() == {"status": "start"}
    assert client.get("/v1/engine_info").status_code == 200
    assert client.get("/v1/download_info").status_code == 200
    assert client.get("/voicevox/download_infos").status_code == 404
    assert client.post("/audio_query").status_code == 404
    assert client.get("/speakers").status_code == 404


def test_combined_fastapi_documentation_remains_available(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()

    docs = client.get("/docs")
    openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert "swagger-ui" in docs.text.lower()
    assert openapi.status_code == 200
    assert "/voicevox/audio_query" in openapi.json()["paths"]
    assert "/v1/engine_info" in openapi.json()["paths"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/voicevox/add_preset"),
        ("post", "/voicevox/update_preset"),
        ("post", "/voicevox/delete_preset"),
        ("post", "/voicevox/user_dict_word"),
        ("put", "/voicevox/user_dict_word/dummy"),
        ("delete", "/voicevox/user_dict_word/dummy"),
        ("post", "/voicevox/import_user_dict"),
        ("post", "/voicevox/setting"),
        ("post", "/v1/set_dictionary"),
        ("post", "/v1/set_default_processing_algorithm"),
        ("post", "/v1/set_default_trim_buffer"),
    ],
)
def test_disable_mutable_api_returns_403_before_mutation(
    engine_client_factory: EngineClientFactory,
    method: str,
    path: str,
) -> None:
    client, _ = engine_client_factory.create_client(disable_mutable_api=True)

    response = client.request(method, path)

    assert response.status_code == 403
    assert "無効化" in response.json()["detail"]


def test_disable_mutable_api_does_not_disable_read_only_routes(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client(disable_mutable_api=True)

    assert client.get("/voicevox/version").status_code == 200
    assert client.get("/v1/engine_info").status_code == 200


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("0", False), ("", False)],
)
def test_voicevox_boolean_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VV_TEST_BOOLEAN", value)

    assert engine_run.decide_boolean_from_env("VV_TEST_BOOLEAN") is expected


def test_invalid_voicevox_boolean_environment_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VV_TEST_BOOLEAN", "true")

    with pytest.warns(UserWarning, match="VV_TEST_BOOLEAN=true"):
        assert engine_run.decide_boolean_from_env("VV_TEST_BOOLEAN") is False


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/voicevox/initialize_speaker"),
        ("get", "/voicevox/is_initialized_speaker"),
    ],
)
def test_ambiguous_style_initialization_returns_422(
    engine_client_factory: EngineClientFactory,
    method: str,
    path: str,
) -> None:
    client, audio_manager = engine_client_factory.create_client(duplicate_style=True)

    response = client.request(
        method, path, params={"speaker": engine_client_factory.style_id}
    )

    assert response.status_code == 422
    audio_manager.initialize_speaker.assert_not_called()
    audio_manager.is_speaker_initialized.assert_not_called()


def test_metadata_loading_ignores_macos_zip_sidecars(
    engine_client_factory: EngineClientFactory,
) -> None:
    (engine_client_factory.tmp_path / "speaker_info" / "__MACOSX").mkdir(parents=True)
    client, _ = engine_client_factory.create_client()

    for path, uuid_key in (
        ("/voicevox/speakers", "speaker_uuid"),
        ("/v1/speakers", "speakerUuid"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert [speaker[uuid_key] for speaker in response.json()] == [
            engine_client_factory.speaker_uuid
        ]


@pytest.mark.parametrize("version", [1, None])
def test_metadata_version_matches_core_normalization(
    engine_client_factory: EngineClientFactory,
    version: object,
) -> None:
    client, _ = engine_client_factory.create_client(metadata_version=version)

    for path in ("/v1/speakers", "/voicevox/speakers"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()[0]["version"] == str(version)


def test_speaker_info_enumerates_existing_voice_samples(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    sample_dir = (
        engine_client_factory.tmp_path
        / "speaker_info"
        / engine_client_factory.speaker_uuid
        / "voice_samples"
    )
    (sample_dir / f"{style_id}_002.wav").unlink()
    (sample_dir / f"{style_id}_004.wav").write_bytes(b"sample 4")

    response = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": engine_client_factory.speaker_uuid},
    )

    assert response.status_code == 200
    samples = response.json()["style_infos"][0]["voice_samples"]
    assert [b64decode(value) for value in samples] == [
        b"sample 1",
        b"sample 3",
        b"sample 4",
    ]


def test_speaker_info_supports_non_uuid_folder_name(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client(folder_name="speaker_ver1.0")

    response = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": engine_client_factory.speaker_uuid},
    )

    assert response.status_code == 200
    assert response.json()["style_infos"][0]["id"] == engine_client_factory.style_id


def test_unknown_legacy_style_is_rejected_by_audio_query(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()

    response = client.post(
        "/voicevox/audio_query",
        params={"text": "テストです", "speaker": 999999999},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "MYCOEIROINK style is not installed: 999999999"
    }


def test_v1_unknown_style_is_rejected_as_not_found(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    audio_manager.predict.side_effect = StyleNotFoundError(
        "MYCOEIROINK style is not installed for speakerUuid unknown: 999999999"
    )

    response = client.post(
        "/v1/predict",
        json={
            "speakerUuid": engine_client_factory.speaker_uuid,
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


def test_v2_routes_use_the_public_core_audio_manager(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client(
        folder_name="speaker_ver1.0",
        device="directml",
    )
    expected_version = voicevox_engine.__version__

    assert client.get("/").json() == {"status": "start"}
    assert client.get("/openapi.json").json()["info"]["version"] == expected_version
    assert client.get("/voicevox/version").json() == expected_version
    assert client.get("/voicevox/engine_manifest").json()["version"] == expected_version
    assert client.get("/v1/engine_info").json() == {
        "device": "directml",
        "version": expected_version,
    }
    assert (
        client.get("/v1/speakers").json()[0]["speakerUuid"]
        == engine_client_factory.speaker_uuid
    )

    response = client.post(
        "/v1/predict",
        json={
            "speakerUuid": engine_client_factory.speaker_uuid,
            "styleId": engine_client_factory.style_id,
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
        style_id=engine_client_factory.style_id,
        speaker_uuid=engine_client_factory.speaker_uuid,
        speed_scale=1.0,
    )


def test_corrupt_model_returns_explicit_error(
    engine_client_factory: EngineClientFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id

    def fail_loading(*_args: object, **_kwargs: object) -> None:
        try:
            raise RuntimeError("invalid model weights")
        except RuntimeError as error:
            raise ModelLoadError(
                f"Failed to load MYCOEIROINK style {style_id}"
            ) from error

    audio_manager.synthesis.side_effect = fail_loading
    query = engine_client_factory.post_audio_query(client)

    response = client.post(
        "/voicevox/synthesis", params={"speaker": style_id}, json=query
    )

    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to load MYCOEIROINK style {style_id}"}
    assert "RuntimeError: invalid model weights" in caplog.text
    assert len(caplog.records) == 1


def test_invalid_synthesis_parameter_returns_422(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    audio_manager.synthesis.side_effect = InvalidSynthesisParameterError(
        "speed_scale must be a positive finite number"
    )
    query = engine_client_factory.post_audio_query(client)
    query["speedScale"] = 0

    response = client.post(
        "/voicevox/synthesis", params={"speaker": style_id}, json=query
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "speed_scale must be a positive finite number"}
