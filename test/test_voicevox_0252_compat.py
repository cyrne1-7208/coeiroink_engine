import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

import run as engine_run
from test.test_old_mycoeiroink import SPEAKER_UUID, STYLE_ID, create_test_client
from voicevox_engine.preset import Preset
from voicevox_engine.voicevox_compat import router as voicevox_compat_router

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


def _audio_query(client):
    response = client.post(
        "/voicevox/audio_query",
        params={
            "text": "テストです",
            "speaker": STYLE_ID,
            "enable_katakana_english": True,
        },
    )
    assert response.status_code == 200
    return response.json()


def _preset(**overrides):
    values = {
        "id": 1,
        "name": "テストプリセット",
        "speaker_uuid": SPEAKER_UUID,
        "style_id": STYLE_ID,
        "speedScale": 1,
        "pitchScale": 0,
        "intonationScale": 1,
        "volumeScale": 1,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "pauseLength": None,
        "pauseLengthScale": 1,
    }
    values.update(overrides)
    return Preset(**values)


def test_voicevox_routes_are_prefixed_without_moving_coeiroink_v1(tmp_path: Path):
    client, _ = create_test_client(tmp_path)
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


def test_combined_fastapi_documentation_remains_available(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    docs = client.get("/docs")
    openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert "swagger-ui" in docs.text.lower()
    assert openapi.status_code == 200
    assert "/voicevox/audio_query" in openapi.json()["paths"]
    assert "/v1/engine_info" in openapi.json()["paths"]


def test_audio_query_matches_voicevox_0252_defaults_and_legacy_body(tmp_path: Path):
    client, _ = create_test_client(tmp_path)
    query = _audio_query(client)

    assert query["pauseLength"] is None
    assert query["pauseLengthScale"] == 1
    assert query["outputSamplingRate"] == 44100

    query.pop("pauseLength")
    query.pop("pauseLengthScale")
    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": STYLE_ID},
        json=query,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")


def test_katakana_english_parameter_changes_only_unknown_english_reading(
    tmp_path: Path,
):
    client, _ = create_test_client(tmp_path)

    queries = {}
    for enabled in (False, True):
        response = client.post(
            "/voicevox/audio_query",
            params={
                "text": "synthesis",
                "speaker": STYLE_ID,
                "enable_katakana_english": enabled,
            },
        )
        assert response.status_code == 200
        queries[enabled] = response.json()

    assert queries[False]["kana"] != queries[True]["kana"]
    assert "シンセシス" in queries[True]["kana"].replace("'", "")

    manifest = client.get("/voicevox/engine_manifest").json()
    assert manifest["supported_features"]["apply_katakana_english"] is True


def test_audio_query_from_legacy_preset_adds_0252_pause_defaults(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        engine_run.PresetManager, "load_presets", lambda _self: [_preset()]
    )
    client, _ = create_test_client(tmp_path)

    response = client.post(
        "/voicevox/audio_query_from_preset",
        params={
            "text": "テストです",
            "preset_id": 1,
            "enable_katakana_english": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["pauseLength"] is None
    assert response.json()["pauseLengthScale"] == 1


def test_audio_query_from_preset_preserves_pause_controls(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        engine_run.PresetManager,
        "load_presets",
        lambda _self: [_preset(pauseLength=0.25)],
    )
    client, _ = create_test_client(tmp_path)

    response = client.post(
        "/voicevox/audio_query_from_preset",
        params={"text": "テストです", "preset_id": 1},
    )

    assert response.status_code == 200
    assert response.json()["pauseLength"] == 0.25
    assert response.json()["pauseLengthScale"] == 1


def test_multi_synthesis_accepts_modern_interrogative_flag(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    query = _audio_query(client)

    response = client.post(
        "/voicevox/multi_synthesis",
        params={
            "speaker": STYLE_ID,
            "enable_interrogative_upspeak": False,
        },
        json=[query],
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    audio_manager.synthesis.assert_called_once()

    empty = client.post(
        "/voicevox/multi_synthesis",
        params={"speaker": STYLE_ID},
        json=[],
    )
    assert empty.status_code == 422

    mismatched_query = dict(query)
    mismatched_query["outputSamplingRate"] = 24000
    mismatched = client.post(
        "/voicevox/multi_synthesis",
        params={"speaker": STYLE_ID},
        json=[query, mismatched_query],
    )
    assert mismatched.status_code == 422
    assert audio_manager.synthesis.call_count == 1


def test_multi_synthesis_removes_partial_zip_after_failure(tmp_path: Path, monkeypatch):
    client, audio_manager = create_test_client(tmp_path)
    query = _audio_query(client)
    temporary_dir = tmp_path / "multi-synthesis-temp"
    temporary_dir.mkdir()
    original_named_temporary_file = voicevox_compat_router.NamedTemporaryFile

    def temporary_file_in_test_dir(**kwargs):
        return original_named_temporary_file(dir=temporary_dir, **kwargs)

    monkeypatch.setattr(
        voicevox_compat_router, "NamedTemporaryFile", temporary_file_in_test_dir
    )
    audio_manager.synthesis.side_effect = RuntimeError("test synthesis failure")

    with pytest.raises(RuntimeError, match="test synthesis failure"):
        client.post(
            "/voicevox/multi_synthesis",
            params={"speaker": STYLE_ID},
            json=[query],
        )

    assert list(temporary_dir.iterdir()) == []


def test_disabled_cancellable_synthesis_returns_explicit_404(tmp_path: Path):
    client, _ = create_test_client(tmp_path)
    query = _audio_query(client)

    response = client.post(
        "/voicevox/cancellable_synthesis",
        params={"speaker": STYLE_ID},
        json=query,
    )

    assert response.status_code == 404
    assert "デフォルトで無効" in response.json()["detail"]


def test_enabled_cancellable_synthesis_uses_injected_engine(tmp_path: Path):
    wav_path = tmp_path / "cancellable.wav"
    wav_path.write_bytes(b"RIFF-cancellable-test")
    cancellable_engine = Mock()
    cancellable_engine._synthesis_impl.return_value = str(wav_path)
    client, _ = create_test_client(
        tmp_path,
        cancellable_engine=cancellable_engine,
    )
    query = _audio_query(client)

    response = client.post(
        "/voicevox/cancellable_synthesis",
        params={
            "speaker": STYLE_ID,
            "enable_interrogative_upspeak": False,
        },
        json=query,
    )

    assert response.status_code == 200
    assert response.content == b"RIFF-cancellable-test"
    assert (
        cancellable_engine._synthesis_impl.call_args.kwargs[
            "enable_interrogative_upspeak"
        ]
        is False
    )


def test_cancellable_disconnection_monitor_follows_app_lifespan(tmp_path: Path):
    class MonitoringCancellableEngine:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()

        async def catch_disconnection(self):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.set()

    cancellable_engine = MonitoringCancellableEngine()
    client, _ = create_test_client(
        tmp_path,
        cancellable_engine=cancellable_engine,
    )

    with client:
        assert cancellable_engine.started.wait(timeout=1)

    assert cancellable_engine.stopped.wait(timeout=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [("pauseLength", 0.25), ("pauseLengthScale", 1.5)],
)
def test_pause_controls_are_forwarded_to_core(tmp_path: Path, field: str, value: float):
    client, audio_manager = create_test_client(tmp_path)
    query = _audio_query(client)
    query[field] = value

    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": STYLE_ID},
        json=query,
    )

    assert response.status_code == 200
    call = audio_manager.synthesis.call_args.kwargs
    assert call["pause_length"] == query.get("pauseLength")
    assert call["pause_length_scale"] == query.get("pauseLengthScale", 1)


def test_missing_required_audio_query_field_returns_422(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    query = _audio_query(client)
    query.pop("outputSamplingRate")

    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": STYLE_ID},
        json=query,
    )

    assert response.status_code == 422
    audio_manager.synthesis.assert_not_called()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/voicevox/singers"),
        ("get", "/voicevox/singer_info"),
        ("post", "/voicevox/sing_frame_audio_query"),
        ("post", "/voicevox/frame_synthesis"),
        ("get", "/voicevox/installed_libraries"),
        ("get", "/voicevox/downloadable_libraries"),
        ("post", "/voicevox/install_library/example"),
    ],
)
def test_out_of_scope_apis_return_explicit_501(tmp_path: Path, method: str, path: str):
    client, _ = create_test_client(tmp_path)
    response = getattr(client, method)(path)

    assert response.status_code == 501
    assert response.json()["detail"]


@pytest.mark.parametrize("path", ["mora_data", "mora_length", "mora_pitch"])
def test_manifest_disabled_mora_editing_returns_501(tmp_path: Path, path: str):
    client, _ = create_test_client(tmp_path)
    query = _audio_query(client)

    response = client.post(
        f"/voicevox/{path}",
        params={"speaker": STYLE_ID},
        json=query["accent_phrases"],
    )

    assert response.status_code == 501
    assert "提供していません" in response.json()["detail"]


def test_manifest_disabled_morphing_returns_501(tmp_path: Path):
    client, audio_manager = create_test_client(tmp_path)
    query = _audio_query(client)
    audio_manager.synthesis.reset_mock()

    targets = client.post("/voicevox/morphable_targets", json=[STYLE_ID])
    synthesis = client.post(
        "/voicevox/synthesis_morphing",
        params={
            "base_speaker": STYLE_ID,
            "target_speaker": STYLE_ID,
            "morph_rate": 0.5,
        },
        json=query,
    )

    assert targets.status_code == 501
    assert synthesis.status_code == 501
    audio_manager.synthesis.assert_not_called()


def test_validate_kana_matches_voicevox_0252_contract(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    valid = client.post("/voicevox/validate_kana", params={"text": "ア'"})
    invalid = client.post("/voicevox/validate_kana", params={"text": "ア'ク'セント"})

    assert valid.status_code == 200
    assert valid.json() is True
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error_name"] == "ACCENT_TWICE"


def test_connect_waves_returns_http_error_for_invalid_input(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    response = client.post("/voicevox/connect_waves", json=[])

    assert response.status_code == 422
    assert "wav" in response.json()["detail"]


def test_modern_metadata_device_and_manifest_shapes(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    speakers = client.get("/voicevox/speakers").json()
    assert speakers[0]["styles"][0]["type"] == "talk"

    speaker_info = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": SPEAKER_UUID, "resource_format": "base64"},
    )
    assert speaker_info.status_code == 200
    unsupported_resource_format = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": SPEAKER_UUID, "resource_format": "url"},
    )
    assert unsupported_resource_format.status_code == 200
    url_info = unsupported_resource_format.json()
    assert client.get(url_info["portrait"]).content == b"portrait"
    assert client.get(url_info["style_infos"][0]["icon"]).content == b"icon"
    assert [
        client.get(url).content for url in url_info["style_infos"][0]["voice_samples"]
    ] == [
        b"sample 1",
        b"sample 2",
        b"sample 3",
    ]
    assert client.get("/voicevox/_resources/not-registered").status_code == 404

    devices = client.get("/voicevox/supported_devices")
    assert devices.json() == {"cpu": True, "cuda": False, "dml": False}

    manifest = client.get("/voicevox/engine_manifest")
    assert manifest.status_code == 200
    manifest_json = manifest.json()
    assert manifest_json["frame_rate"] == 93.75
    assert manifest_json["supported_vvlib_manifest_version"] is None
    assert manifest_json["supported_features"]["sing"] is False
    assert manifest_json["supported_features"]["manage_library"] is False
    assert manifest_json["supported_features"]["adjust_pause_length"] is True
    assert manifest_json["supported_features"]["interrogative_upspeak"] is False
    assert manifest_json["supported_features"]["return_resource_url"] is True


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
    ],
)
def test_disable_mutable_api_returns_403_before_mutation(
    tmp_path: Path, method: str, path: str
):
    client, _ = create_test_client(tmp_path, disable_mutable_api=True)

    response = client.request(method, path)

    assert response.status_code == 403
    assert "無効化" in response.json()["detail"]


def test_disable_mutable_api_does_not_disable_read_only_routes(tmp_path: Path):
    client, _ = create_test_client(tmp_path, disable_mutable_api=True)

    assert client.get("/voicevox/version").status_code == 200
    assert client.get("/v1/engine_info").status_code == 200


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("0", False), ("", False)],
)
def test_voicevox_boolean_environment_values(monkeypatch, value: str, expected: bool):
    monkeypatch.setenv("VV_TEST_BOOLEAN", value)

    assert engine_run.decide_boolean_from_env("VV_TEST_BOOLEAN") is expected


def test_invalid_voicevox_boolean_environment_warns_and_disables(monkeypatch):
    monkeypatch.setenv("VV_TEST_BOOLEAN", "true")

    with pytest.warns(UserWarning, match="VV_TEST_BOOLEAN=true"):
        assert engine_run.decide_boolean_from_env("VV_TEST_BOOLEAN") is False


def test_openapi_contains_voicevox_0252_talk_parameters(tmp_path: Path):
    client, _ = create_test_client(tmp_path)
    schema = client.app.openapi()

    audio_query_parameters = {
        parameter["name"]
        for parameter in schema["paths"]["/voicevox/audio_query"]["post"]["parameters"]
    }
    assert "enable_katakana_english" in audio_query_parameters

    model = schema["components"]["schemas"]["voicevox_engine__model__AudioQuery"]
    assert model["properties"]["pauseLengthScale"]["default"] == 1
    assert "pauseLength" in model["properties"]
    assert set(model["required"]) == {
        "accent_phrases",
        "speedScale",
        "pitchScale",
        "intonationScale",
        "volumeScale",
        "prePhonemeLength",
        "postPhonemeLength",
        "outputSamplingRate",
        "outputStereo",
    }

    accent_phrase = schema["components"]["schemas"][
        "voicevox_engine__model__AccentPhrase"
    ]
    mora = schema["components"]["schemas"]["voicevox_engine__model__Mora"]
    assert set(accent_phrase["required"]) == {"moras", "accent"}
    assert set(mora["required"]) == {
        "text",
        "vowel",
        "vowel_length",
        "pitch",
    }

    assert (
        schema["paths"]["/voicevox/audio_query"]["post"]["operationId"] == "audio_query"
    )
    operation_ids = [
        operation["operationId"]
        for methods in schema["paths"].values()
        for operation in methods.values()
        if "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_setting_post_uses_voicevox_0252_no_content_response(tmp_path: Path):
    client, _ = create_test_client(tmp_path)

    response = client.post(
        "/voicevox/setting",
        data={"cors_policy_mode": "localapps", "allow_origin": ""},
    )

    assert response.status_code == 204
    assert response.content == b""
