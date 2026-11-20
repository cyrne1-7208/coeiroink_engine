# voicevox_compatのVOICEVOX互換APIにおけるクエリ・プリセット・未提供機能の501・OpenAPI契約のテスト

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import run as engine_run
from voicevox_engine.engine_manifest import EngineManifestLoader
from voicevox_engine.preset import Preset
from voicevox_engine.voicevox_compat import router as voicevox_compat_router
from voicevox_engine.voicevox_compat.resources import (
    ResourceManager,
    ResourceNotFoundError,
)

if TYPE_CHECKING:
    from conftest import EngineClientFactory


def _preset(speaker_uuid: str, style_id: int, **overrides: object) -> Preset:
    values: dict[str, object] = {
        "id": 1,
        "name": "テストプリセット",
        "speaker_uuid": speaker_uuid,
        "style_id": style_id,
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


def test_manifest_cache_is_not_mutable_from_callers() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    loader = EngineManifestLoader(root_dir / "engine_manifest.json", root_dir)

    manifest = loader.load_manifest()
    manifest.supported_features.sing = True

    assert loader.load_manifest().supported_features.sing is False


def test_audio_query_matches_voicevox_0252_defaults_and_legacy_body(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)

    assert query["pauseLength"] is None
    assert query["pauseLengthScale"] == 1
    assert query["outputSamplingRate"] == 44100

    query.pop("pauseLength")
    query.pop("pauseLengthScale")
    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id},
        json=query,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    audio_manager.synthesis.assert_called_once()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outputSamplingRate", 384_001),
        ("prePhonemeLength", 60.001),
        ("postPhonemeLength", 60.001),
        ("pauseLength", 60.001),
    ],
)
def test_voicevox_audio_query_rejects_allocation_heavy_numeric_values(
    engine_client_factory: EngineClientFactory,
    field: str,
    value: float,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    audio_manager.synthesis.reset_mock()
    query[field] = value

    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id},
        json=query,
    )

    assert response.status_code == 422
    audio_manager.synthesis.assert_not_called()


def test_katakana_english_parameter_changes_only_unknown_english_reading(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id

    queries: dict[bool, dict[str, object]] = {}
    for enabled in (False, True):
        response = client.post(
            "/voicevox/audio_query",
            params={
                "text": "synthesis",
                "speaker": style_id,
                "enable_katakana_english": enabled,
            },
        )
        assert response.status_code == 200
        queries[enabled] = response.json()

    assert queries[False]["kana"] != queries[True]["kana"]
    assert "シンセシス" in str(queries[True]["kana"]).replace("'", "")


@pytest.mark.parametrize("pause_length", [None, 0.25])
def test_audio_query_from_preset_preserves_0252_pause_controls(
    engine_client_factory: EngineClientFactory,
    monkeypatch: pytest.MonkeyPatch,
    pause_length: float | None,
) -> None:
    preset = _preset(
        engine_client_factory.speaker_uuid,
        engine_client_factory.style_id,
        pauseLength=pause_length,
    )
    monkeypatch.setattr(
        engine_run.PresetManager, "load_presets", lambda _self: [preset]
    )
    client, _ = engine_client_factory.create_client()

    response = client.post(
        "/voicevox/audio_query_from_preset",
        params={
            "text": "テストです",
            "preset_id": 1,
            "enable_katakana_english": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["pauseLength"] == pause_length
    assert response.json()["pauseLengthScale"] == 1


def test_audio_query_from_preset_rejects_duplicate_style_id(
    engine_client_factory: EngineClientFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _preset(engine_client_factory.speaker_uuid, engine_client_factory.style_id)
    monkeypatch.setattr(
        engine_run.PresetManager, "load_presets", lambda _self: [preset]
    )
    client, _ = engine_client_factory.create_client(duplicate_style=True)

    response = client.post(
        "/voicevox/audio_query_from_preset",
        params={"text": "テストです", "preset_id": 1},
    )

    assert response.status_code == 422
    assert "ambiguous" in response.json()["detail"]


def test_multi_synthesis_accepts_modern_interrogative_flag(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)

    response = client.post(
        "/voicevox/multi_synthesis",
        params={
            "speaker": style_id,
            "enable_interrogative_upspeak": False,
        },
        json=[query],
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    audio_manager.synthesis.assert_called_once()

    empty = client.post(
        "/voicevox/multi_synthesis",
        params={"speaker": style_id},
        json=[],
    )
    assert empty.status_code == 422

    mismatched_query = dict(query)
    mismatched_query["outputSamplingRate"] = 24000
    mismatched = client.post(
        "/voicevox/multi_synthesis",
        params={"speaker": style_id},
        json=[query, mismatched_query],
    )
    assert mismatched.status_code == 422
    assert audio_manager.synthesis.call_count == 1


def test_multi_synthesis_removes_partial_zip_after_failure(
    engine_client_factory: EngineClientFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    temporary_dir = engine_client_factory.tmp_path / "multi-synthesis-temp"
    temporary_dir.mkdir()
    monkeypatch.setattr(
        voicevox_compat_router,
        "NamedTemporaryFile",
        partial(voicevox_compat_router.NamedTemporaryFile, dir=temporary_dir),
    )
    audio_manager.synthesis.side_effect = RuntimeError("test synthesis failure")

    with pytest.raises(RuntimeError, match="test synthesis failure"):
        client.post(
            "/voicevox/multi_synthesis",
            params={"speaker": style_id},
            json=[query],
        )

    assert list(temporary_dir.iterdir()) == []


def test_wave_response_removes_temporary_file_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_dir = tmp_path / "wave-response-temp"
    temporary_dir.mkdir()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        voicevox_compat_router,
        "NamedTemporaryFile",
        partial(voicevox_compat_router.NamedTemporaryFile, dir=temporary_dir),
    )
    monkeypatch.setattr(voicevox_compat_router.soundfile, "write", fail_write)

    with pytest.raises(RuntimeError, match="write failed"):
        voicevox_compat_router._wave_file_response(
            np.zeros(8, dtype=np.float32), 16_000
        )

    assert list(temporary_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("pauseLength", 0.25), ("pauseLengthScale", 1.5)],
)
def test_pause_controls_are_forwarded_to_core(
    engine_client_factory: EngineClientFactory,
    field: str,
    value: float,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    query[field] = value

    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id},
        json=query,
    )

    assert response.status_code == 200
    call = audio_manager.synthesis.call_args.kwargs
    assert call["pause_length"] == query.get("pauseLength")
    assert call["pause_length_scale"] == query.get("pauseLengthScale", 1)


def test_missing_required_audio_query_field_returns_422(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    query.pop("outputSamplingRate")

    response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id},
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
def test_out_of_scope_apis_return_explicit_501(
    engine_client_factory: EngineClientFactory,
    method: str,
    path: str,
) -> None:
    client, _ = engine_client_factory.create_client()
    response = getattr(client, method)(path)

    assert response.status_code == 501
    assert response.json()["detail"]


@pytest.mark.parametrize("path", ["mora_data", "mora_length", "mora_pitch"])
def test_manifest_disabled_mora_editing_returns_501(
    engine_client_factory: EngineClientFactory,
    path: str,
) -> None:
    client, _ = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)

    response = client.post(
        f"/voicevox/{path}",
        params={"speaker": style_id},
        json=query["accent_phrases"],
    )

    assert response.status_code == 501
    assert "提供していません" in response.json()["detail"]


def test_manifest_disabled_morphing_returns_501(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    audio_manager.synthesis.reset_mock()

    targets = client.post("/voicevox/morphable_targets", json=[style_id])
    synthesis = client.post(
        "/voicevox/synthesis_morphing",
        params={
            "base_speaker": style_id,
            "target_speaker": style_id,
            "morph_rate": 0.5,
        },
        json=query,
    )

    assert targets.status_code == 501
    assert synthesis.status_code == 501
    audio_manager.synthesis.assert_not_called()


def test_manifest_disabled_interrogative_upspeak_is_not_enabled_implicitly(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, audio_manager = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)
    audio_manager.synthesis.reset_mock()

    default_response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id},
        json=query,
    )
    explicit_response = client.post(
        "/voicevox/synthesis",
        params={"speaker": style_id, "enable_interrogative_upspeak": True},
        json=query,
    )

    assert default_response.status_code == 200
    assert explicit_response.status_code == 501
    assert audio_manager.synthesis.call_count == 1


def test_validate_kana_matches_voicevox_0252_contract(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()

    valid = client.post("/voicevox/validate_kana", params={"text": "ア'"})
    invalid = client.post("/voicevox/validate_kana", params={"text": "ア'ク'セント"})
    empty = client.post("/voicevox/validate_kana", params={"text": ""})

    assert valid.status_code == 200
    assert valid.json() is True
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error_name"] == "ACCENT_TWICE"
    assert empty.status_code == 400
    assert empty.json()["detail"]["error_args"] == {"position": "1"}


def test_connect_waves_returns_http_error_for_invalid_input(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()

    response = client.post("/voicevox/connect_waves", json=[])

    assert response.status_code == 422
    assert "wav" in response.json()["detail"]


def test_modern_metadata_device_and_manifest_shapes(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()
    speaker_uuid = engine_client_factory.speaker_uuid
    style_id = engine_client_factory.style_id

    assert client.get("/voicevox/speakers").json() == [
        {
            "name": "テスト話者",
            "speaker_uuid": speaker_uuid,
            "styles": [{"name": "テスト", "id": style_id, "type": "talk"}],
            "version": "0.0.1",
            "supported_features": {"permitted_synthesis_morphing": "ALL"},
        }
    ]

    speaker_info = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": speaker_uuid, "resource_format": "base64"},
    )
    assert speaker_info.status_code == 200
    base64_info = speaker_info.json()
    assert base64_info["policy"] == "test policy"
    assert base64_info["style_infos"][0]["id"] == style_id
    assert len(base64_info["style_infos"][0]["voice_samples"]) == 3
    url_response = client.get(
        "/voicevox/speaker_info",
        params={"speaker_uuid": speaker_uuid, "resource_format": "url"},
    )
    assert url_response.status_code == 200
    url_info = url_response.json()
    assert client.get(url_info["portrait"]).content == b"portrait"
    assert client.get(url_info["style_infos"][0]["icon"]).content == b"icon"
    assert [
        client.get(url).content for url in url_info["style_infos"][0]["voice_samples"]
    ] == [b"sample 1", b"sample 2", b"sample 3"]
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
    assert manifest_json["supported_features"]["apply_katakana_english"] is True
    assert manifest_json["supported_features"]["interrogative_upspeak"] is False
    assert manifest_json["supported_features"]["return_resource_url"] is True


def test_openapi_contains_voicevox_0252_talk_parameters(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()
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


def test_resource_manager_restricts_registration_to_files_under_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "speaker_info"
    nested = root / "speaker"
    nested.mkdir(parents=True)
    allowed = nested / "icon.png"
    allowed.write_bytes(b"icon")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    manager = ResourceManager(root)
    resource_id = manager.register(allowed)
    assert manager.path(resource_id) == allowed.resolve()

    for rejected in (outside, root / ".." / "secret.txt", nested, root / "missing.png"):
        with pytest.raises(ResourceNotFoundError):
            manager.register(rejected)
    with pytest.raises(ResourceNotFoundError):
        manager.path("unregistered")


def test_setting_post_uses_voicevox_0252_no_content_response(
    engine_client_factory: EngineClientFactory,
) -> None:
    client, _ = engine_client_factory.create_client()

    response = client.post(
        "/voicevox/setting",
        data={"cors_policy_mode": "localapps", "allow_origin": ""},
    )

    assert response.status_code == 204
    assert response.content == b""
