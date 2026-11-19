"""COEIROINK v2 HTTPルーターのエンドポイント契約・エラー応答・OpenAPIスキーマを検証する。"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    SynthesisError,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

import voicevox_engine
from voicevox_engine.coeiroink_v2.audio import encode_pcm_wav
from voicevox_engine.coeiroink_v2.catalog import CatalogNetworkError
from voicevox_engine.coeiroink_v2.metadata import MetadataAssetNotFoundError
from voicevox_engine.coeiroink_v2.models import (
    SpeakerMeta,
    SpeakerMetaPathVariant,
    SpeakerMetaStyle,
    SpeakerPolicy,
)
from voicevox_engine.coeiroink_v2.router import create_v2_router

SPEAKER_UUID = "00000000-0000-0000-0000-000000000001"
STYLE_ID = 7


class FakeMetadata:
    speaker_info_dir = Path("/tmp/fake-speaker-info")

    def __init__(self) -> None:
        self.speaker = SpeakerMeta(
            speakerName="Fake",
            speakerUuid=SPEAKER_UUID,
            styles=[
                SpeakerMetaStyle(
                    styleName="normal",
                    styleId=STYLE_ID,
                    base64Icon="aWNvbg==",
                )
            ],
            base64Portrait="cG9ydHJhaXQ=",
        )
        self.path_variant = SpeakerMetaPathVariant(
            speakerName="Fake",
            speakerUuid=SPEAKER_UUID,
            styles=[
                {
                    "styleName": "normal",
                    "styleId": STYLE_ID,
                    "pathIcon": "/tmp/icon.png",
                }
            ],
            pathPortrait="/tmp/portrait.png",
        )
        self.sample = encode_pcm_wav(np.ones(4, dtype=np.float32) * 0.1, 16000)

    @property
    def speaker_uuids(self) -> tuple[str, ...]:
        return (SPEAKER_UUID,)

    def list_speakers(self) -> list[SpeakerMeta]:
        return [self.speaker]

    def list_speakers_path_variant(self) -> list[SpeakerMetaPathVariant]:
        return [self.path_variant]

    def speaker_meta(self, speaker_uuid: str) -> SpeakerMeta:
        assert speaker_uuid == SPEAKER_UUID
        return self.speaker

    def speaker_path(self, speaker_uuid: str) -> Path:
        assert speaker_uuid == SPEAKER_UUID
        return self.speaker_info_dir / "fake"

    def get_style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        assert speaker_uuid == SPEAKER_UUID
        assert style_id == STYLE_ID
        return self.speaker.styles[0]

    def style_id_to_speaker_meta(self, style_id: int) -> dict[str, object]:
        assert style_id == STYLE_ID
        return {
            "speakerUuid": SPEAKER_UUID,
            "styleId": STYLE_ID,
            "speakerName": "Fake",
            "styleName": "normal",
        }

    def read_sample_voice(self, speaker_uuid: str, style_id: int, index: int) -> bytes:
        assert (speaker_uuid, style_id, index) == (SPEAKER_UUID, STYLE_ID, 0)
        return self.sample

    def speaker_policy(self, speaker_uuid: str) -> SpeakerPolicy:
        assert speaker_uuid == SPEAKER_UUID
        return SpeakerPolicy(policy="policy", license="license")


def _make_app(
    v2_fake_audio_manager_cls: type,
    *,
    voice_smoothing: bool = False,
    metadata: object | None = None,
    **callbacks: object,
) -> tuple[FastAPI, object, list[object]]:
    manager = v2_fake_audio_manager_cls(
        voice_smoothing=voice_smoothing,
        expected_speaker_uuid=SPEAKER_UUID,
        expected_style_id=STYLE_ID,
    )
    dictionary_calls: list[object] = []

    def set_dictionary(words: object) -> None:
        dictionary_calls.append(words)

    router_callbacks: dict[str, object] = {
        "dictionary_callback": set_dictionary,
        "download_info_callback": list,
        "downloadable_speakers_callback": list,
        "update_info_callback": list,
        **callbacks,
    }
    app = FastAPI()
    app.include_router(
        create_v2_router(
            manager,
            FakeMetadata() if metadata is None else metadata,
            **router_callbacks,
            default_trim_buffer={
                "startTrimBuffer": 0.0,
                "endTrimBuffer": 0.0,
                "pauseStartTrimBuffer": 0.0,
                "pauseEndTrimBuffer": 0.0,
            },
        )
    )
    return app, manager, dictionary_calls


def _making_payload() -> dict[str, object]:
    return {
        "speakerUuid": SPEAKER_UUID,
        "styleId": STYLE_ID,
        "text": "this text is deliberately ignored",
        "prosodyDetail": [[{"phoneme": "a", "hira": "あ", "accent": 0}]],
        "speedScale": 1.0,
    }


def _processing_wav_base64() -> str:
    return base64.standard_b64encode(
        encode_pcm_wav(np.ones(8, dtype=np.float32) * 0.1, 16000)
    ).decode("ascii")


def _processing_payload(wav_base64: str) -> dict[str, object]:
    return {
        "volumeScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
        "wavBase64": wav_base64,
    }


def _synthesis_payload() -> dict[str, object]:
    return {
        **_making_payload(),
        "volumeScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
    }


@pytest.mark.parametrize(
    ("path", "callback_name"),
    [
        ("/v1/download_info", "download_info_callback"),
        ("/v1/downloadable_speakers", "downloadable_speakers_callback"),
        ("/v1/update_info", "update_info_callback"),
    ],
)
def test_catalog_failure_returns_json_502(
    v2_fake_audio_manager_cls: type,
    path: str,
    callback_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable_catalog() -> None:
        raise CatalogNetworkError("catalog connection failed")

    app, _, _ = _make_app(
        v2_fake_audio_manager_cls, **{callback_name: unavailable_catalog}
    )

    response = TestClient(app).get(path)

    assert response.status_code == 502
    assert response.json() == {"detail": "catalog connection failed"}
    assert "CatalogNetworkError: catalog connection failed" in caplog.text


def test_catalog_programming_error_is_not_reported_as_transport_failure(
    v2_fake_audio_manager_cls: type,
) -> None:
    def broken_callback() -> None:
        raise ValueError("callback implementation failed")

    app, _, _ = _make_app(
        v2_fake_audio_manager_cls, download_info_callback=broken_callback
    )

    with pytest.raises(ValueError, match="callback implementation failed"):
        TestClient(app).get("/v1/download_info")


def test_status_and_metadata_endpoints(v2_fake_audio_manager_cls: type) -> None:
    app, _, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    assert client.get("/").json() == {"status": "start"}
    assert client.get("/v1/engine_info").json() == {
        "device": "cpu",
        "version": voicevox_engine.__version__,
    }
    assert client.get("/v1/speakers").json()[0]["speakerUuid"] == SPEAKER_UUID
    assert client.get("/v1/speakers_path_variant").status_code == 200
    assert (
        client.get("/v1/speaker_folder_path", params={"speakerUuid": SPEAKER_UUID})
        .json()["speakerFolderPath"]
        .endswith("/fake")
    )
    assert client.get("/v1/speaker_folder_path").json() == {"speakerFolderPath": "None"}
    assert (
        client.post(
            "/v1/style_id_to_speaker_meta", params={"styleId": STYLE_ID}
        ).json()["styleId"]
        == STYLE_ID
    )
    assert client.post("/v1/style_id_to_speaker_meta").json() == {
        "speakerUuid": "None",
        "styleId": 0,
        "speakerName": "None",
        "styleName": "None",
    }
    assert (
        client.get(
            "/v1/sample_voice",
            params={"speakerUuid": SPEAKER_UUID, "styleId": STYLE_ID},
        ).headers["content-type"]
        == "audio/wav"
    )
    assert client.get(
        "/v1/speaker_policy", params={"speakerUuid": SPEAKER_UUID}
    ).json() == {"policy": "policy", "license": "license"}


def test_prosody_query_and_f0_endpoints(v2_fake_audio_manager_cls: type) -> None:
    app, _, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    prosody = client.post("/v1/estimate_prosody_from_kana", json={"text": "ア'"})
    assert prosody.status_code == 200
    assert prosody.json() == {"detail": [[{"phoneme": "a", "hira": "あ", "accent": 1}]]}

    query = {
        "accentPhrases": [
            {
                "moras": [
                    {
                        "text": "あ",
                        "vowel": "a",
                        "vowelLength": 0.1,
                        "pitch": 0.0,
                    }
                ],
                "accent": 0,
                "isInterrogative": False,
            }
        ],
        "speedScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
        "outputStereo": False,
    }
    assert client.post("/v1/query2prosody", json=query).json()["plain"] == [
        "^",
        "a",
        "$",
    ]

    f0 = client.post(
        "/v1/estimate_f0",
        json={"wavBase64": _processing_wav_base64(), "moraDurations": []},
    )
    assert f0.status_code == 200
    assert f0.json()["f0"] == [110.0, 120.0]


def test_control_and_catalog_endpoints(v2_fake_audio_manager_cls: type) -> None:
    app, _, dictionary_calls = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    assert client.post("/v1/set_dictionary", json={"dictionaryWords": []}).json() == {}
    assert len(dictionary_calls) == 1
    assert (
        client.post(
            "/v1/set_default_processing_algorithm",
            json={"processingAlgorithm": "world"},
        ).json()
        is None
    )
    invalid_algorithm = client.post(
        "/v1/set_default_processing_algorithm",
        json={"processingAlgorithm": "unknown"},
    )
    assert invalid_algorithm.status_code == 422
    assert "processing_algorithm" in invalid_algorithm.json()["detail"]
    assert (
        client.post(
            "/v1/set_default_trim_buffer",
            json={
                "startTrimBuffer": 0.0,
                "endTrimBuffer": 0.0,
                "pauseStartTrimBuffer": 0.0,
                "pauseEndTrimBuffer": 0.0,
            },
        ).json()
        is None
    )
    assert client.get("/v1/download_info").json() == []
    assert client.get("/v1/downloadable_speakers").json() == []
    assert client.get("/v1/update_info").json() == []


def test_predict_and_predict_with_duration_endpoints(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    payload = _making_payload()
    predicted = client.post("/v1/predict", json=payload)
    assert predicted.status_code == 200
    assert predicted.headers["content-type"] == "audio/wav"
    assert manager.prediction_calls[-1]["text"] == ["^", "a", "$"]

    estimated = client.post(
        "/v1/predict", json={**payload, "text": "い", "prosodyDetail": []}
    )
    assert estimated.status_code == 200
    assert manager.prediction_calls[-1]["text"] == ["^", "i", "$"]

    with_duration = client.post("/v1/predict_with_duration", json=payload)
    assert with_duration.status_code == 200
    assert with_duration.headers["content-type"].startswith("application/json")
    assert with_duration.json()["moraDurations"]


def test_process_synthesis_and_redirect_endpoints(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    processed = client.post(
        "/v1/process",
        json={
            **_processing_payload(_processing_wav_base64()),
            "sampledIntervalValue": 0,
            "adjustedF0": [],
            "processingAlgorithm": "coeiroink",
        },
    )
    assert processed.status_code == 200
    assert processed.headers["content-type"] == "audio/wav"

    pitch_time = np.arange(4096, dtype=np.float32) / 16000.0
    pitch_wav = encode_pcm_wav(0.1 * np.sin(2.0 * np.pi * 220.0 * pitch_time), 16000)
    pitched = client.post(
        "/v1/process",
        json={
            **_processing_payload(base64.standard_b64encode(pitch_wav).decode("ascii")),
            "processingAlgorithm": "resampling",
            "pitchScale": 0.25,
            "adjustedF0": [],
        },
    )
    assert pitched.status_code == 200
    assert pitched.headers["content-type"] == "audio/wav"

    synthesis = client.post("/v1/synthesis", json=_synthesis_payload())
    assert synthesis.status_code == 200
    assert synthesis.headers["content-type"] == "audio/wav"
    assert manager.prediction_calls[-1]["kind"] == "predict"
    assert manager.smoothing_calls == []

    redirected = client.post("/v1/process_with_pitch", follow_redirects=False)
    assert redirected.status_code == 307
    assert redirected.headers["location"] == "/v1/process"
    assert redirected.content == b""


def test_v2_synthesis_voice_smoothing_gets_duration_and_runs_once(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls, voice_smoothing=True)

    response = TestClient(app).post("/v1/synthesis", json=_synthesis_payload())

    assert response.status_code == 200
    assert manager.prediction_calls == [
        {
            "kind": "predict_with_duration",
            "text": ["^", "a", "$"],
            "style_id": STYLE_ID,
            "speed_scale": 1.0,
            "speaker_uuid": SPEAKER_UUID,
        }
    ]
    assert len(manager.smoothing_calls) == 1
    call = manager.smoothing_calls[0]
    np.testing.assert_array_equal(
        call["wave"], np.linspace(-0.25, 0.25, 8, dtype=np.float32)
    )
    assert call["tokens"] == ["^", "a", "$"]
    assert call["duration_frames"] == [1, 2, 1]
    assert call["hop_length"] == 2


@pytest.mark.parametrize(
    ("text", "special_phoneme", "special_hira"),
    [
        ("これは。あれは。", "_", "、"),
        ("これは質問ですか？", "?", "？"),
    ],
)
def test_v2_estimated_prosody_detail_round_trips_special_phrases(
    v2_fake_audio_manager_cls: type,
    text: str,
    special_phoneme: str,
    special_hira: str,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)

    estimated = client.post("/v1/estimate_prosody", json={"text": text})
    assert estimated.status_code == 200
    estimated_data = estimated.json()
    assert [
        phrase
        for phrase in estimated_data["detail"]
        if len(phrase) == 1 and phrase[0]["phoneme"] == special_phoneme
    ] == [[{"phoneme": special_phoneme, "hira": special_hira, "accent": 0}]]

    predicted = client.post(
        "/v1/predict",
        json={
            "speakerUuid": SPEAKER_UUID,
            "styleId": STYLE_ID,
            "text": text,
            "prosodyDetail": estimated_data["detail"],
            "speedScale": 1.0,
        },
    )
    assert predicted.status_code == 200
    assert manager.prediction_calls[-1]["text"] == estimated_data["plain"]


def test_question_in_middle_can_be_predicted_with_or_without_prosody_detail(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)
    client = TestClient(app)
    text = "お元気ですか？今日は晴れです。"
    estimated = client.post("/v1/estimate_prosody", json={"text": text})
    assert estimated.status_code == 200

    for detail in (estimated.json()["detail"], []):
        response = client.post(
            "/v1/predict",
            json={**_making_payload(), "text": text, "prosodyDetail": detail},
        )
        assert response.status_code == 200

    tokens = manager.prediction_calls[0]["text"]
    assert tokens == manager.prediction_calls[1]["text"]
    assert "_" in tokens
    assert "?" not in tokens
    assert tokens[-1] == "$"


@pytest.mark.parametrize(
    "overrides",
    [
        {"outputSamplingRate": 0},
        {"wavBase64": "not-valid-base64!"},
        {
            "pauseLength": 0.3,
            "moraDurations": [
                {
                    "mora": mora,
                    "hira": "",
                    "phonemePitches": [],
                    "wavRange": {"start": start, "end": end},
                }
                for mora, start, end in (("a", 0, 2), ("pau", 2, 2), ("a", 2, 8))
            ],
        },
    ],
)
def test_v2_process_rejects_invalid_payloads_with_422(
    v2_fake_audio_manager_cls: type,
    overrides: dict[str, object],
) -> None:
    app, _, _ = _make_app(v2_fake_audio_manager_cls)
    payload = {**_processing_payload(_processing_wav_base64()), **overrides}

    assert TestClient(app).post("/v1/process", json=payload).status_code == 422


def test_v2_router_openapi_uses_aliases_and_audio_content_types(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, _, _ = _make_app(v2_fake_audio_manager_cls)
    schema = app.openapi()

    assert (
        "speakerUuid" in schema["components"]["schemas"]["WavMakingParam"]["properties"]
    )
    assert "styleId" in schema["components"]["schemas"]["WavMakingParam"]["properties"]
    assert (
        "audio/wav"
        in schema["paths"]["/v1/predict"]["post"]["responses"]["200"]["content"]
    )
    assert (
        "audio/wav"
        in schema["paths"]["/v1/process"]["post"]["responses"]["200"]["content"]
    )
    assert "307" in schema["paths"]["/v1/process_with_pitch"]["post"]["responses"]
    sample_parameters = schema["paths"]["/v1/sample_voice"]["get"]["parameters"]
    index_parameter = next(
        item for item in sample_parameters if item["name"] == "index"
    )
    assert "default" not in index_parameter["schema"]


def test_missing_sample_voice_uses_official_error_contract(
    tmp_path: Path,
    v2_fake_audio_manager_cls: type,
) -> None:
    class MissingSampleMetadata(FakeMetadata):
        def read_sample_voice(
            self, speaker_uuid: str, style_id: int, index: int
        ) -> bytes:
            raise MetadataAssetNotFoundError(tmp_path / "missing.wav", "voice sample")

    missing_app, _, _ = _make_app(
        v2_fake_audio_manager_cls, metadata=MissingSampleMetadata()
    )
    response = TestClient(missing_app).get(
        "/v1/sample_voice",
        params={"speakerUuid": SPEAKER_UUID, "styleId": STYLE_ID},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Sample voice file not found"}


def test_sample_voice_rejects_negative_index(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, _, _ = _make_app(v2_fake_audio_manager_cls)

    response = TestClient(app).get(
        "/v1/sample_voice",
        params={"speakerUuid": SPEAKER_UUID, "styleId": STYLE_ID, "index": -1},
    )

    assert response.status_code == 422


def test_unexpected_value_error_is_not_reported_as_request_validation(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)

    def fail_prediction(*_args: object, **_kwargs: object) -> None:
        raise ValueError("implementation defect")

    manager.predict = fail_prediction

    with pytest.raises(ValueError, match="implementation defect"):
        TestClient(app).post("/v1/predict", json=_making_payload())


def test_synthesis_error_logs_the_original_cause(
    v2_fake_audio_manager_cls: type,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)

    def fail_prediction(*_args: object, **_kwargs: object) -> None:
        try:
            raise RuntimeError("inference failure")
        except RuntimeError as error:
            raise SynthesisError("synthesis failed") from error

    manager.predict = fail_prediction
    response = TestClient(app).post("/v1/predict", json=_making_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "synthesis failed"}
    assert "RuntimeError: inference failure" in caplog.text
    assert len(caplog.records) == 1


def test_core_parameter_error_is_reported_as_request_validation(
    v2_fake_audio_manager_cls: type,
) -> None:
    app, manager, _ = _make_app(v2_fake_audio_manager_cls)

    def reject_parameter(*_args: object, **_kwargs: object) -> None:
        raise InvalidSynthesisParameterError("speed_scale must be positive")

    manager.predict = reject_parameter
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/predict",
        json=_making_payload(),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "speed_scale must be positive"}
