import base64
from pathlib import Path

import numpy as np
import pytest
from coeirocore.pyworld_compat import load_pyworld
from fastapi import FastAPI
from fastapi.testclient import TestClient

import voicevox_engine.coeiroink_v2.router as router_module
from voicevox_engine.coeiroink_v2.audio import (
    MAX_PAUSE_LENGTH_SECONDS,
    MAX_SAMPLING_RATE,
    encode_pcm_wav,
)
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


class FakeAudioManager:
    fs = 16000
    hop_length = 2

    def __init__(self):
        self.prediction_calls = []
        self.dictionary_calls = []

    def predict(self, text, style_id, speed_scale, speaker_uuid):
        self.prediction_calls.append(
            {
                "kind": "predict",
                "text": list(text),
                "style_id": style_id,
                "speed_scale": speed_scale,
                "speaker_uuid": speaker_uuid,
            }
        )
        return np.linspace(-0.25, 0.25, 8, dtype=np.float32)

    def predict_with_duration(self, text, style_id, speed_scale, speaker_uuid):
        self.prediction_calls.append(
            {
                "kind": "predict_with_duration",
                "text": list(text),
                "style_id": style_id,
                "speed_scale": speed_scale,
                "speaker_uuid": speaker_uuid,
            }
        )
        return {
            "wav": np.linspace(-0.25, 0.25, 8, dtype=np.float32),
            "duration_frames": [1, 2, 1],
        }

    @staticmethod
    def get_world(wave, sampling_rate):
        assert sampling_rate == 16000
        return np.array([110.0, 120.0], dtype=np.float64), None, None

    @staticmethod
    def trim(wave):
        return wave

    @staticmethod
    def volume(wave, scale):
        return wave * scale

    @staticmethod
    def pitch_intonation(wave, sampling_rate, pitch_scale, intonation_scale):
        return wave + np.float32(pitch_scale + intonation_scale - 1.0)

    @staticmethod
    def sil(wave, sampling_rate, pre, post):
        return np.concatenate(
            (
                np.zeros(int(sampling_rate * pre), dtype=np.float32),
                wave,
                np.zeros(int(sampling_rate * post), dtype=np.float32),
            )
        )

    @staticmethod
    def resampling(wave, sampling_rate, output_sampling_rate):
        return np.repeat(wave, output_sampling_rate // sampling_rate)


class RecordingFakeAudioManager(FakeAudioManager):
    def __init__(self):
        super().__init__()
        self.events = []


class FakeMetadata:
    speaker_info_dir = Path("/tmp/fake-speaker-info")

    def __init__(self):
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
    def speaker_uuids(self):
        return (SPEAKER_UUID,)

    def list_speakers(self):
        return [self.speaker]

    def list_speakers_path_variant(self):
        return [self.path_variant]

    def speaker_meta(self, speaker_uuid):
        assert speaker_uuid == SPEAKER_UUID
        return self.speaker

    def speaker_path(self, speaker_uuid):
        assert speaker_uuid == SPEAKER_UUID
        return self.speaker_info_dir / "fake"

    def get_style(self, speaker_uuid, style_id):
        assert speaker_uuid == SPEAKER_UUID
        assert style_id == STYLE_ID
        return self.speaker.styles[0]

    def style_id_to_speaker_meta(self, style_id):
        assert style_id == STYLE_ID
        return {
            "speakerUuid": SPEAKER_UUID,
            "styleId": STYLE_ID,
            "speakerName": "Fake",
            "styleName": "normal",
        }

    def read_sample_voice(self, speaker_uuid, style_id, index):
        assert (speaker_uuid, style_id, index) == (SPEAKER_UUID, STYLE_ID, 0)
        return self.sample

    def speaker_policy(self, speaker_uuid):
        assert speaker_uuid == SPEAKER_UUID
        return SpeakerPolicy(policy="policy", license="license")


def _app():
    manager = FakeAudioManager()
    dictionary_calls = []

    def set_dictionary(words):
        dictionary_calls.append(words)

    app = FastAPI()
    app.include_router(
        create_v2_router(
            manager,
            FakeMetadata(),
            dictionary_callback=set_dictionary,
            catalog={
                "download_info": [],
                "downloadable_speakers": [],
                "update_info": [],
            },
            default_trim_buffer={
                "startTrimBuffer": 0.0,
                "endTrimBuffer": 0.0,
                "pauseStartTrimBuffer": 0.0,
                "pauseEndTrimBuffer": 0.0,
            },
        )
    )
    return app, manager, dictionary_calls


def _detail():
    return [[{"phoneme": "a", "hira": "あ", "accent": 0}]]


def _making_payload():
    return {
        "speakerUuid": SPEAKER_UUID,
        "styleId": STYLE_ID,
        "text": "this text is deliberately ignored",
        "prosodyDetail": _detail(),
        "speedScale": 1.0,
    }


def _processing_payload(wav_base64):
    return {
        "volumeScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
        "wavBase64": wav_base64,
    }


def _processing_wav_base64():
    return base64.standard_b64encode(
        encode_pcm_wav(np.ones(8, dtype=np.float32) * 0.1, 16000)
    ).decode("ascii")


def _mora_duration(mora, start, end):
    return {
        "mora": mora,
        "hira": "",
        "phonemePitches": [],
        "wavRange": {"start": start, "end": end},
    }


def _internal_pause_durations():
    return [
        _mora_duration("a", 0, 2),
        _mora_duration("pau", 2, 4),
        _mora_duration("a", 4, 8),
    ]


def test_v2_router_covers_json_metadata_and_control_endpoints():
    app, _manager, dictionary_calls = _app()
    client = TestClient(app)

    assert client.get("/").json() == {"status": "start"}
    assert client.get("/v1/engine_info").json() == {
        "device": "cpu",
        "version": "0.1.2+coeiroink.1.7.3",
    }
    assert client.get("/v1/speakers").json()[0]["speakerUuid"] == SPEAKER_UUID
    assert client.get("/v1/speakers_path_variant").status_code == 200

    prosody = client.post(
        "/v1/estimate_prosody_from_kana",
        json={"text": "ア'"},
    )
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

    wav = encode_pcm_wav(np.ones(8, dtype=np.float32) * 0.1, 16000)
    wav_base64 = base64.standard_b64encode(wav).decode("ascii")
    f0 = client.post(
        "/v1/estimate_f0",
        json={"wavBase64": wav_base64, "moraDurations": []},
    )
    assert f0.status_code == 200
    assert f0.json()["f0"] == [110.0, 120.0]

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

    assert client.post("/v1/set_dictionary", json={"dictionaryWords": []}).json() == {}
    assert len(dictionary_calls) == 1
    assert (
        client.post(
            "/v1/set_default_processing_algorithm",
            json={"processingAlgorithm": "world"},
        ).json()
        is None
    )
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


def test_v2_router_prediction_process_and_redirect_contract():
    app, manager, _ = _app()
    client = TestClient(app)

    payload = _making_payload()
    predicted = client.post("/v1/predict", json=payload)
    assert predicted.status_code == 200
    assert predicted.headers["content-type"] == "audio/wav"
    assert manager.prediction_calls[-1]["text"] == ["^", "a", "$"]

    estimated_payload = {
        **payload,
        "text": "い",
        "prosodyDetail": [],
    }
    estimated = client.post("/v1/predict", json=estimated_payload)
    assert estimated.status_code == 200
    assert manager.prediction_calls[-1]["text"] == ["^", "i", "$"]

    with_duration = client.post("/v1/predict_with_duration", json=payload)
    assert with_duration.status_code == 200
    assert with_duration.headers["content-type"].startswith("application/json")
    assert with_duration.json()["moraDurations"]

    processed = client.post(
        "/v1/process",
        json={
            **_processing_payload(
                base64.standard_b64encode(
                    encode_pcm_wav(np.ones(8, dtype=np.float32) * 0.1, 16000)
                ).decode("ascii")
            ),
            "sampledIntervalValue": 0,
            "adjustedF0": [],
            "processingAlgorithm": "coeiroink",
        },
    )
    assert processed.status_code == 200
    assert processed.headers["content-type"] == "audio/wav"

    pitch_time = np.arange(4096, dtype=np.float32) / 16000.0
    pitch_wav = encode_pcm_wav(0.1 * np.sin(2.0 * np.pi * 220.0 * pitch_time), 16000)
    pitch_payload = _processing_payload(
        base64.standard_b64encode(pitch_wav).decode("ascii")
    )
    pitch_payload.update(
        {
            "processingAlgorithm": "resampling",
            "pitchScale": 0.25,
            "adjustedF0": [],
        }
    )
    pitched = client.post("/v1/process", json=pitch_payload)
    assert pitched.status_code == 200
    assert pitched.headers["content-type"] == "audio/wav"

    synthesis = client.post(
        "/v1/synthesis",
        json={
            **_making_payload(),
            "volumeScale": 1.0,
            "pitchScale": 0.0,
            "intonationScale": 1.0,
            "prePhonemeLength": 0.0,
            "postPhonemeLength": 0.0,
            "outputSamplingRate": 16000,
        },
    )
    assert synthesis.status_code == 200
    assert synthesis.headers["content-type"] == "audio/wav"

    redirected = client.post("/v1/process_with_pitch", follow_redirects=False)
    assert redirected.status_code == 307
    assert redirected.headers["location"] == "/v1/process"
    assert redirected.content == b""


@pytest.mark.parametrize(
    "text, special_phoneme, special_hira",
    [
        ("これは。あれは。", "_", "、"),
        ("これは質問ですか？", "?", "？"),
    ],
)
def test_v2_estimated_prosody_detail_round_trips_special_phrases(
    text, special_phoneme, special_hira
):
    app, manager, _ = _app()
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


def test_v2_process_rejects_an_oversized_pause_length_with_422():
    app, _, _ = _app()
    client = TestClient(app)
    payload = _processing_payload(_processing_wav_base64())
    payload.update(
        {
            "pauseLength": MAX_PAUSE_LENGTH_SECONDS + 0.001,
            "moraDurations": _internal_pause_durations(),
        }
    )

    response = client.post("/v1/process", json=payload)

    assert response.status_code == 422


def test_v2_process_rejects_a_zero_length_internal_pau_with_422():
    app, _, _ = _app()
    client = TestClient(app)
    payload = _processing_payload(_processing_wav_base64())
    payload.update(
        {
            "pauseLength": 0.3,
            "moraDurations": [
                _mora_duration("a", 0, 2),
                _mora_duration("pau", 2, 2),
                _mora_duration("a", 2, 8),
            ],
        }
    )

    response = client.post("/v1/process", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [
        ("outputSamplingRate", 0),
        ("outputSamplingRate", MAX_SAMPLING_RATE + 1),
        ("prePhonemeLength", -0.001),
        ("postPhonemeLength", -0.001),
        ("volumeScale", -0.001),
        ("intonationScale", -0.001),
    ],
)
def test_v2_process_rejects_invalid_sampling_and_processing_values_with_422(
    field, value
):
    app, _, _ = _app()
    client = TestClient(app)
    payload = _processing_payload(_processing_wav_base64())
    payload[field] = value

    response = client.post("/v1/process", json=payload)

    assert response.status_code == 422


def test_v2_process_applies_adjusted_f0_before_pause_replacement_with_fake(
    monkeypatch,
):
    manager = RecordingFakeAudioManager()
    wave = np.linspace(-0.25, 0.25, 8, dtype=np.float32)

    def fake_world(
        audio_manager,
        current,
        sampling_rate,
        pitch_scale,
        intonation_scale,
        adjusted_f0,
    ):
        assert audio_manager is manager
        assert sampling_rate == 16000
        assert pitch_scale == 0.0
        assert intonation_scale == 1.0
        np.testing.assert_array_equal(adjusted_f0, [110.0, 120.0])
        manager.events.append("adjusted_f0")
        return current + 1.0

    def fake_pause(current, *args, **kwargs):
        np.testing.assert_array_equal(current, wave + 1.0)
        manager.events.append("pause")
        return current

    monkeypatch.setattr(router_module, "_world_process", fake_world)
    monkeypatch.setattr(
        router_module.audio_helpers,
        "replace_pause_segments",
        fake_pause,
    )

    result, sampling_rate = router_module._process_wave(
        manager,
        wave,
        16000,
        volume_scale=1.0,
        pitch_scale=0.0,
        intonation_scale=1.0,
        pre_phoneme_length=0.0,
        post_phoneme_length=0.0,
        output_sampling_rate=16000,
        start_trim_buffer=0.0,
        end_trim_buffer=0.0,
        processing_algorithm="world",
        adjusted_f0=[110.0, 120.0],
        pause_length=0.3,
        pause_start_trim_buffer=0.0,
        pause_end_trim_buffer=0.0,
        mora_durations=_internal_pause_durations(),
    )

    assert sampling_rate == 16000
    np.testing.assert_array_equal(result, wave + 1.0)
    assert manager.events == ["adjusted_f0", "pause"]


def test_world_processing_preserves_unvoiced_f0_frames(monkeypatch):
    manager = FakeAudioManager()
    wave = np.zeros(32, dtype=np.float32)
    captured = {}

    def fake_world(wave, sampling_rate):
        return (
            np.array([0.0, 100.0, 200.0, 0.0]),
            None,
            None,
        )

    def fake_synthesize(f0, spectral_envelope, aperiodicity, sampling_rate):
        captured["f0"] = np.asarray(f0).copy()
        return np.zeros(32, dtype=np.float64)

    manager.get_world = fake_world
    monkeypatch.setattr(load_pyworld(), "synthesize", fake_synthesize)

    result = router_module._world_process(
        manager,
        wave,
        16000,
        pitch_scale=0.0,
        intonation_scale=0.5,
        adjusted_f0=[0.0, 100.0, 200.0, 0.0],
    )

    np.testing.assert_array_equal(captured["f0"], [0.0, 125.0, 175.0, 0.0])
    assert result.dtype == np.float32
    assert result.shape == wave.shape


def test_v2_router_openapi_uses_aliases_and_audio_content_types():
    app, _, _ = _app()
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


def test_missing_sample_voice_uses_official_error_contract(tmp_path):
    class MissingSampleMetadata(FakeMetadata):
        def read_sample_voice(self, speaker_uuid, style_id, index):
            raise MetadataAssetNotFoundError(tmp_path / "missing.wav", "voice sample")

    manager = FakeAudioManager()
    missing_app = FastAPI()
    missing_app.include_router(
        create_v2_router(manager, MissingSampleMetadata(), catalog={})
    )
    response = TestClient(missing_app).get(
        "/v1/sample_voice",
        params={"speakerUuid": SPEAKER_UUID, "styleId": STYLE_ID},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Sample voice file not found"}
