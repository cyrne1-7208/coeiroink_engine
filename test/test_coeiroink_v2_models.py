from typing import Dict, Type

import pytest
from pydantic import BaseModel, ValidationError

from voicevox_engine.coeiroink_v2.models import (
    AccentPhrase,
    AlgorithmSettings,
    AudioQuery,
    DictionaryWord,
    DictionaryWords,
    DownloadableModel,
    DownloadableSpeaker,
    DownloadableStyle,
    EngineInfo,
    HTTPValidationError,
    Mora,
    MoraDuration,
    PhonemeDuration,
    Phrase,
    Prosody,
    ProsodyMakingParam,
    Speaker,
    SpeakerFolderPath,
    SpeakerInfo,
    SpeakerMeta,
    SpeakerMetaForTextBox,
    SpeakerMetaPathVariant,
    SpeakerMetaStyle,
    SpeakerPolicy,
    Status,
    Style,
    StyleInfo,
    StylePathVariant,
    SynthesisParam,
    TimeRange,
    TrimBufferSettings,
    UpdateInfo,
    UtilMora,
    ValidationError as ModelValidationError,
    WavMakingParam,
    WavProcessingParam,
    WavWithDuration,
    WorldF0,
)


def _schema_shape(model: Type[BaseModel]) -> Dict[str, object]:
    schema = model.model_json_schema(by_alias=True)
    return {
        "properties": list(schema["properties"]),
        "required": schema.get("required", []),
        "defaults": {
            name: value["default"]
            for name, value in schema["properties"].items()
            if "default" in value
        },
    }


@pytest.mark.parametrize(
    "model, properties, required, defaults",
    [
        (Status, ["status"], ["status"], {}),
        (Mora, ["phoneme", "hira", "accent"], ["phoneme", "hira", "accent"], {}),
        (
            UtilMora,
            ["text", "consonant", "consonantLength", "vowel", "vowelLength", "pitch"],
            ["text", "vowel", "vowelLength", "pitch"],
            {},
        ),
        (
            AccentPhrase,
            ["moras", "accent", "pauseMora", "isInterrogative"],
            ["moras", "accent", "isInterrogative"],
            {},
        ),
        (
            AudioQuery,
            [
                "accentPhrases",
                "speedScale",
                "pitchScale",
                "intonationScale",
                "volumeScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "outputStereo",
                "kana",
            ],
            [
                "accentPhrases",
                "speedScale",
                "pitchScale",
                "intonationScale",
                "volumeScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "outputStereo",
            ],
            {},
        ),
        (Phrase, ["detail"], ["detail"], {}),
        (Prosody, ["plain", "detail"], ["plain", "detail"], {}),
        (ProsodyMakingParam, ["text"], ["text"], {}),
        (
            WavMakingParam,
            ["speakerUuid", "styleId", "text", "prosodyDetail", "speedScale"],
            ["speakerUuid", "styleId", "text", "speedScale"],
            {},
        ),
        (TimeRange, ["start", "end"], ["start", "end"], {}),
        (PhonemeDuration, ["phoneme", "wavRange"], ["phoneme", "wavRange"], {}),
        (
            MoraDuration,
            ["mora", "hira", "phonemePitches", "wavRange"],
            ["mora", "hira", "phonemePitches", "wavRange"],
            {},
        ),
        (
            WavWithDuration,
            ["wavBase64", "moraDurations", "startTrimBuffer", "endTrimBuffer"],
            ["wavBase64", "moraDurations"],
            {"startTrimBuffer": 0.0, "endTrimBuffer": 0.0},
        ),
        (
            WavProcessingParam,
            [
                "volumeScale",
                "pitchScale",
                "intonationScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "sampledIntervalValue",
                "adjustedF0",
                "processingAlgorithm",
                "startTrimBuffer",
                "endTrimBuffer",
                "pauseLength",
                "pauseStartTrimBuffer",
                "pauseEndTrimBuffer",
                "wavBase64",
                "moraDurations",
            ],
            [
                "volumeScale",
                "pitchScale",
                "intonationScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "wavBase64",
            ],
            {},
        ),
        (
            SynthesisParam,
            [
                "volumeScale",
                "pitchScale",
                "intonationScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "sampledIntervalValue",
                "adjustedF0",
                "processingAlgorithm",
                "startTrimBuffer",
                "endTrimBuffer",
                "pauseLength",
                "pauseStartTrimBuffer",
                "pauseEndTrimBuffer",
                "speakerUuid",
                "styleId",
                "text",
                "prosodyDetail",
                "speedScale",
            ],
            [
                "volumeScale",
                "pitchScale",
                "intonationScale",
                "prePhonemeLength",
                "postPhonemeLength",
                "outputSamplingRate",
                "speakerUuid",
                "styleId",
                "text",
                "speedScale",
            ],
            {},
        ),
        (AlgorithmSettings, ["processingAlgorithm"], ["processingAlgorithm"], {}),
        (
            TrimBufferSettings,
            [
                "startTrimBuffer",
                "endTrimBuffer",
                "pauseStartTrimBuffer",
                "pauseEndTrimBuffer",
            ],
            [
                "startTrimBuffer",
                "endTrimBuffer",
                "pauseStartTrimBuffer",
                "pauseEndTrimBuffer",
            ],
            {},
        ),
        (
            DictionaryWord,
            ["word", "yomi", "accent", "numMoras"],
            ["word", "yomi", "accent", "numMoras"],
            {},
        ),
        (DictionaryWords, ["dictionaryWords"], ["dictionaryWords"], {}),
        (Style, ["name", "id"], ["name", "id"], {}),
        (
            Speaker,
            ["name", "speaker_uuid", "styles", "version"],
            ["name", "speaker_uuid", "styles", "version"],
            {},
        ),
        (StyleInfo, ["id", "icon", "voice_samples"], ["id", "icon", "voice_samples"], {}),
        (
            SpeakerInfo,
            ["policy", "portrait", "style_infos"],
            ["policy", "portrait", "style_infos"],
            {},
        ),
        (
            SpeakerMetaStyle,
            ["styleName", "styleId", "base64Icon", "base64Portrait"],
            ["styleName", "styleId", "base64Icon"],
            {},
        ),
        (
            SpeakerMeta,
            ["speakerName", "speakerUuid", "styles", "version", "base64Portrait"],
            ["speakerName", "speakerUuid", "styles", "base64Portrait"],
            {"version": "0.0.1"},
        ),
        (
            StylePathVariant,
            ["styleName", "styleId", "pathIcon", "pathPortrait"],
            ["styleName", "styleId", "pathIcon"],
            {},
        ),
        (
            SpeakerMetaPathVariant,
            ["speakerName", "speakerUuid", "styles", "version", "pathPortrait"],
            ["speakerName", "speakerUuid", "styles", "pathPortrait"],
            {"version": "0.0.1"},
        ),
        (
            SpeakerMetaForTextBox,
            ["speakerUuid", "styleId", "speakerName", "styleName"],
            ["speakerUuid", "styleId", "speakerName", "styleName"],
            {},
        ),
        (SpeakerFolderPath, ["speakerFolderPath"], ["speakerFolderPath"], {}),
        (SpeakerPolicy, ["policy", "license"], [], {}),
        (
            DownloadableStyle,
            [
                "styleName",
                "styleId",
                "version",
                "iconBase64",
                "voiceSampleBase64s",
                "downloadUrl",
            ],
            [
                "styleName",
                "styleId",
                "version",
                "iconBase64",
                "voiceSampleBase64s",
                "downloadUrl",
            ],
            {},
        ),
        (
            DownloadableSpeaker,
            [
                "speakerName",
                "speakerUuid",
                "subSpeakerUuids",
                "styles",
                "version",
                "portraitBase64",
                "metaDownloadUrl",
                "prefix",
            ],
            [
                "speakerName",
                "speakerUuid",
                "subSpeakerUuids",
                "styles",
                "version",
                "portraitBase64",
                "metaDownloadUrl",
                "prefix",
            ],
            {},
        ),
        (
            DownloadableModel,
            ["download_path", "volume", "speaker", "speaker_info"],
            ["download_path", "volume", "speaker", "speaker_info"],
            {},
        ),
        (UpdateInfo, ["version", "date", "contents"], ["version", "date", "contents"], {}),
        (WorldF0, ["f0", "moraDurations"], ["f0", "moraDurations"], {}),
        (EngineInfo, ["device", "version"], ["device", "version"], {}),
        (ModelValidationError, ["loc", "msg", "type"], ["loc", "msg", "type"], {}),
        (HTTPValidationError, ["detail"], [], {}),
    ],
)
def test_openapi_shape(model, properties, required, defaults):
    assert _schema_shape(model) == {
        "properties": properties,
        "required": required,
        "defaults": defaults,
    }


def test_camel_case_aliases_accept_and_emit_wire_names():
    payload = {
        "speakerName": "テスト話者",
        "speakerUuid": "00000000-0000-0000-0000-000000000003",
        "styles": [
            {"styleName": "標準", "styleId": 1003, "base64Icon": "icon"}
        ],
        "base64Portrait": "portrait",
    }

    model = SpeakerMeta.model_validate(payload)

    assert model.speaker_name == "テスト話者"
    assert model.styles[0].style_id == 1003
    assert model.model_dump(by_alias=True) == {
        **payload,
        "version": "0.0.1",
        "styles": [
            {
                "styleName": "標準",
                "styleId": 1003,
                "base64Icon": "icon",
                "base64Portrait": None,
            }
        ],
    }


def test_snake_case_wire_models_do_not_get_camel_case_aliases():
    info = SpeakerInfo(
        policy="policy",
        portrait="portrait",
        style_infos=[StyleInfo(id=1, icon="icon", voice_samples=["sample"])],
    )

    assert info.model_dump(by_alias=True) == {
        "policy": "policy",
        "portrait": "portrait",
        "style_infos": [{"id": 1, "icon": "icon", "voice_samples": ["sample"]}],
    }


def test_required_and_optional_fields_follow_the_contract():
    with pytest.raises(ValidationError):
        AccentPhrase(moras=[], accent=1)

    phrase = AccentPhrase(moras=[], accent=1, isInterrogative=False)
    assert phrase.pause_mora is None

    wav = WavWithDuration(wavBase64="", moraDurations=[])
    assert wav.start_trim_buffer == 0.0
    assert wav.end_trim_buffer == 0.0

    assert SpeakerPolicy().policy is None
    assert SpeakerPolicy().license is None

    param = SynthesisParam(
        volumeScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=44100,
        speakerUuid="speaker",
        styleId=1,
        text="テスト",
        speedScale=1.0,
    )
    assert param.prosody_detail is None
    assert param.processing_algorithm is None


def test_nested_wire_types_are_the_expected_mora_variants():
    query = AudioQuery(
        accentPhrases=[
            {
                "moras": [
                    {
                        "text": "テ",
                        "vowel": "e",
                        "vowelLength": 0.1,
                        "pitch": 5.0,
                    }
                ],
                "accent": 1,
                "isInterrogative": False,
            }
        ],
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=44100,
        outputStereo=False,
    )
    prosody = Prosody(plain=["テ"], detail=[[{"phoneme": "e", "hira": "て", "accent": 0}]])

    assert isinstance(query.accent_phrases[0].moras[0], UtilMora)
    assert isinstance(prosody.detail[0][0], Mora)


def test_pydantic_v1_model_serialization_is_available():
    assert issubclass(Status, BaseModel)
    assert Status(status="ok").json() == '{"status": "ok"}'
