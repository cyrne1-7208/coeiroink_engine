"""全DTOの写しではなく、v1互換処理と混在するwire形式の境界を検証する。"""

from voicevox_engine.coeiroink_v2.models import (
    AccentPhrase,
    SpeakerInfo,
    SpeakerMeta,
    Status,
    StyleInfo,
    WavWithDuration,
)


def test_camel_case_aliases_accept_and_emit_nested_wire_names():
    payload = {
        "speakerName": "テスト話者",
        "speakerUuid": "00000000-0000-4000-8000-000000000001",
        "styles": [{"styleName": "テスト", "styleId": 1001, "base64Icon": "icon"}],
        "base64Portrait": "portrait",
    }
    model = SpeakerMeta.model_validate(payload)
    assert model.speaker_name == "テスト話者"
    assert model.styles[0].style_id == 1001
    assert model.model_dump(by_alias=True) == {
        **payload,
        "version": "0.0.1",
        "styles": [{**payload["styles"][0], "base64Portrait": None}],
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


def test_compatibility_schema_strips_only_null_defaults_including_nested_models():
    schema = AccentPhrase.model_json_schema(by_alias=True)
    assert "default" not in schema["properties"]["pauseMora"]
    assert "default" not in schema["$defs"]["UtilMora"]["properties"]["consonant"]
    assert "isInterrogative" in schema["required"]
    waveform = WavWithDuration.model_json_schema(by_alias=True)
    assert waveform["properties"]["startTrimBuffer"]["default"] == 0.0


def test_legacy_dict_comparison_and_json_formatting():
    status = Status(status="ok")
    assert status == {"status": "ok"}
    assert status != {"status": "other"}
    assert status.json() == '{"status": "ok"}'
