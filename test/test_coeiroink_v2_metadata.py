import base64
import json
from pathlib import Path

import pytest

from voicevox_engine.coeiroink_v2.metadata import (
    AmbiguousStyleError,
    MetadataAssetNotFoundError,
    MetadataError,
    SpeakerMetadataStore,
    SpeakerNotFoundError,
    StyleNotFoundError,
)

FIRST_SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"
FIRST_STYLE_ID = 1001
SECOND_SPEAKER_UUID = "00000000-0000-4000-8000-000000000002"
SECOND_STYLE_ID = 1002


@pytest.fixture(scope="module")
def metadata_store(tmp_path_factory) -> SpeakerMetadataStore:
    speaker_info = tmp_path_factory.mktemp("speaker_info")
    speakers = (
        (
            FIRST_SPEAKER_UUID,
            FIRST_SPEAKER_UUID,
            "テスト話者A",
            FIRST_STYLE_ID,
            "ノーマル",
        ),
        (
            "speaker_ver1.0",
            SECOND_SPEAKER_UUID,
            "テスト話者B",
            SECOND_STYLE_ID,
            "別スタイル",
        ),
    )
    for folder_name, speaker_uuid, speaker_name, style_id, style_name in speakers:
        folder = speaker_info / folder_name
        (folder / "icons").mkdir(parents=True)
        (folder / "voice_samples").mkdir()
        (folder / "metas.json").write_text(
            json.dumps(
                {
                    "speakerName": speaker_name,
                    "speakerUuid": speaker_uuid,
                    "styles": [{"styleName": style_name, "styleId": style_id}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (folder / "portrait.png").write_bytes(
            (speaker_uuid + "-portrait").encode("utf-8")
        )
        (folder / "icons" / f"{style_id}.png").write_bytes(
            (speaker_uuid + "-icon").encode("utf-8")
        )
        (folder / "policy.md").write_text(
            "policy for " + speaker_name, encoding="utf-8"
        )
        (folder / "LICENSE.txt").write_text(
            "license for " + speaker_name, encoding="utf-8"
        )
        for index in range(1, 4):
            (folder / "voice_samples" / f"{style_id}_{index:03d}.wav").write_bytes(
                f"sample-{style_id}-{index}".encode("ascii")
            )
    (speaker_info / "speaker_ver1.0" / "LICENSE_extra.txt").write_text(
        "additional test license", encoding="utf-8"
    )
    return SpeakerMetadataStore(speaker_info)


def test_speakers_include_base64_portraits_and_icons(metadata_store):
    speakers = metadata_store.list_speakers()

    assert [speaker.speaker_uuid for speaker in speakers] == [
        FIRST_SPEAKER_UUID,
        SECOND_SPEAKER_UUID,
    ]
    assert [style.style_id for style in speakers[0].styles] == [FIRST_STYLE_ID]
    assert [style.style_id for style in speakers[1].styles] == [SECOND_STYLE_ID]

    first = speakers[0]
    assert (
        base64.b64decode(first.base64_portrait)
        == (
            metadata_store.speaker_info_dir / FIRST_SPEAKER_UUID / "portrait.png"
        ).read_bytes()
    )
    assert (
        base64.b64decode(first.styles[0].base64_icon)
        == (
            metadata_store.speaker_info_dir
            / FIRST_SPEAKER_UUID
            / "icons"
            / f"{FIRST_STYLE_ID}.png"
        ).read_bytes()
    )
    assert first.styles[0].base64_portrait is None


def test_speakers_path_variant_uses_physical_folder(metadata_store):
    variants = metadata_store.list_speakers_path_variant()

    assert [variant.speaker_uuid for variant in variants] == [
        FIRST_SPEAKER_UUID,
        SECOND_SPEAKER_UUID,
    ]
    second = variants[1]
    assert Path(second.path_portrait).parent.name == "speaker_ver1.0"
    assert Path(second.styles[0].path_icon).parent.parent.name == "speaker_ver1.0"


def test_uuid_and_style_id_resolve_independently_of_folder_name(metadata_store):
    assert metadata_store.speaker_path(SECOND_SPEAKER_UUID).name == "speaker_ver1.0"
    assert metadata_store.lookup_speaker_folder(
        SECOND_SPEAKER_UUID
    ) == metadata_store.speaker_path(SECOND_SPEAKER_UUID)
    assert (
        metadata_store.get_style(SECOND_SPEAKER_UUID, SECOND_STYLE_ID).style_name
        == "別スタイル"
    )
    assert metadata_store.speaker_meta_for_style(
        SECOND_STYLE_ID, SECOND_SPEAKER_UUID
    ).model_dump(by_alias=True) == {
        "speakerUuid": SECOND_SPEAKER_UUID,
        "styleId": SECOND_STYLE_ID,
        "speakerName": "テスト話者B",
        "styleName": "別スタイル",
    }

    with pytest.raises(SpeakerNotFoundError, match="speakerUuid"):
        metadata_store.speaker_path("missing-speaker")
    with pytest.raises(StyleNotFoundError, match="styleId"):
        metadata_store.get_style(SECOND_SPEAKER_UUID, FIRST_STYLE_ID)


def test_style_lookup_rejects_ambiguous_style_ids(tmp_path: Path):
    for folder_name, speaker_uuid in (
        ("first", "speaker-one"),
        ("second", "speaker-two"),
    ):
        folder = tmp_path / folder_name
        folder.mkdir()
        (folder / "metas.json").write_text(
            json.dumps(
                {
                    "speakerName": folder_name,
                    "speakerUuid": speaker_uuid,
                    "styles": [{"styleName": "same", "styleId": 7}],
                }
            ),
            encoding="utf-8",
        )

    store = SpeakerMetadataStore(tmp_path)
    with pytest.raises(AmbiguousStyleError, match="specify speakerUuid"):
        store.find_style(7)
    assert store.find_style(7, "speaker-two")[0] == "speaker-two"
    assert store.style_id_to_speaker_meta(7).model_dump(by_alias=True) == {
        "speakerUuid": "speaker-two",
        "styleId": 7,
        "speakerName": "second",
        "styleName": "same",
    }


def test_sample_voice_lookup_is_zero_based_and_deterministic(metadata_store):
    paths = metadata_store.voice_sample_paths(SECOND_SPEAKER_UUID, SECOND_STYLE_ID)
    assert [path.name for path in paths] == [
        f"{SECOND_STYLE_ID}_001.wav",
        f"{SECOND_STYLE_ID}_002.wav",
        f"{SECOND_STYLE_ID}_003.wav",
    ]
    assert (
        metadata_store.sample_voice_path(SECOND_SPEAKER_UUID, SECOND_STYLE_ID, 0)
        == paths[0]
    )
    assert (
        metadata_store.read_sample_voice(SECOND_SPEAKER_UUID, SECOND_STYLE_ID, 1)
        == paths[1].read_bytes()
    )
    assert (
        base64.b64decode(
            metadata_store.sample_voice_base64(SECOND_SPEAKER_UUID, SECOND_STYLE_ID, 2)
        )
        == paths[2].read_bytes()
    )

    with pytest.raises(MetadataAssetNotFoundError, match="index 3"):
        metadata_store.sample_voice_path(SECOND_SPEAKER_UUID, SECOND_STYLE_ID, 3)


def test_policy_and_license_reading(metadata_store):
    policy = metadata_store.speaker_policy(SECOND_SPEAKER_UUID)
    assert policy.policy == (
        metadata_store.speaker_info_dir / "speaker_ver1.0" / "policy.md"
    ).read_text(encoding="utf-8")
    assert policy.license == (
        metadata_store.speaker_info_dir / "speaker_ver1.0" / "LICENSE.txt"
    ).read_text(encoding="utf-8")
    assert [
        path.name for path in metadata_store.license_paths(SECOND_SPEAKER_UUID)
    ] == [
        "LICENSE.txt",
        "LICENSE_extra.txt",
    ]


def test_repeated_listing_has_stable_order(metadata_store):
    first = [speaker.model_dump(by_alias=True) for speaker in metadata_store.speakers()]
    second = [
        speaker.model_dump(by_alias=True) for speaker in metadata_store.speakers()
    ]
    assert first == second


def test_duplicate_style_ids_in_one_metadata_file_are_rejected(tmp_path: Path):
    folder = tmp_path / "speaker"
    folder.mkdir()
    (folder / "metas.json").write_text(
        json.dumps(
            {
                "speakerName": "duplicate",
                "speakerUuid": "duplicate-speaker",
                "styles": [
                    {"styleName": "one", "styleId": 1},
                    {"styleName": "two", "styleId": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MetadataError, match="duplicate styleId"):
        SpeakerMetadataStore(tmp_path)
