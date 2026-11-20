# モーフィング可否判定のテスト

from __future__ import annotations

import pytest

from voicevox_engine.metas.metas import (
    Speaker,
    SpeakerStyle,
    SpeakerSupportedFeatures,
    SpeakerSupportPermittedSynthesisMorphing,
)
from voicevox_engine.metas.metas_store import construct_lookup
from voicevox_engine.model import SpeakerNotFoundError
from voicevox_engine.morphing import is_synthesis_morphing_permitted


def _speaker(speaker_uuid: str, style_id: int) -> Speaker:
    return Speaker(
        name=speaker_uuid,
        speaker_uuid=speaker_uuid,
        styles=[SpeakerStyle(name="ノーマル", id=style_id)],
        version="1.0.0",
        supported_features=SpeakerSupportedFeatures(
            permitted_synthesis_morphing=SpeakerSupportPermittedSynthesisMorphing.ALL
        ),
    )


def test_known_styles_follow_permission() -> None:
    lookup = construct_lookup([_speaker("speaker-a", 1), _speaker("speaker-b", 2)])

    assert is_synthesis_morphing_permitted(lookup, 1, 2) is True


# 未登録のスタイルIDはKeyErrorではなくSpeakerNotFoundErrorとして報告する。
@pytest.mark.parametrize(("base_speaker", "target_speaker"), [(999, 1), (1, 999)])
def test_unknown_style_raises_speaker_not_found(
    base_speaker: int, target_speaker: int
) -> None:
    lookup = construct_lookup([_speaker("speaker-a", 1)])

    with pytest.raises(SpeakerNotFoundError) as error:
        is_synthesis_morphing_permitted(lookup, base_speaker, target_speaker)

    assert error.value.speaker == 999
