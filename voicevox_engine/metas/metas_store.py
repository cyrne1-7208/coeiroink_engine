"""Coreごとのメタデータを統合するストア。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from voicevox_engine.metas.metas import CoreSpeaker, EngineSpeaker, Speaker, StyleInfo

if TYPE_CHECKING:
    from voicevox_engine.synthesis_engine.synthesis_engine_base import (
        SynthesisEngineBase,
    )


class MetasStore:
    """
    話者やスタイルのメタ情報を管理する
    """

    def __init__(self, engine_speakers_path: Path) -> None:
        self._engine_speakers_path = engine_speakers_path
        self._loaded_metas: dict[str, EngineSpeaker] = {}
        self._speaker_paths: dict[str, Path] = {}

        for folder in sorted(engine_speakers_path.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            meta = json.loads((folder / "metas.json").read_text(encoding="utf-8"))
            speaker_uuid = meta.get("speakerUuid")
            if not isinstance(speaker_uuid, str) or not speaker_uuid:
                raise ValueError(f"speakerUuid is missing or invalid: {folder}")
            if speaker_uuid in self._loaded_metas:
                raise ValueError(f"Duplicate speakerUuid: {speaker_uuid}")

            self._loaded_metas[speaker_uuid] = EngineSpeaker(**meta)
            self._speaker_paths[speaker_uuid] = folder

    def speaker_engine_metas(self, speaker_uuid: str) -> EngineSpeaker:
        return self.loaded_metas[speaker_uuid]

    def speaker_path(self, speaker_uuid: str) -> Path:
        return self._speaker_paths[speaker_uuid]

    def combine_metas(self, core_metas: list[CoreSpeaker]) -> list[Speaker]:
        """
        与えられたmetaにエンジンのコア情報を付加して返す
        core_metas: コアのmetas()が返すJSONのModel
        """

        return [
            Speaker(
                **self.speaker_engine_metas(speaker_meta.speaker_uuid).model_dump(),
                **speaker_meta.model_dump(),
            )
            for speaker_meta in core_metas
        ]

    # FIXME: engineではなくList[CoreSpeaker]を渡す形にすることで
    # SynthesisEngineBaseによる循環importを修正する
    def load_combined_metas(self, engine: SynthesisEngineBase) -> list[Speaker]:
        """
        与えられたエンジンから、コア・エンジン両方の情報を含んだMetasを返す
        """

        core_metas = [CoreSpeaker(**speaker) for speaker in json.loads(engine.speakers)]
        return self.combine_metas(core_metas)

    @property
    def engine_speakers_path(self) -> Path:
        return self._engine_speakers_path

    @property
    def loaded_metas(self) -> dict[str, EngineSpeaker]:
        return self._loaded_metas


def construct_lookup(speakers: list[Speaker]) -> dict[int, tuple[Speaker, StyleInfo]]:
    """
    `{style.id: StyleInfo}`の変換テーブル
    """

    lookup_table = {}
    for speaker in speakers:
        for style in speaker.styles:
            lookup_table[style.id] = (speaker, style)
    return lookup_table
