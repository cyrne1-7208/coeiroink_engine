"""話者・スタイルのメタデータモデル。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StyleType(str, Enum):
    TALK = "talk"
    SINGING_TEACHER = "singing_teacher"
    FRAME_DECODE = "frame_decode"
    SING = "sing"


class SpeakerStyle(BaseModel):
    """
    スピーカーのスタイル情報
    """

    name: str = Field(title="スタイル名")
    id: int = Field(title="スタイルID")
    type: StyleType = Field(
        default=StyleType.TALK,
        title="スタイルの種類",
        description=(
            "talk:音声合成クエリの作成と音声合成が可能。"
            "COEIROINKではtalkのみを提供します。"
        ),
    )


class SpeakerSupportPermittedSynthesisMorphing(str, Enum):
    ALL = "ALL"  # 全て許可
    SELF_ONLY = "SELF_ONLY"  # 同じ話者内でのみ許可
    NOTHING = "NOTHING"  # 全て禁止

    @classmethod
    def _missing_(cls, value: object) -> SpeakerSupportPermittedSynthesisMorphing:
        return SpeakerSupportPermittedSynthesisMorphing.ALL


class SpeakerSupportedFeatures(BaseModel):
    """
    話者の対応機能の情報
    """

    permitted_synthesis_morphing: SpeakerSupportPermittedSynthesisMorphing = Field(
        title="モーフィング機能への対応",
        default=SpeakerSupportPermittedSynthesisMorphing(None),
    )


class CoreSpeaker(BaseModel):
    """
    コアに含まれるスピーカー情報
    """

    name: str = Field(title="名前")
    speaker_uuid: str = Field(title="スピーカーのUUID")
    styles: list[SpeakerStyle] = Field(title="スピーカースタイルの一覧")
    version: str = Field(title="スピーカーのバージョン")


class EngineSpeaker(BaseModel):
    """
    エンジンに含まれるスピーカー情報
    """

    supported_features: SpeakerSupportedFeatures = Field(
        title="スピーカーの対応機能", default_factory=SpeakerSupportedFeatures
    )


class Speaker(CoreSpeaker, EngineSpeaker):
    """
    スピーカー情報
    """


class StyleInfo(BaseModel):
    """
    スタイルの追加情報
    """

    id: int = Field(title="スタイルID")
    icon: str = Field(title="当該スタイルのアイコンをbase64エンコードしたもの")
    portrait: str | None = Field(
        default=None,
        title="当該スタイルのportrait.pngをbase64エンコードしたもの",
    )
    voice_samples: list[str] = Field(
        title="voice_sampleのwavファイルをbase64エンコードしたもの"
    )


class SpeakerInfo(BaseModel):
    """
    話者の追加情報
    """

    policy: str = Field(title="policy.md")
    portrait: str = Field(title="portrait.pngをbase64エンコードしたもの")
    style_infos: list[StyleInfo] = Field(title="スタイルの追加情報")
