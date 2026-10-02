"""COEIROINK v2 HTTPサーバー契約のPydanticモデルです。
フィールド定義は公開OpenAPI契約から再構成し、HTTPルート・GUI・ダウンロード・合成処理は含めません。
APIには``Mora``という名前の異なるモデルが2つあります。
ここでの``Mora``は韻律系エンドポイントの``coeirocore.mora.Mora``形状、``UtilMora``はAudioQueryの``coeirocore.v_util.Mora``形状です。
"""

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class V2BaseModel(BaseModel):
    """v2通信モデルで共通に使うPydantic設定です。"""

    # Python形式のフィールド名とAPIの別名を両方受け付け、FastAPIの応答では別名を出力します。
    model_config = ConfigDict(populate_by_name=True)

    def __eq__(self, other: object) -> bool:
        """このAPIが使ってきたPydantic 1のモデルと辞書の比較を維持します。
        Pydantic 2では``model == dict``が一致しなくなったため、v2変換層との互換境界で従来の比較を保ちます。
        """

        if isinstance(other, dict):
            return self.model_dump() == other
        return super().__eq__(other)

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """暗黙の``null``デフォルトを除いた旧通信スキーマを出力します。
        Pydantic 2はNoneを既定値に持つ任意フィールドへ``default: null``を追加するため、実値の既定値は残して暗黙の項目だけを削除します。
        """

        schema = super().model_json_schema(*args, **kwargs)

        def strip_implicit_null_defaults(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("default") is None:
                    value.pop("default", None)
                for child in value.values():
                    strip_implicit_null_defaults(child)
            elif isinstance(value, list):
                for child in value:
                    strip_implicit_null_defaults(child)

        strip_implicit_null_defaults(schema)
        return schema

    def json(self, *args: Any, **kwargs: Any) -> str:
        """呼び出し元が使うPydantic 1互換のJSON形式を提供します。"""

        if args:
            raise TypeError("positional arguments are not supported")
        include = kwargs.pop("include", None)
        exclude = kwargs.pop("exclude", None)
        by_alias = kwargs.pop("by_alias", False)
        exclude_unset = kwargs.pop("exclude_unset", False)
        exclude_defaults = kwargs.pop("exclude_defaults", False)
        exclude_none = kwargs.pop("exclude_none", False)
        kwargs.pop("skip_defaults", None)
        encoder = kwargs.pop("encoder", None)
        if encoder is not None:
            kwargs.setdefault("default", encoder)
        return json.dumps(
            self.model_dump(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            ),
            **kwargs,
        )


class Mora(V2BaseModel):
    """韻律・波形パラメータモデルで使うモーラです。"""

    phoneme: str
    hira: str
    accent: int


class UtilMora(V2BaseModel):
    """AudioQueryとAccentPhraseの内部で使うモーラです。"""

    text: str
    consonant: Optional[str] = None
    consonant_length: Optional[float] = Field(None, alias="consonantLength")
    vowel: str
    vowel_length: float = Field(..., alias="vowelLength")
    pitch: float


# OpenAPI生成名に依存せず2つのAPIスキーマを区別したい呼び出し元向けの説明的な別名です。
AudioQueryMora = UtilMora
VUtilMora = UtilMora


class AccentPhrase(V2BaseModel):
    moras: List[UtilMora]
    accent: int
    pause_mora: Optional[UtilMora] = Field(None, alias="pauseMora")
    is_interrogative: bool = Field(..., alias="isInterrogative")


class AudioQuery(V2BaseModel):
    accent_phrases: List[AccentPhrase] = Field(..., alias="accentPhrases")
    speed_scale: float = Field(..., alias="speedScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    volume_scale: float = Field(..., alias="volumeScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    output_stereo: bool = Field(..., alias="outputStereo")
    kana: Optional[str] = None


class Phrase(V2BaseModel):
    detail: List[List[Mora]]


class Prosody(V2BaseModel):
    plain: List[str]
    detail: List[List[Mora]]


class ProsodyMakingParam(V2BaseModel):
    text: str


class WavMakingParam(V2BaseModel):
    speaker_uuid: str = Field(..., alias="speakerUuid")
    style_id: int = Field(..., alias="styleId")
    text: str
    prosody_detail: Optional[List[List[Mora]]] = Field(None, alias="prosodyDetail")
    speed_scale: float = Field(..., alias="speedScale")


class TimeRange(V2BaseModel):
    start: int
    end: int


class PhonemeDuration(V2BaseModel):
    phoneme: str
    wav_range: TimeRange = Field(..., alias="wavRange")


class MoraDuration(V2BaseModel):
    mora: str
    hira: str
    phoneme_pitches: List[PhonemeDuration] = Field(..., alias="phonemePitches")
    wav_range: TimeRange = Field(..., alias="wavRange")


class WavWithDuration(V2BaseModel):
    wav_base64: str = Field(..., alias="wavBase64")
    mora_durations: List[MoraDuration] = Field(..., alias="moraDurations")
    start_trim_buffer: float = Field(0.0, alias="startTrimBuffer")
    end_trim_buffer: float = Field(0.0, alias="endTrimBuffer")


class WavProcessingParam(V2BaseModel):
    volume_scale: float = Field(..., alias="volumeScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    sampled_interval_value: Optional[int] = Field(
        None, alias="sampledIntervalValue"
    )
    adjusted_f0: Optional[List[float]] = Field(None, alias="adjustedF0")
    processing_algorithm: Optional[str] = Field(None, alias="processingAlgorithm")
    start_trim_buffer: Optional[float] = Field(None, alias="startTrimBuffer")
    end_trim_buffer: Optional[float] = Field(None, alias="endTrimBuffer")
    pause_length: Optional[float] = Field(None, alias="pauseLength")
    pause_start_trim_buffer: Optional[float] = Field(
        None, alias="pauseStartTrimBuffer"
    )
    pause_end_trim_buffer: Optional[float] = Field(None, alias="pauseEndTrimBuffer")
    wav_base64: str = Field(..., alias="wavBase64")
    mora_durations: Optional[List[MoraDuration]] = Field(
        None, alias="moraDurations"
    )


class SynthesisParam(V2BaseModel):
    volume_scale: float = Field(..., alias="volumeScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    sampled_interval_value: Optional[int] = Field(
        None, alias="sampledIntervalValue"
    )
    adjusted_f0: Optional[List[float]] = Field(None, alias="adjustedF0")
    processing_algorithm: Optional[str] = Field(None, alias="processingAlgorithm")
    start_trim_buffer: Optional[float] = Field(None, alias="startTrimBuffer")
    end_trim_buffer: Optional[float] = Field(None, alias="endTrimBuffer")
    pause_length: Optional[float] = Field(None, alias="pauseLength")
    pause_start_trim_buffer: Optional[float] = Field(
        None, alias="pauseStartTrimBuffer"
    )
    pause_end_trim_buffer: Optional[float] = Field(None, alias="pauseEndTrimBuffer")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    style_id: int = Field(..., alias="styleId")
    text: str
    prosody_detail: Optional[List[List[Mora]]] = Field(None, alias="prosodyDetail")
    speed_scale: float = Field(..., alias="speedScale")


class AlgorithmSettings(V2BaseModel):
    processing_algorithm: str = Field(..., alias="processingAlgorithm")


class TrimBufferSettings(V2BaseModel):
    start_trim_buffer: float = Field(..., alias="startTrimBuffer")
    end_trim_buffer: float = Field(..., alias="endTrimBuffer")
    pause_start_trim_buffer: float = Field(..., alias="pauseStartTrimBuffer")
    pause_end_trim_buffer: float = Field(..., alias="pauseEndTrimBuffer")


class DictionaryWord(V2BaseModel):
    word: str
    yomi: str
    accent: int
    num_moras: int = Field(..., alias="numMoras")


class DictionaryWords(V2BaseModel):
    dictionary_words: List[DictionaryWord] = Field(..., alias="dictionaryWords")


class Style(V2BaseModel):
    """``/v1/speakers``が返す簡潔なスタイル情報です。"""

    name: str
    id: int


class Speaker(V2BaseModel):
    name: str
    speaker_uuid: str
    styles: List[Style]
    version: str


class StyleInfo(V2BaseModel):
    id: int
    icon: str
    voice_samples: List[str]


class SpeakerInfo(V2BaseModel):
    policy: str
    portrait: str
    style_infos: List[StyleInfo]


class SpeakerMetaStyle(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    base64_icon: str = Field(..., alias="base64Icon")
    base64_portrait: Optional[str] = Field(None, alias="base64Portrait")


class SpeakerMeta(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    styles: List[SpeakerMetaStyle]
    version: str = "0.0.1"
    base64_portrait: str = Field(..., alias="base64Portrait")


class StylePathVariant(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    path_icon: str = Field(..., alias="pathIcon")
    path_portrait: Optional[str] = Field(None, alias="pathPortrait")


class SpeakerMetaPathVariant(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    styles: List[StylePathVariant]
    version: str = "0.0.1"
    path_portrait: str = Field(..., alias="pathPortrait")


class SpeakerMetaForTextBox(V2BaseModel):
    speaker_uuid: str = Field(..., alias="speakerUuid")
    style_id: int = Field(..., alias="styleId")
    speaker_name: str = Field(..., alias="speakerName")
    style_name: str = Field(..., alias="styleName")


class SpeakerFolderPath(V2BaseModel):
    speaker_folder_path: str = Field(..., alias="speakerFolderPath")


class SpeakerPolicy(V2BaseModel):
    policy: Optional[str] = None
    license: Optional[str] = None


class DownloadableStyle(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    version: str
    icon_base64: str = Field(..., alias="iconBase64")
    voice_sample_base64s: List[str] = Field(..., alias="voiceSampleBase64s")
    download_url: str = Field(..., alias="downloadUrl")


class DownloadableSpeaker(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    sub_speaker_uuids: List[str] = Field(..., alias="subSpeakerUuids")
    styles: List[DownloadableStyle]
    version: str
    portrait_base64: str = Field(..., alias="portraitBase64")
    meta_download_url: str = Field(..., alias="metaDownloadUrl")
    prefix: str


class DownloadableModel(V2BaseModel):
    download_path: str
    volume: str
    speaker: Speaker
    speaker_info: SpeakerInfo


class UpdateInfo(V2BaseModel):
    version: str
    date: str
    contents: List[str]


class WorldF0(V2BaseModel):
    f0: List[float]
    mora_durations: List[MoraDuration] = Field(..., alias="moraDurations")


class EngineInfo(V2BaseModel):
    device: str
    version: str


class Status(V2BaseModel):
    status: str


class ValidationError(V2BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class HTTPValidationError(V2BaseModel):
    detail: Optional[List[ValidationError]] = None


__all__ = [
    "AccentPhrase",
    "AlgorithmSettings",
    "AudioQuery",
    "AudioQueryMora",
    "DictionaryWord",
    "DictionaryWords",
    "DownloadableModel",
    "DownloadableSpeaker",
    "DownloadableStyle",
    "EngineInfo",
    "HTTPValidationError",
    "Mora",
    "MoraDuration",
    "PhonemeDuration",
    "Phrase",
    "Prosody",
    "ProsodyMakingParam",
    "Speaker",
    "SpeakerFolderPath",
    "SpeakerInfo",
    "SpeakerMeta",
    "SpeakerMetaForTextBox",
    "SpeakerMetaPathVariant",
    "SpeakerMetaStyle",
    "SpeakerPolicy",
    "Status",
    "Style",
    "StyleInfo",
    "StylePathVariant",
    "SynthesisParam",
    "TimeRange",
    "TrimBufferSettings",
    "UpdateInfo",
    "UtilMora",
    "VUtilMora",
    "ValidationError",
    "WavMakingParam",
    "WavProcessingParam",
    "WavWithDuration",
    "WorldF0",
]
