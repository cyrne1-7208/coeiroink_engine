"""Pydantic models for the COEIROINK v2 HTTP server contract.

The field definitions in this module are reconstructed from the public
OpenAPI contract.  It intentionally contains only data models: HTTP routes,
GUI behavior, downloading, and synthesis implementation belong elsewhere.

The API has two different models named ``Mora``.  ``Mora`` below is the
``coeirocore.mora.Mora`` shape used by prosody-related endpoints, while
``UtilMora`` is the ``coeirocore.v_util.Mora`` shape used by AudioQuery.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class V2BaseModel(BaseModel):
    """Common Pydantic configuration for the v2 wire models."""

    # Python側の属性名とAPI上のaliasをどちらも入力で受け付け、FastAPIの応答ではaliasを維持する。
    model_config = ConfigDict(populate_by_name=True)

    def __eq__(self, other: object) -> bool:
        """Keep the Pydantic 1 model-to-dict comparison used by this API.

        Pydantic 2 intentionally stopped treating ``model == dict`` as equal.
        The v2 conversion layer and its callers historically relied on that
        behavior, so preserve it at this compatibility boundary.
        """

        if isinstance(other, dict):
            return self.model_dump() == other
        return super().__eq__(other)

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Emit the old wire schema without implicit ``null`` defaults.

        Pydantic 2 adds ``default: null`` for optional fields with a
        ``None`` default.  The existing COEIROINK v2 schema did not expose
        those entries, so remove only these implicit defaults while retaining
        real defaults such as ``0.0`` and ``"0.0.1"``.
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
        """Provide the Pydantic 1-compatible JSON formatting used by callers."""

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
    """A mora used by prosody and waveform parameter models."""

    phoneme: str
    hira: str
    accent: int


class UtilMora(V2BaseModel):
    """A mora used inside the AudioQuery/AccentPhrase representation."""

    text: str
    consonant: str | None = None
    consonant_length: float | None = Field(None, alias="consonantLength")
    vowel: str
    vowel_length: float = Field(..., alias="vowelLength")
    pitch: float


# OpenAPIが生成するコンポーネント名に依存せず、2種類のMora schemaを区別したい呼出元向けの別名。
AudioQueryMora = UtilMora
VUtilMora = UtilMora


class AccentPhrase(V2BaseModel):
    moras: list[UtilMora]
    accent: int
    pause_mora: UtilMora | None = Field(None, alias="pauseMora")
    is_interrogative: bool = Field(..., alias="isInterrogative")


class AudioQuery(V2BaseModel):
    accent_phrases: list[AccentPhrase] = Field(..., alias="accentPhrases")
    speed_scale: float = Field(..., alias="speedScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    volume_scale: float = Field(..., alias="volumeScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    output_stereo: bool = Field(..., alias="outputStereo")
    kana: str | None = None


class Phrase(V2BaseModel):
    detail: list[list[Mora]]


class Prosody(V2BaseModel):
    plain: list[str]
    detail: list[list[Mora]]


class ProsodyMakingParam(V2BaseModel):
    text: str


class WavMakingParam(V2BaseModel):
    speaker_uuid: str = Field(..., alias="speakerUuid")
    style_id: int = Field(..., alias="styleId")
    text: str
    prosody_detail: list[list[Mora]] | None = Field(None, alias="prosodyDetail")
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
    phoneme_pitches: list[PhonemeDuration] = Field(..., alias="phonemePitches")
    wav_range: TimeRange = Field(..., alias="wavRange")


class WavWithDuration(V2BaseModel):
    wav_base64: str = Field(..., alias="wavBase64")
    mora_durations: list[MoraDuration] = Field(..., alias="moraDurations")
    start_trim_buffer: float = Field(0.0, alias="startTrimBuffer")
    end_trim_buffer: float = Field(0.0, alias="endTrimBuffer")


class WavProcessingParam(V2BaseModel):
    volume_scale: float = Field(..., alias="volumeScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    sampled_interval_value: int | None = Field(None, alias="sampledIntervalValue")
    adjusted_f0: list[float] | None = Field(None, alias="adjustedF0")
    processing_algorithm: str | None = Field(None, alias="processingAlgorithm")
    start_trim_buffer: float | None = Field(None, alias="startTrimBuffer")
    end_trim_buffer: float | None = Field(None, alias="endTrimBuffer")
    pause_length: float | None = Field(None, alias="pauseLength")
    pause_start_trim_buffer: float | None = Field(None, alias="pauseStartTrimBuffer")
    pause_end_trim_buffer: float | None = Field(None, alias="pauseEndTrimBuffer")
    wav_base64: str = Field(..., alias="wavBase64")
    mora_durations: list[MoraDuration] | None = Field(None, alias="moraDurations")


class SynthesisParam(V2BaseModel):
    volume_scale: float = Field(..., alias="volumeScale")
    pitch_scale: float = Field(..., alias="pitchScale")
    intonation_scale: float = Field(..., alias="intonationScale")
    pre_phoneme_length: float = Field(..., alias="prePhonemeLength")
    post_phoneme_length: float = Field(..., alias="postPhonemeLength")
    output_sampling_rate: int = Field(..., alias="outputSamplingRate")
    sampled_interval_value: int | None = Field(None, alias="sampledIntervalValue")
    adjusted_f0: list[float] | None = Field(None, alias="adjustedF0")
    processing_algorithm: str | None = Field(None, alias="processingAlgorithm")
    start_trim_buffer: float | None = Field(None, alias="startTrimBuffer")
    end_trim_buffer: float | None = Field(None, alias="endTrimBuffer")
    pause_length: float | None = Field(None, alias="pauseLength")
    pause_start_trim_buffer: float | None = Field(None, alias="pauseStartTrimBuffer")
    pause_end_trim_buffer: float | None = Field(None, alias="pauseEndTrimBuffer")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    style_id: int = Field(..., alias="styleId")
    text: str
    prosody_detail: list[list[Mora]] | None = Field(None, alias="prosodyDetail")
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
    dictionary_words: list[DictionaryWord] = Field(..., alias="dictionaryWords")


class Style(V2BaseModel):
    """The compact style object returned by ``/v1/speakers``."""

    name: str
    id: int


class Speaker(V2BaseModel):
    name: str
    speaker_uuid: str
    styles: list[Style]
    version: str


class StyleInfo(V2BaseModel):
    id: int
    icon: str
    voice_samples: list[str]


class SpeakerInfo(V2BaseModel):
    policy: str
    portrait: str
    style_infos: list[StyleInfo]


class SpeakerMetaStyle(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    base64_icon: str = Field(..., alias="base64Icon")
    base64_portrait: str | None = Field(None, alias="base64Portrait")


class SpeakerMeta(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    styles: list[SpeakerMetaStyle]
    version: str = "0.0.1"
    base64_portrait: str = Field(..., alias="base64Portrait")


class StylePathVariant(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    path_icon: str = Field(..., alias="pathIcon")
    path_portrait: str | None = Field(None, alias="pathPortrait")


class SpeakerMetaPathVariant(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    styles: list[StylePathVariant]
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
    policy: str | None = None
    license: str | None = None


class DownloadableStyle(V2BaseModel):
    style_name: str = Field(..., alias="styleName")
    style_id: int = Field(..., alias="styleId")
    version: str
    icon_base64: str = Field(..., alias="iconBase64")
    voice_sample_base64s: list[str] = Field(..., alias="voiceSampleBase64s")
    download_url: str = Field(..., alias="downloadUrl")


class DownloadableSpeaker(V2BaseModel):
    speaker_name: str = Field(..., alias="speakerName")
    speaker_uuid: str = Field(..., alias="speakerUuid")
    sub_speaker_uuids: list[str] = Field(..., alias="subSpeakerUuids")
    styles: list[DownloadableStyle]
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
    contents: list[str]


class WorldF0(V2BaseModel):
    f0: list[float]
    mora_durations: list[MoraDuration] = Field(..., alias="moraDurations")


class EngineInfo(V2BaseModel):
    device: str
    version: str


class Status(V2BaseModel):
    status: str


class ValidationError(V2BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class HTTPValidationError(V2BaseModel):
    detail: list[ValidationError] | None = None


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
