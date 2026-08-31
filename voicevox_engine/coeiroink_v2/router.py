"""公開COEIROINK v2 APIをHTTPへ接続するアダプター。

ルーターは`run.py`から独立させ、AudioManager互換オブジェクトと話者メタデータストアを外部から受け取る。
音声合成と波形処理はCoreの公開APIまたはこのパッケージのv2ヘルパーへ委譲する。
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from coeirocore.coeiro_manager import (
    AmbiguousStyleError as CoreAmbiguousStyleError,
)
from coeirocore.coeiro_manager import AudioManager, CoeiroCoreError
from coeirocore.coeiro_manager import (
    StyleNotFoundError as CoreStyleNotFoundError,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, Response

from voicevox_engine import __version__

from . import audio as audio_helpers
from .catalog import OfficialSiteCatalogClient
from .duration import DurationConversionError, convert_duration
from .metadata import (
    AmbiguousStyleError,
    MetadataAssetNotFoundError,
    MetadataError,
    SpeakerMetadataStore,
    SpeakerNotFoundError,
    StyleNotFoundError,
)
from .models import (
    AlgorithmSettings,
    AudioQuery,
    DictionaryWords,
    DownloadableModel,
    DownloadableSpeaker,
    EngineInfo,
    HTTPValidationError,
    Mora,
    Phrase,
    Prosody,
    ProsodyMakingParam,
    SpeakerFolderPath,
    SpeakerMeta,
    SpeakerMetaForTextBox,
    SpeakerMetaPathVariant,
    SpeakerPolicy,
    Status,
    SynthesisParam,
    TrimBufferSettings,
    UpdateInfo,
    WavMakingParam,
    WavProcessingParam,
    WavWithDuration,
    WorldF0,
)
from .prosody import (
    ProsodyError,
    clear_prosody_cache,
    estimate_prosody,
    estimate_prosody_from_kana,
)
from .td_psola import (
    TDPSOLAProcessingError,
    TDPSOLAValidationError,
)
from .wave_processing import (
    WaveProcessingOptions,
)
from .wave_processing import (
    as_waveform as _as_waveform,
)
from .wave_processing import (
    normalize_processing_algorithm as _normalize_processing_algorithm,
)
from .wave_processing import (
    process_wave as _process_wave,
)

CatalogCallback = Callable[[], Any]
DictionaryCallback = Callable[[DictionaryWords], Any]

# 想定済みの公開APIエラーだけをHTTPへ変換し、未知の例外はASGIまで伝播させてtracebackを残す。
HANDLED_API_ERRORS = (
    HTTPException,
    CoeiroCoreError,
    MetadataError,
    OSError,
    audio_helpers.AudioProcessingError,
    TDPSOLAProcessingError,
    TDPSOLAValidationError,
    TypeError,
    ValueError,
)


def _model_value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _as_http_error(error: Exception, default_status: int = 500) -> HTTPException:
    """既知の公開例外だけを安定したHTTPエラーへ変換し、未知の障害はこの関数へ渡さない。"""

    if isinstance(error, HTTPException):
        return error
    if isinstance(
        error,
        (
            SpeakerNotFoundError,
            StyleNotFoundError,
            AmbiguousStyleError,
            CoreStyleNotFoundError,
            CoreAmbiguousStyleError,
            FileNotFoundError,
        ),
    ):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(
        error,
        (
            audio_helpers.AudioValidationError,
            ProsodyError,
            DurationConversionError,
            ValueError,
            TypeError,
        ),
    ):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, MetadataError):
        return HTTPException(status_code=500, detail=str(error))
    return HTTPException(status_code=default_status, detail=str(error))


def _mora_phonemes(mora: Any) -> list[str]:
    value = _model_value(mora, "phoneme")
    if not isinstance(value, str) or not value:
        raise ProsodyError("prosodyDetail contains an invalid phoneme")
    result = value.split("-")
    if any(not phoneme for phoneme in result):
        raise ProsodyError("prosodyDetail contains an invalid phoneme")
    return [phoneme if phoneme == "N" else phoneme.lower() for phoneme in result]


def _detail_special_kind(phrase: Sequence[Any]) -> str | None:
    """公式v2形式の特殊プロソディ句を検証して分類する。"""

    moras = list(phrase)
    if not moras:
        raise ProsodyError("prosodyDetail contains an empty phrase")

    first_phoneme = _model_value(moras[0], "phoneme", default=None)
    if first_phoneme not in {"_", "?"}:
        for mora in moras:
            if any(phoneme in {"_", "?"} for phoneme in _mora_phonemes(mora)):
                raise ProsodyError("prosodyDetail contains an invalid special phrase")
        return None

    if len(moras) != 1:
        raise ProsodyError("a special prosody phrase must contain one mora")
    mora = moras[0]
    accent = _model_value(mora, "accent", default=None)
    hira = _model_value(mora, "hira", default=None)
    expected_hira = {"_": "、", "?": "？"}[first_phoneme]
    if (
        isinstance(accent, bool)
        or not isinstance(accent, int)
        or accent != 0
        or hira != expected_hira
    ):
        raise ProsodyError("prosodyDetail contains an invalid special phrase")
    return first_phoneme


def _split_prosody_detail(
    detail: Sequence[Sequence[Any]],
) -> tuple[list[list[Any]], list[bool], bool, bool]:
    """通常句と公式形式の休止・疑問句を分離する。"""

    try:
        phrases = [list(phrase) for phrase in detail]
    except TypeError as error:
        raise ProsodyError("prosodyDetail must be a sequence of phrases") from error

    ordinary: list[list[Any]] = []
    pause_moras: list[bool] = []
    pending_pause = False
    terminal_interrogative = False
    has_special_phrase = False

    for phrase_index, phrase in enumerate(phrases):
        special = _detail_special_kind(phrase)
        if special == "_":
            has_special_phrase = True
            if not ordinary or pending_pause or terminal_interrogative:
                raise ProsodyError("prosodyDetail contains an unexpected pause phrase")
            pending_pause = True
            continue
        if special == "?":
            has_special_phrase = True
            if (
                not ordinary
                or pending_pause
                or terminal_interrogative
                or phrase_index != len(phrases) - 1
            ):
                raise ProsodyError(
                    "the terminal question phrase must be the final prosody phrase"
                )
            terminal_interrogative = True
            continue
        if terminal_interrogative:
            raise ProsodyError(
                "prosodyDetail contains a phrase after the terminal question"
            )
        if pending_pause:
            pause_moras[-1] = True
            pending_pause = False
        ordinary.append(phrase)
        pause_moras.append(False)

    if pending_pause:
        raise ProsodyError("prosodyDetail contains a trailing pause phrase")
    return ordinary, pause_moras, terminal_interrogative, has_special_phrase


def _accent_markers(accent: int, mora_count: int) -> list[int]:
    if accent == 0:
        return [0] * mora_count
    if accent == 1:
        return [int(index == 0) for index in range(mora_count)]
    return [int(0 < index < accent) for index in range(mora_count)]


def _detail_accent(moras: Sequence[Any]) -> int:
    """v2の高低アクセント値からCoreのアクセント位置を復元する。"""

    accents = [_model_value(mora, "accent") for mora in moras]
    if not accents or any(
        isinstance(value, bool) or value not in (0, 1) for value in accents
    ):
        raise ProsodyError("prosodyDetail contains an invalid accent")
    candidates = [
        accent
        for accent in range(len(accents) + 1)
        if _accent_markers(accent, len(accents)) == accents
    ]
    if not candidates:
        raise ProsodyError("prosodyDetail contains an invalid accent pattern")
    return candidates[0]


def _detail_to_plain(
    detail: Sequence[Sequence[Any]],
    pause_moras: Sequence[bool] | None = None,
    terminal_interrogative: bool = False,
) -> list[str]:
    """v2のprosodyDetailからCore公開形式のトークン列を組み立てる。"""

    phrases, detail_pause_moras, detail_terminal, has_special = _split_prosody_detail(
        detail
    )
    if pause_moras is None:
        pause_moras = detail_pause_moras
        terminal_interrogative = detail_terminal
    elif has_special:
        raise ProsodyError(
            "pause_moras cannot be combined with special prosody phrases"
        )
    if pause_moras is not None and len(pause_moras) != len(phrases):
        raise ProsodyError("pause_moras and prosodyDetail must have the same length")
    tokens = ["^"]
    for phrase_index, phrase in enumerate(phrases):
        moras = list(phrase)
        accent = _detail_accent(moras)
        raised = False
        for mora_index, mora in enumerate(moras):
            tokens.extend(_mora_phonemes(mora))
            if accent == mora_index + 1 and mora_index + 1 != len(moras):
                tokens.append("]")
            if accent - 1 >= mora_index + 1 and not raised:
                tokens.append("[")
                raised = True
        if phrase_index + 1 != len(phrases):
            use_pause = (
                bool(pause_moras[phrase_index]) if pause_moras is not None else False
            )
            tokens.append("_" if use_pause else "#")
    tokens.append("?" if terminal_interrogative else "$")
    return tokens


def _mora_to_prosody(mora: Any, index: int, accent: int, count: int) -> Mora:
    if _model_value(mora, "phoneme", default=None) is not None:
        phonemes = _mora_phonemes(mora)
    else:
        consonant = _model_value(mora, "consonant", default=None)
        vowel = _model_value(mora, "vowel", default=None)
        if not isinstance(vowel, str) or not vowel:
            raise ProsodyError("AudioQuery contains an invalid vowel")
        vowel = vowel if vowel == "N" else vowel.lower()
        phonemes = ([consonant.lower()] if consonant else []) + [vowel]
    text = _model_value(mora, "text", "hira", default="")
    if not isinstance(text, str):
        text = str(text)
    hira = "".join(
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char for char in text
    )
    if not 0 <= accent <= count:
        raise ProsodyError("accent must be between 0 and the number of moras")
    if accent == 0:
        high = 0
    elif accent == 1:
        high = int(index == 0)
    else:
        high = int(0 < index < accent)
    return Mora(phoneme="-".join(phonemes), hira=hira, accent=high)


def _query_to_prosody(query: AudioQuery) -> Prosody:
    detail: list[list[Mora]] = []
    for phrase_index, phrase in enumerate(query.accent_phrases):
        moras = list(phrase.moras)
        accent = int(phrase.accent)
        detail.append(
            [
                _mora_to_prosody(mora, index, accent, len(moras))
                for index, mora in enumerate(moras)
            ]
        )
        if phrase_index + 1 != len(query.accent_phrases):
            if phrase.pause_mora is not None:
                detail.append([Mora(phoneme="_", hira="、", accent=0)])
        elif phrase.is_interrogative:
            detail.append([Mora(phoneme="?", hira="？", accent=0)])
    return Prosody(
        plain=_detail_to_plain(detail),
        detail=detail,
    )


def _hop_length(
    audio_manager: AudioManager,
    style_id: int,
    speaker_uuid: str,
) -> int:
    value = audio_manager.get_hop_length(
        style_id=style_id,
        speaker_uuid=speaker_uuid,
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("audio manager hop length must be a positive integer")
    return value


def _call_prediction(
    audio_manager: AudioManager,
    tokens: Sequence[str],
    speaker_uuid: str,
    style_id: int,
    speed_scale: float,
    with_duration: bool,
) -> tuple[np.ndarray, list[int]]:
    """公開Coreの通常推論または継続長付き推論を呼び分ける。"""

    arguments = {
        "text": list(tokens),
        "style_id": style_id,
        "speaker_uuid": speaker_uuid,
        "speed_scale": speed_scale,
    }
    if not with_duration:
        return _as_waveform(audio_manager.predict(**arguments)), []

    result = audio_manager.predict_with_duration(**arguments)
    wave = _as_waveform(result.wav)
    try:
        frames = [int(frame) for frame in result.duration_frames]
    except (TypeError, ValueError) as error:
        raise RuntimeError("audio manager returned invalid duration frames") from error
    if not frames:
        raise RuntimeError("audio manager did not return token durations")
    if any(frame < 0 for frame in frames):
        raise RuntimeError("audio manager returned a negative duration")
    return wave, frames


def _sampling_rate(audio_manager: AudioManager) -> int:
    value = audio_manager.fs
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("audio manager sampling rate must be a positive integer")
    return value


def _catalog_result(callback: CatalogCallback) -> Any:
    result = callback()
    return [] if result is None else result


def _public_dictionary_callback(payload: DictionaryWords) -> None:
    """辞書エンドポイントが使われた時だけ公開辞書アダプターを読み込む。"""

    from .dictionary import set_dictionary

    set_dictionary(payload)


def _first_speaker_uuid(store: SpeakerMetadataStore) -> str:
    if store.speaker_uuids:
        return store.speaker_uuids[0]
    raise SpeakerNotFoundError("<first>")


def _speaker_style(
    store: SpeakerMetadataStore,
    speaker_uuid: str | None,
    style_id: int | None,
) -> tuple[str, int]:
    """明示指定、スタイル検索、先頭話者の順にサンプル音声用の話者・スタイルを確定する。"""

    if speaker_uuid is not None and style_id is not None:
        store.get_style(speaker_uuid, style_id)
        return speaker_uuid, style_id

    if style_id is not None:
        found_uuid, style = store.find_style(style_id, speaker_uuid)
        return found_uuid, style.style_id

    resolved_uuid = speaker_uuid or _first_speaker_uuid(store)
    styles = store.speaker_meta(resolved_uuid).styles
    if not styles:
        raise StyleNotFoundError(resolved_uuid, -1)
    return resolved_uuid, styles[0].style_id


def _default_trim_values(
    value: TrimBufferSettings | Mapping[str, Any] | None,
) -> dict[str, float]:
    if value is None:
        # 公式v2合成と同じ短い端部バッファを既定値とし、4項目とも設定APIから実行時に変更できる。
        value = TrimBufferSettings(
            startTrimBuffer=0.05,
            endTrimBuffer=0.05,
            pauseStartTrimBuffer=0.05,
            pauseEndTrimBuffer=0.05,
        )
    elif not isinstance(value, TrimBufferSettings):
        value = TrimBufferSettings.model_validate(value)
    return {
        "start_trim_buffer": float(value.start_trim_buffer),
        "end_trim_buffer": float(value.end_trim_buffer),
        "pause_start_trim_buffer": float(value.pause_start_trim_buffer),
        "pause_end_trim_buffer": float(value.pause_end_trim_buffer),
    }


@dataclass(slots=True)
class _V2RouterContext:
    """v2ルート群が共有する、解決済みの依存オブジェクトと可変設定。"""

    audio_manager: AudioManager
    metadata_store: SpeakerMetadataStore | None
    catalog: OfficialSiteCatalogClient
    dictionary_callback: DictionaryCallback | None
    download_info_callback: CatalogCallback | None
    downloadable_speakers_callback: CatalogCallback | None
    update_info_callback: CatalogCallback | None
    engine_version: str
    settings: dict[str, Any]

    def metadata_required(self) -> SpeakerMetadataStore:
        if self.metadata_store is None:
            raise HTTPException(
                status_code=500, detail="speaker metadata is not configured"
            )
        return self.metadata_store

    @staticmethod
    def make_wave_response(wave: np.ndarray, sampling_rate: int) -> Response:
        return Response(
            content=audio_helpers.encode_pcm_wav(wave, sampling_rate),
            media_type="audio/wav",
        )

    @staticmethod
    def request_detail(
        param: WavMakingParam | SynthesisParam,
    ) -> tuple[list[str], list[list[Any]]]:
        # v2仕様では空のprosodyDetailが、textからサーバー側でプロソディを推定する指定になる。
        if param.prosody_detail:
            detail = [list(phrase) for phrase in param.prosody_detail]
            plain = _detail_to_plain(detail)
            ordinary, _, _, _ = _split_prosody_detail(detail)
            return plain, ordinary
        estimated = estimate_prosody(param.text)
        ordinary, _, _, _ = _split_prosody_detail(estimated.detail)
        return estimated.plain, ordinary

    def predict_request(
        self,
        param: WavMakingParam | SynthesisParam,
        with_duration: bool = False,
    ) -> tuple[np.ndarray, list[int], list[str], list[list[Any]], int]:
        plain, detail = self.request_detail(param)
        wave, frames = _call_prediction(
            self.audio_manager,
            plain,
            param.speaker_uuid,
            param.style_id,
            param.speed_scale,
            with_duration=with_duration,
        )
        return wave, frames, plain, detail, _sampling_rate(self.audio_manager)

    def processing_options(
        self,
        param: WavProcessingParam | SynthesisParam,
        mora_durations: Sequence[Any] | None,
    ) -> WaveProcessingOptions:
        """リクエスト値を優先し、未指定値だけをサーバー既定値で補う。"""

        start = (
            param.start_trim_buffer
            if param.start_trim_buffer is not None
            else self.settings["start_trim_buffer"]
        )
        end = (
            param.end_trim_buffer
            if param.end_trim_buffer is not None
            else self.settings["end_trim_buffer"]
        )
        pause_start = (
            param.pause_start_trim_buffer
            if param.pause_start_trim_buffer is not None
            else self.settings["pause_start_trim_buffer"]
        )
        pause_end = (
            param.pause_end_trim_buffer
            if param.pause_end_trim_buffer is not None
            else self.settings["pause_end_trim_buffer"]
        )
        return WaveProcessingOptions(
            volume_scale=param.volume_scale,
            pitch_scale=param.pitch_scale,
            intonation_scale=param.intonation_scale,
            pre_phoneme_length=param.pre_phoneme_length,
            post_phoneme_length=param.post_phoneme_length,
            output_sampling_rate=param.output_sampling_rate,
            start_trim_buffer=start,
            end_trim_buffer=end,
            processing_algorithm=(
                param.processing_algorithm or self.settings["processing_algorithm"]
            ),
            adjusted_f0=param.adjusted_f0,
            sampled_interval_value=param.sampled_interval_value,
            pause_length=param.pause_length,
            pause_start_trim_buffer=pause_start,
            pause_end_trim_buffer=pause_end,
            mora_durations=mora_durations,
        )

    def process_parameter(
        self,
        param: WavProcessingParam,
        wave: np.ndarray,
        sampling_rate: int,
    ) -> tuple[np.ndarray, int]:
        return _process_wave(
            self.audio_manager,
            wave,
            sampling_rate,
            self.processing_options(param, param.mora_durations),
        )


def _add_status_routes(router: APIRouter) -> None:
    @router.get(
        "/",
        response_model=Status,
        operation_id="read_root__get",
        tags=["その他"],
    )
    def read_root() -> Status:
        return Status(status="start")


def _add_speaker_list_routes(router: APIRouter, context: _V2RouterContext) -> None:
    metadata_required = context.metadata_required

    @router.get(
        "/v1/speakers",
        response_model=list[SpeakerMeta],
        operation_id="get_speakers_v1_speakers_get",
    )
    def get_speakers() -> list[SpeakerMeta]:
        store = metadata_required()
        try:
            return store.list_speakers()
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.get(
        "/v1/speakers_path_variant",
        response_model=list[SpeakerMetaPathVariant],
        operation_id="get_speakers_v1_speakers_path_variant_get",
    )
    def get_speakers_path_variant() -> list[SpeakerMetaPathVariant]:
        store = metadata_required()
        try:
            return store.list_speakers_path_variant()
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def _add_prosody_routes(router: APIRouter, context: _V2RouterContext) -> None:
    audio_manager = context.audio_manager

    @router.post(
        "/v1/estimate_prosody",
        response_model=Prosody,
        operation_id="estimate_prosody_v1_estimate_prosody_post",
    )
    def get_estimated_prosody(param: ProsodyMakingParam) -> Prosody:
        try:
            return estimate_prosody(param.text)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/estimate_prosody_from_kana",
        response_model=Phrase,
        operation_id="estimate_prosody_from_kana_v1_estimate_prosody_from_kana_post",
    )
    def get_estimated_prosody_from_kana(param: ProsodyMakingParam) -> Phrase:
        try:
            return Phrase(detail=estimate_prosody_from_kana(param.text).detail)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/estimate_f0",
        response_model=WorldF0,
        operation_id="estimate_f0_v1_estimate_f0_post",
    )
    def estimate_f0(param: WavWithDuration) -> WorldF0:
        try:
            wave, sampling_rate = audio_helpers.decode_pcm_wav_base64(param.wav_base64)
            f0, _, _ = audio_manager.get_world(wave.astype(np.float64), sampling_rate)
            f0_array = np.asarray(f0, dtype=np.float32).reshape(-1)
            if not np.isfinite(f0_array).all():
                raise audio_helpers.AudioProcessingError(
                    "WORLD F0 contains non-finite values"
                )
            return WorldF0(
                f0=[float(value) for value in f0_array],
                moraDurations=param.mora_durations,
            )
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def _add_prediction_routes(router: APIRouter, context: _V2RouterContext) -> None:
    audio_manager = context.audio_manager
    make_wave_response = context.make_wave_response
    predict_request = context.predict_request

    @router.post(
        "/v1/predict",
        response_class=Response,
        operation_id="predict_v1_predict_post",
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            422: {"model": HTTPValidationError},
        },
    )
    def predict(param: WavMakingParam) -> Response:
        try:
            wave, _, _, _, sampling_rate = predict_request(param, with_duration=False)
            return make_wave_response(wave, sampling_rate)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/predict_with_duration",
        response_model=WavWithDuration,
        operation_id="predict_v1_predict_with_duration_post",
    )
    def predict_with_duration(param: WavMakingParam) -> WavWithDuration:
        try:
            wave, frames, plain, detail, sampling_rate = predict_request(
                param, with_duration=True
            )
            durations = convert_duration(
                plain,
                detail,
                frames,
                _hop_length(audio_manager, param.style_id, param.speaker_uuid),
            )
            return WavWithDuration(
                wavBase64=audio_helpers.encode_pcm_wav_base64(wave, sampling_rate),
                moraDurations=durations,
            )
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def _add_processing_routes(router: APIRouter, context: _V2RouterContext) -> None:
    audio_manager = context.audio_manager
    make_wave_response = context.make_wave_response
    predict_request = context.predict_request
    processing_options = context.processing_options
    process_parameter = context.process_parameter

    @router.post(
        "/v1/process",
        response_class=Response,
        operation_id="process_v1_process_post",
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            422: {"model": HTTPValidationError},
        },
    )
    def process(param: WavProcessingParam) -> Response:
        try:
            wave, sampling_rate = audio_helpers.decode_pcm_wav_base64(param.wav_base64)
            output, output_sampling_rate = process_parameter(param, wave, sampling_rate)
            return make_wave_response(output, output_sampling_rate)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/process_with_pitch",
        response_class=Response,
        operation_id="process_with_pitch_v1_process_with_pitch_post",
        responses={
            200: {"description": "Successful Response"},
            307: {"description": "Temporary Redirect to /v1/process"},
        },
    )
    def process_with_pitch() -> Response:
        return RedirectResponse(url="/v1/process", status_code=307)

    @router.post(
        "/v1/synthesis",
        response_class=Response,
        operation_id="synthesis_v1_synthesis_post",
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            422: {"model": HTTPValidationError},
        },
    )
    def synthesis(param: SynthesisParam) -> Response:
        """生波形を推論し、休止長変更に必要な場合だけ継続長を取得してから後処理する。"""

        try:
            needs_duration = param.pause_length is not None
            wave, frames, plain, detail, sampling_rate = predict_request(
                param, with_duration=needs_duration
            )
            mora_durations = (
                convert_duration(
                    plain,
                    detail,
                    frames,
                    _hop_length(audio_manager, param.style_id, param.speaker_uuid),
                )
                if needs_duration
                else None
            )
            output, output_sampling_rate = _process_wave(
                audio_manager,
                wave,
                sampling_rate,
                processing_options(param, mora_durations),
            )
            return make_wave_response(output, output_sampling_rate)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def _add_settings_routes(router: APIRouter, context: _V2RouterContext) -> None:
    dictionary_callback = context.dictionary_callback
    settings = context.settings

    @router.post(
        "/v1/set_dictionary",
        response_model=None,
        operation_id="set_dictionary_v1_set_dictionary_post",
        responses={
            200: {"content": {"application/json": {"schema": {}}}},
            422: {"model": HTTPValidationError},
        },
    )
    def set_dictionary(words: DictionaryWords) -> dict[str, Any]:
        try:
            callback = dictionary_callback or _public_dictionary_callback
            callback(words)
            clear_prosody_cache()
            return {}
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/set_default_processing_algorithm",
        response_model=None,
        operation_id="set_default_processing_algorithm_v1_set_default_processing_algorithm_post",
        responses={
            200: {"content": {"application/json": {"schema": {}}}},
            422: {"model": HTTPValidationError},
        },
    )
    def set_default_processing_algorithm(param: AlgorithmSettings) -> None:
        try:
            settings["processing_algorithm"] = _normalize_processing_algorithm(
                param.processing_algorithm
            )
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/set_default_trim_buffer",
        response_model=None,
        operation_id="set_default_trim_buffer_v1_set_default_trim_buffer_post",
        responses={
            200: {"content": {"application/json": {"schema": {}}}},
            422: {"model": HTTPValidationError},
        },
    )
    def set_default_trim_buffer(param: TrimBufferSettings) -> None:
        settings.update(
            start_trim_buffer=param.start_trim_buffer,
            end_trim_buffer=param.end_trim_buffer,
            pause_start_trim_buffer=param.pause_start_trim_buffer,
            pause_end_trim_buffer=param.pause_end_trim_buffer,
        )


def _add_catalog_routes(router: APIRouter, context: _V2RouterContext) -> None:
    catalog = context.catalog
    download_info_callback = context.download_info_callback
    downloadable_speakers_callback = context.downloadable_speakers_callback
    update_info_callback = context.update_info_callback
    engine_version = context.engine_version

    @router.get(
        "/v1/download_info",
        response_model=list[DownloadableModel],
        operation_id="get_download_info_v1_download_info_get",
    )
    def download_info() -> list[DownloadableModel]:
        callback = download_info_callback or catalog.get_download_info
        result = _catalog_result(callback)
        return list(result)

    @router.get(
        "/v1/downloadable_speakers",
        response_model=list[DownloadableSpeaker],
        operation_id="get_downloadable_speakers_v1_downloadable_speakers_get",
    )
    def downloadable_speakers() -> list[DownloadableSpeaker]:
        callback = downloadable_speakers_callback or catalog.get_downloadable_speakers
        result = _catalog_result(callback)
        return list(result)

    @router.get(
        "/v1/update_info",
        response_model=list[UpdateInfo],
        operation_id="get_update_info_v1_update_info_get",
    )
    def update_info() -> list[UpdateInfo]:
        callback = update_info_callback or catalog.get_update_info
        result = _catalog_result(callback)
        return list(result)

    @router.get(
        "/v1/engine_info",
        response_model=EngineInfo,
        operation_id="get_engine_info_v1_engine_info_get",
    )
    def engine_info() -> EngineInfo:
        return EngineInfo(
            device=context.audio_manager.device,
            version=engine_version,
        )


def _add_metadata_lookup_routes(router: APIRouter, context: _V2RouterContext) -> None:
    metadata_required = context.metadata_required

    @router.get(
        "/v1/speaker_folder_path",
        response_model=SpeakerFolderPath,
        operation_id="get_speaker_folder_path_v1_speaker_folder_path_get",
    )
    def speaker_folder_path(
        speaker_uuid: str | None = Query(None, alias="speakerUuid"),
    ) -> SpeakerFolderPath:
        if speaker_uuid is None:
            return SpeakerFolderPath(speakerFolderPath="None")
        try:
            store = metadata_required()
            path = store.speaker_path(speaker_uuid)
            return SpeakerFolderPath(speakerFolderPath=str(path))
        except SpeakerNotFoundError:
            return SpeakerFolderPath(speakerFolderPath="None")
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/query2prosody",
        response_model=Prosody,
        operation_id="speaker_folder_path_v1_query2prosody_post",
    )
    def query2prosody(query: AudioQuery) -> Prosody:
        try:
            return _query_to_prosody(query)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.post(
        "/v1/style_id_to_speaker_meta",
        response_model=SpeakerMetaForTextBox,
        operation_id="speaker_folder_path_v1_style_id_to_speaker_meta_post",
    )
    def style_id_to_speaker_meta(
        style_id: int | None = Query(None, alias="styleId"),
    ) -> SpeakerMetaForTextBox:
        # 未指定・未導入スタイルを`None`文字列で返すのは、公式v2 APIの既存レスポンス契約に合わせた挙動。
        if style_id is None:
            return SpeakerMetaForTextBox(
                speakerUuid="None",
                styleId=0,
                speakerName="None",
                styleName="None",
            )
        try:
            store = metadata_required()
            return store.style_id_to_speaker_meta(style_id)
        except (StyleNotFoundError, AmbiguousStyleError):
            return SpeakerMetaForTextBox(
                speakerUuid="None",
                styleId=0,
                speakerName="None",
                styleName="None",
            )
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def _add_speaker_asset_routes(router: APIRouter, context: _V2RouterContext) -> None:
    metadata_required = context.metadata_required

    @router.get(
        "/v1/sample_voice",
        response_class=Response,
        operation_id="get_sample_voice_v1_sample_voice_get",
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            422: {"model": HTTPValidationError},
        },
    )
    def sample_voice(
        speaker_uuid: str | None = Query(None, alias="speakerUuid"),
        style_id: int | None = Query(None, alias="styleId"),
        index: int | None = Query(None),
    ) -> Response:
        try:
            store = metadata_required()
            resolved_uuid, resolved_style = _speaker_style(
                store, speaker_uuid, style_id
            )
            resolved_index = 0 if index is None else index
            content = store.read_sample_voice(
                resolved_uuid, resolved_style, resolved_index
            )
            return Response(content=content, media_type="audio/wav")
        except MetadataAssetNotFoundError as error:
            raise HTTPException(
                status_code=400, detail="Sample voice file not found"
            ) from error
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error

    @router.get(
        "/v1/speaker_policy",
        response_model=SpeakerPolicy,
        operation_id="get_speaker_policy_v1_speaker_policy_get",
    )
    def speaker_policy(
        speaker_uuid: str | None = Query(None, alias="speakerUuid"),
    ) -> SpeakerPolicy:
        try:
            store = metadata_required()
            resolved_uuid = speaker_uuid or _first_speaker_uuid(store)
            return store.speaker_policy(resolved_uuid)
        except HANDLED_API_ERRORS as error:
            raise _as_http_error(error) from error


def create_v2_router(
    audio_manager: AudioManager,
    metadata_store: SpeakerMetadataStore | str | Path | None = None,
    *,
    speaker_info_dir: str | Path | None = None,
    catalog: OfficialSiteCatalogClient | None = None,
    dictionary_callback: DictionaryCallback | None = None,
    download_info_callback: CatalogCallback | None = None,
    downloadable_speakers_callback: CatalogCallback | None = None,
    update_info_callback: CatalogCallback | None = None,
    engine_version: str = __version__,
    default_processing_algorithm: str = "td-psola",
    default_trim_buffer: TrimBufferSettings | Mapping[str, Any] | None = None,
) -> APIRouter:
    """ルートパスに依存しないCOEIROINK v2ルーターを構築する。

    audio_managerとmetadata_storeは公開実装または同じAPIを持つテストダブルを受け付ける。
    外部カタログのコールバックは任意で、未指定時の一覧APIは空配列を返す。
    """

    if audio_manager is None:
        raise ValueError("audio_manager is required")

    if metadata_store is None and speaker_info_dir is not None:
        metadata_store = SpeakerMetadataStore(Path(speaker_info_dir))
    elif isinstance(metadata_store, (str, Path)):
        metadata_store = SpeakerMetadataStore(Path(metadata_store))

    if catalog is None:
        catalog = OfficialSiteCatalogClient()

    router = APIRouter()
    settings = {
        "processing_algorithm": _normalize_processing_algorithm(
            default_processing_algorithm
        ),
        **_default_trim_values(default_trim_buffer),
    }

    context = _V2RouterContext(
        audio_manager=audio_manager,
        metadata_store=metadata_store,
        catalog=catalog,
        dictionary_callback=dictionary_callback,
        download_info_callback=download_info_callback,
        downloadable_speakers_callback=downloadable_speakers_callback,
        update_info_callback=update_info_callback,
        engine_version=engine_version,
        settings=settings,
    )
    # 各ルート群には必要な依存だけを共有コンテキストから渡す。
    _add_status_routes(router)
    _add_speaker_list_routes(router, context)
    _add_prosody_routes(router, context)
    _add_prediction_routes(router, context)
    _add_processing_routes(router, context)
    _add_settings_routes(router, context)
    _add_catalog_routes(router, context)
    _add_metadata_lookup_routes(router, context)
    _add_speaker_asset_routes(router, context)

    return router


__all__ = ["create_v2_router"]
