"""公開COEIROINK v2サーバー機能のHTTPアダプターです。
ルーターを``run.py``から独立させ、アプリケーションがAudioManager相当のオブジェクトと話者メタデータストアを渡す構成にします。
そのためニューラルモデルを読み込まずにテストでき、合成と音声処理は公開オブジェクトまたはv2ヘルパーへ委譲します。
"""

import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from coeirocore.coeiro_manager import (
    AmbiguousStyleError as CoreAmbiguousStyleError,
)
from coeirocore.coeiro_manager import (
    StyleNotFoundError as CoreStyleNotFoundError,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, Response

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
from .td_psola import process_td_psola

CatalogCallback = Callable[[], Any]
DictionaryCallback = Callable[[DictionaryWords], Any]


def _call_with_supported_kwargs(function: Callable[..., Any], **kwargs: Any) -> Any:
    """小さなテスト用代替オブジェクトも受け付けてマネージャーのメソッドを呼び出します。
    実際のCoreは``speaker_uuid``を受け付けますが、簡易スタブは省略することがあるため、検査できるシグネチャの引数だけを渡します。
    """

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return function(**kwargs)

    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**{name: value for name, value in kwargs.items() if name in accepted})


def _attribute_or_call(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """最初に一致した属性を取得し、呼び出し可能なら実行して返します。"""

    for name in names:
        if not hasattr(value, name):
            continue
        candidate = getattr(value, name)
        return candidate() if callable(candidate) else candidate
    return default


def _metadata_value(store: Any, names: Sequence[str], default: Any = None) -> Any:
    return _attribute_or_call(store, names, default=default)


def _raw_attribute(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """引数が必要なメソッド向けに、属性を実行せず取得します。"""

    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _model_value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _as_http_error(error: Exception, default_status: int = 500) -> HTTPException:
    """公開アダプターの例外をトレースバックなしの安定したHTTPエラーへ変換します。"""

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


def _as_waveform(value: Any) -> np.ndarray:
    wave = np.asarray(value, dtype=np.float32)
    if wave.ndim != 1:
        wave = wave.reshape(-1)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise audio_helpers.AudioValidationError("audio manager returned an invalid waveform")
    return wave


def _prediction_parts(result: Any) -> Tuple[np.ndarray, List[int]]:
    """公開CoreのPredictionResultと同等の小さな代替結果を受け付けます。"""

    if isinstance(result, Mapping):
        wave = result.get("wav")
        frames = result.get("duration_frames", result.get("durationFrames"))
    elif isinstance(result, np.ndarray):
        wave, frames = result, []
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        wave, frames = result
    else:
        wave = _model_value(result, "wav", "waveform")
        frames = _model_value(result, "duration_frames", "durationFrames")

    if wave is None:
        raise RuntimeError("audio manager did not return a waveform")
    if frames is None:
        return _as_waveform(wave), []
    try:
        duration_frames = [int(frame) for frame in frames]
    except (TypeError, ValueError) as error:
        raise RuntimeError("audio manager returned invalid duration frames") from error
    if any(frame < 0 for frame in duration_frames):
        raise RuntimeError("audio manager returned a negative duration")
    return _as_waveform(wave), duration_frames


def _mora_phonemes(mora: Any) -> List[str]:
    value = _model_value(mora, "phoneme")
    if not isinstance(value, str) or not value:
        raise ProsodyError("prosodyDetail contains an invalid phoneme")
    result = value.split("-")
    if any(not phoneme for phoneme in result):
        raise ProsodyError("prosodyDetail contains an invalid phoneme")
    return [phoneme if phoneme == "N" else phoneme.lower() for phoneme in result]


def _detail_accent(moras: Sequence[Any]) -> int:
    """v2のモーラ高低記号からCoreのアクセント位置を復元します。"""

    accents = [_model_value(mora, "accent") for mora in moras]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in accents):
        raise ProsodyError("prosodyDetail contains an invalid accent")
    if not accents or all(value == 0 for value in accents):
        return 0
    if accents[0] == 1:
        return 1
    try:
        return accents.index(1) + 1
    except ValueError:
        return 0


def _detail_to_plain(
    detail: Sequence[Sequence[Any]],
    pause_moras: Optional[Sequence[bool]] = None,
    terminal_interrogative: bool = False,
) -> List[str]:
    """v2の``prosodyDetail``から公開Coreのトークン列を構築します。"""

    phrases = list(detail)
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
            use_pause = bool(pause_moras[phrase_index]) if pause_moras is not None else False
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
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char
        for char in text
    )
    if not 0 <= accent <= count:
        raise ProsodyError("accent must be between 0 and the number of moras")
    if accent == 0:
        high = int(index > 0)
    elif accent == 1:
        high = int(index == 0)
    else:
        high = int(0 < index < accent)
    return Mora(phoneme="-".join(phonemes), hira=hira, accent=high)


def _query_to_prosody(query: AudioQuery) -> Prosody:
    detail: List[List[Mora]] = []
    pause_moras: List[bool] = []
    for phrase in query.accent_phrases:
        moras = list(phrase.moras)
        accent = int(phrase.accent)
        detail.append(
            [
                _mora_to_prosody(mora, index, accent, len(moras))
                for index, mora in enumerate(moras)
            ]
        )
        pause_moras.append(phrase.pause_mora is not None)
    return Prosody(
        plain=_detail_to_plain(
            detail,
            pause_moras=pause_moras,
            terminal_interrogative=(
                bool(query.accent_phrases[-1].is_interrogative)
                if query.accent_phrases
                else False
            ),
        ),
        detail=detail,
    )


def _hop_length(audio_manager: Any, style_id: int, speaker_uuid: str) -> int:
    value = _attribute_or_call(audio_manager, ("hop_length", "frame_shift", "samples_per_frame"))
    if value is None:
        getter = getattr(audio_manager, "get_hop_length", None)
        if getter is not None:
            value = _call_with_supported_kwargs(
                getter, style_id=style_id, speaker_uuid=speaker_uuid
            )
    if value is None:
        value = 512
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("audio manager hop length must be a positive integer")
    return value


def _call_prediction(
    audio_manager: Any,
    tokens: Sequence[str],
    speaker_uuid: str,
    style_id: int,
    speed_scale: float,
    with_duration: bool,
) -> Tuple[np.ndarray, List[int]]:
    method_names = (
        ("predict_with_duration", "predict_with_durations")
        if with_duration
        else ("predict", "predict_with_duration", "predict_with_durations")
    )
    method = next(
        (getattr(audio_manager, name) for name in method_names if hasattr(audio_manager, name)),
        None,
    )
    if method is None:
        raise RuntimeError("audio manager does not provide prediction")
    result = _call_with_supported_kwargs(
        method,
        text=list(tokens),
        style_id=style_id,
        speaker_uuid=speaker_uuid,
        speed_scale=speed_scale,
    )
    wave, frames = _prediction_parts(result)
    if with_duration and not frames:
        raise RuntimeError("audio manager did not return token durations")
    return wave, frames


def _sampling_rate(audio_manager: Any) -> int:
    value = _attribute_or_call(audio_manager, ("fs", "sampling_rate", "default_sampling_rate"), 44100)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("audio manager sampling rate must be a positive integer")
    return value


def _manager_audio_method(audio_manager: Any, name: str) -> Optional[Callable[..., Any]]:
    method = getattr(audio_manager, name, None)
    return method if callable(method) else None


def _validated_f0(value: Sequence[float]) -> np.ndarray:
    try:
        f0 = np.asarray(value, dtype=np.float64)
    except Exception as error:
        raise audio_helpers.AudioValidationError("adjusted_f0 must be numeric") from error
    if f0.ndim != 1 or f0.size == 0 or not np.isfinite(f0).all():
        raise audio_helpers.AudioValidationError(
            "adjusted_f0 must be a non-empty finite one-dimensional array"
        )
    if np.any(f0 < 0.0):
        raise audio_helpers.AudioValidationError("adjusted_f0 must not be negative")
    return f0


def _manager_world_f0(
    audio_manager: Any, wave: np.ndarray, sampling_rate: int
) -> Optional[np.ndarray]:
    get_world = _manager_audio_method(audio_manager, "get_world")
    if get_world is None:
        return None
    result = get_world(wave.astype(np.float64), sampling_rate)
    if not isinstance(result, (tuple, list)) or not result:
        raise audio_helpers.AudioProcessingError(
            "audio manager returned an invalid WORLD result"
        )
    return _validated_f0(result[0])


def _fit_f0_track(track: np.ndarray, size: int) -> np.ndarray:
    if track.size == size:
        return track.astype(np.float64, copy=True)
    positions = np.linspace(0.0, 1.0, track.size)
    target_positions = np.linspace(0.0, 1.0, size)
    return np.interp(target_positions, positions, track).astype(np.float64)


def _world_process(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    pitch_scale: float,
    intonation_scale: float,
    adjusted_f0: Optional[Sequence[float]],
) -> np.ndarray:
    """任意の明示F0トラックを使い、CoreのWORLD処理を実行します。"""

    if adjusted_f0 is None:
        pitch = _manager_audio_method(audio_manager, "pitch_intonation")
        if pitch is not None:
            return _as_waveform(
                pitch(wave, sampling_rate, pitch_scale, intonation_scale)
            )

    get_world = _manager_audio_method(audio_manager, "get_world")
    if get_world is None:
        raise audio_helpers.AudioProcessingError(
            "WORLD processing with adjustedF0 requires get_world"
        )
    result = get_world(wave.astype(np.float64), sampling_rate)
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise audio_helpers.AudioProcessingError(
            "audio manager returned an invalid WORLD result"
        )
    base_f0, spectral_envelope, aperiodicity = result[:3]
    f0 = _validated_f0(adjusted_f0) if adjusted_f0 is not None else _validated_f0(base_f0)
    f0 = _fit_f0_track(f0, len(np.asarray(base_f0).reshape(-1)))
    if pitch_scale != 0.0:
        f0 *= 2.0 ** float(pitch_scale)
    if intonation_scale != 1.0:
        # WORLDは無声音フレームをF0=0で表すため、有声フレームだけを調整して休止や無声子音を有音化しません。
        voiced = f0 > 0.0
        if np.any(voiced):
            voiced_f0 = f0[voiced]
            mean = float(np.mean(voiced_f0))
            f0[voiced] = (
                (voiced_f0 - mean) * float(intonation_scale) + mean
            )

    try:
        from coeirocore.pyworld_compat import load_pyworld

        pw = load_pyworld()

        processed = pw.synthesize(
            f0.astype(np.float64),
            spectral_envelope,
            aperiodicity,
            sampling_rate,
        )
    except Exception as error:
        raise audio_helpers.AudioProcessingError(
            "failed to synthesize WORLD-processed audio"
        ) from error
    return _as_waveform(processed)


def _processing_number(
    value: object,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise audio_helpers.AudioValidationError(
            "{} must be a finite number".format(name)
        )
    result = float(value)
    if minimum is not None and result < minimum:
        raise audio_helpers.AudioValidationError(
            "{} must be greater than or equal to {}".format(name, minimum)
        )
    if maximum is not None and result > maximum:
        raise audio_helpers.AudioValidationError(
            "{} must be no greater than {}".format(name, maximum)
        )
    return result


def _processing_duration(value: object, name: str) -> float:
    return _processing_number(
        value,
        name,
        minimum=0.0,
        maximum=audio_helpers.MAX_PAUSE_LENGTH_SECONDS,
    )


def _processing_sampling_rate(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        or int(value) > audio_helpers.MAX_SAMPLING_RATE
    ):
        raise audio_helpers.AudioValidationError(
            "{} must be a positive integer no greater than {}".format(
                name, audio_helpers.MAX_SAMPLING_RATE
            )
        )
    return int(value)


def _require_processing_size(size: int, name: str = "processed wave") -> None:
    if size <= 0 or size > audio_helpers.MAX_GENERATED_WAVE_SAMPLES:
        raise audio_helpers.AudioValidationError(
            "{} would exceed the maximum of {} samples".format(
                name, audio_helpers.MAX_GENERATED_WAVE_SAMPLES
            )
        )


def _process_wave(
    audio_manager: Any,
    wave: np.ndarray,
    sampling_rate: int,
    *,
    volume_scale: float,
    pitch_scale: float,
    intonation_scale: float,
    pre_phoneme_length: float,
    post_phoneme_length: float,
    output_sampling_rate: int,
    start_trim_buffer: float,
    end_trim_buffer: float,
    processing_algorithm: Optional[str] = None,
    adjusted_f0: Optional[Sequence[float]] = None,
    sampled_interval_value: Optional[int] = None,
    pause_length: Optional[float] = None,
    pause_start_trim_buffer: Optional[float] = None,
    pause_end_trim_buffer: Optional[float] = None,
    mora_durations: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, int]:
    """指定されたマネージャーの公開プリミティブでv2処理を実行します。"""

    current = _as_waveform(wave)
    _require_processing_size(current.size, "input wave")
    sampling_rate = _processing_sampling_rate(sampling_rate, "sampling_rate")
    output_sampling_rate = _processing_sampling_rate(
        output_sampling_rate, "output_sampling_rate"
    )
    volume_scale = _processing_number(
        volume_scale, "volume_scale", minimum=0.0
    )
    pitch_scale = _processing_number(
        pitch_scale, "pitch_scale", minimum=-32.0, maximum=32.0
    )
    intonation_scale = _processing_number(
        intonation_scale, "intonation_scale", minimum=0.0
    )
    pre_phoneme_length = _processing_duration(
        pre_phoneme_length, "pre_phoneme_length"
    )
    post_phoneme_length = _processing_duration(
        post_phoneme_length, "post_phoneme_length"
    )
    start_trim_buffer = _processing_duration(
        start_trim_buffer, "start_trim_buffer"
    )
    end_trim_buffer = _processing_duration(end_trim_buffer, "end_trim_buffer")
    pause_start_trim_buffer = _processing_duration(
        pause_start_trim_buffer if pause_start_trim_buffer is not None else 0.0,
        "pause_start_trim_buffer",
    )
    pause_end_trim_buffer = _processing_duration(
        pause_end_trim_buffer if pause_end_trim_buffer is not None else 0.0,
        "pause_end_trim_buffer",
    )
    if pause_length is not None:
        pause_length = _processing_duration(pause_length, "pause_length")

    # 空の``adjustedF0``リストは呼び出し元のF0トラックがないことを表します。
    if adjusted_f0 is not None and len(adjusted_f0) == 0:
        adjusted_f0 = None
    target_f0 = (
        _validated_f0(adjusted_f0) if adjusted_f0 is not None else None
    )

    algorithm = (processing_algorithm or "td-psola").lower()
    if algorithm not in ("td-psola", "world", "resampling"):
        algorithm = "td-psola"

    # F0トラックはトリミング前のモデル波形に対応するため、休止長変更や端の無音トリミングより先にピッチ処理を行います。
    if pitch_scale != 0.0 or intonation_scale != 1.0 or target_f0 is not None:
        if algorithm == "world":
            current = _world_process(
                audio_manager,
                current,
                sampling_rate,
                pitch_scale,
                intonation_scale,
                target_f0,
            )
        elif algorithm == "resampling":
            # 固定レートのピッチシフトでは時間変化するF0トラックを表せません。
            # その成分は決定的なピッチマークで処理し、選択されたリサンプリングは全体のピッチ差分だけに適用します。
            if target_f0 is not None or intonation_scale != 1.0:
                source_f0 = _manager_world_f0(
                    audio_manager, current, sampling_rate
                )
                current = _as_waveform(
                    process_td_psola(
                        current,
                        sampling_rate,
                        pitch_scale=0.0,
                        intonation_scale=(
                            1.0 if target_f0 is not None else intonation_scale
                        ),
                        f0=source_f0,
                        target_f0=target_f0,
                    )
                )
            current = audio_helpers.pitch_shift_resampling(
                current,
                sampling_rate,
                pitch_scale,
            )
        else:
            source_f0 = _manager_world_f0(
                audio_manager, current, sampling_rate
            )
            current = _as_waveform(
                process_td_psola(
                    current,
                    sampling_rate,
                    pitch_scale=(0.0 if target_f0 is not None else pitch_scale),
                    intonation_scale=(
                        1.0 if target_f0 is not None else intonation_scale
                    ),
                    f0=source_f0,
                    target_f0=target_f0,
                )
            )

    if pause_length is not None and mora_durations:
        current = audio_helpers.replace_pause_segments(
            current,
            sampling_rate,
            mora_durations,
            pause_length=pause_length,
            pause_start_trim_buffer=pause_start_trim_buffer,
            pause_end_trim_buffer=pause_end_trim_buffer,
        )
    if start_trim_buffer or end_trim_buffer:
        current = audio_helpers.trim_wave(
            current,
            sampling_rate,
            start_trim_buffer=start_trim_buffer,
            end_trim_buffer=end_trim_buffer,
        )
    else:
        trim = _manager_audio_method(audio_manager, "trim")
        if trim is not None:
            current = _as_waveform(trim(current))

    volume = _manager_audio_method(audio_manager, "volume")
    if volume_scale != 1.0:
        current = _as_waveform(
            volume(current, volume_scale)
            if volume is not None
            else current * volume_scale
        )

    silence = _manager_audio_method(audio_manager, "sil")
    if pre_phoneme_length != 0.0 or post_phoneme_length != 0.0:
        projected_size = (
            current.size
            + int(sampling_rate * pre_phoneme_length)
            + int(sampling_rate * post_phoneme_length)
        )
        _require_processing_size(projected_size)
        if silence is not None:
            current = _as_waveform(
                silence(
                    current,
                    sampling_rate,
                    pre_phoneme_length,
                    post_phoneme_length,
                )
            )
        else:
            pre = np.zeros(int(sampling_rate * pre_phoneme_length), dtype=np.float32)
            post = np.zeros(int(sampling_rate * post_phoneme_length), dtype=np.float32)
            current = np.concatenate((pre, current, post)).astype(np.float32)

    if output_sampling_rate != sampling_rate:
        projected_size = int(
            math.ceil(current.size * output_sampling_rate / sampling_rate)
        )
        _require_processing_size(projected_size, "resampled wave")
        resampling = _manager_audio_method(audio_manager, "resampling")
        if resampling is not None:
            current = _as_waveform(
                resampling(current, sampling_rate, output_sampling_rate)
            )
        else:
            try:
                import resampy

                current = _as_waveform(
                    resampy.resample(
                        current,
                        sampling_rate,
                        output_sampling_rate,
                        filter="kaiser_fast",
                    )
                )
            except Exception as error:
                raise audio_helpers.AudioProcessingError(
                    "failed to resample waveform"
                ) from error

    current = _as_waveform(current)
    _require_processing_size(current.size)

    # 旧形式のサンプリング間隔は通信互換性のため受け付けますが、公開処理側が安全な解析間隔を決めます。
    del sampled_interval_value
    return current, output_sampling_rate


def _catalog_result(
    catalog: Any,
    explicit_callback: Optional[CatalogCallback],
    names: Sequence[str],
    default: Any,
) -> Any:
    callback = explicit_callback
    if callback is None and catalog is not None:
        if isinstance(catalog, Mapping):
            for name in names:
                if name in catalog:
                    callback = catalog[name]
                    break
        else:
            for name in names:
                if hasattr(catalog, name):
                    callback = getattr(catalog, name)
                    break
    if callback is None:
        return default
    result = callback() if callable(callback) else callback
    return default if result is None else result


def _public_dictionary_callback(payload: DictionaryWords) -> None:
    """エンドポイントが使われた時だけ公開辞書アダプターを読み込みます。"""

    from .dictionary import set_dictionary

    set_dictionary(payload)


def _first_speaker_uuid(store: Any) -> str:
    uuids = _metadata_value(store, ("speaker_uuids",), default=None)
    if uuids:
        return sorted(str(value) for value in uuids)[0]
    speakers = _metadata_value(store, ("list_speakers", "speakers"), default=[])
    if speakers:
        uuid = _model_value(speakers[0], "speaker_uuid", "speakerUuid")
        if uuid:
            return str(uuid)
    raise SpeakerNotFoundError("<first>")


def _speaker_style(store: Any, speaker_uuid: Optional[str], style_id: Optional[int]) -> Tuple[str, int]:
    if speaker_uuid is not None and style_id is not None:
        get_style = getattr(store, "get_style", None)
        if callable(get_style):
            get_style(speaker_uuid, style_id)
        return speaker_uuid, style_id

    if style_id is not None:
        find_style = getattr(store, "find_style", None)
        if callable(find_style):
            found_uuid, style = find_style(style_id, speaker_uuid)
            return str(found_uuid), int(_model_value(style, "style_id", "styleId"))
        meta_for_style = getattr(store, "speaker_meta_for_style", None)
        if callable(meta_for_style):
            meta = meta_for_style(style_id, speaker_uuid)
            return str(_model_value(meta, "speaker_uuid", "speakerUuid")), int(
                _model_value(meta, "style_id", "styleId")
            )
        raise StyleNotFoundError(speaker_uuid or "<unspecified>", style_id)

    resolved_uuid = speaker_uuid or _first_speaker_uuid(store)
    meta_getter = _raw_attribute(store, ("speaker_meta",), default=None)
    meta = meta_getter(resolved_uuid) if callable(meta_getter) else None
    if meta is None:
        get_speakers = getattr(store, "list_speakers", None)
        records = get_speakers() if callable(get_speakers) else []
        meta = next(
            (
                record
                for record in records
                if _model_value(record, "speaker_uuid", "speakerUuid") == resolved_uuid
            ),
            None,
        )
    styles = _model_value(meta, "styles", default=[]) if meta is not None else []
    if not styles:
        raise StyleNotFoundError(resolved_uuid, -1)
    return resolved_uuid, int(_model_value(styles[0], "style_id", "styleId", "id"))


def _default_trim_values(value: Optional[Union[TrimBufferSettings, Mapping[str, Any]]]) -> Dict[str, float]:
    if value is None:
        # v2合成の既定値は端に短いバッファを残し、設定エンドポイントで4値を実行時に置き換えられます。
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


def create_v2_router(
    audio_manager: Any,
    metadata_store: Any = None,
    *,
    speaker_info_dir: Optional[Union[str, Path]] = None,
    catalog: Any = None,
    dictionary_callback: Optional[DictionaryCallback] = None,
    download_info_callback: Optional[CatalogCallback] = None,
    downloadable_speakers_callback: Optional[CatalogCallback] = None,
    update_info_callback: Optional[CatalogCallback] = None,
    engine_version: str = "2.13.0",
    device: str = "cpu",
    default_processing_algorithm: str = "td-psola",
    default_trim_buffer: Optional[Union[TrimBufferSettings, Mapping[str, Any]]] = None,
    **compatibility_options: Any,
) -> APIRouter:
    """ルートに依存しないCOEIROINK v2 APIルーターを作成します。
    ``audio_manager``は公開Coreの``AudioManager``またはpredict系メソッドと任意の音声処理を持つテスト用オブジェクトを受け付けます。
    ``metadata_store``は``SpeakerMetadataStore``または互換オブジェクトで、パスを渡した場合は公開メタデータヘルパーで読み込みます。
    カタログコールバックは任意で、未指定なら空リストを返します。
    ``dictionary_callback``には解析済みの``DictionaryWords``を渡し、アプリケーションの辞書を更新できます。
    """

    if audio_manager is None:
        raise ValueError("audio_manager is required")

    if metadata_store is None:
        metadata_store = compatibility_options.pop("metas_store", None)
    if metadata_store is None:
        metadata_store = compatibility_options.pop("speaker_metadata_store", None)
    if metadata_store is None and speaker_info_dir is not None:
        metadata_store = SpeakerMetadataStore(Path(speaker_info_dir))
    elif isinstance(metadata_store, (str, Path)):
        metadata_store = SpeakerMetadataStore(Path(metadata_store))

    if catalog is None:
        catalog = OfficialSiteCatalogClient()

    # 短いコールバック名も互換性のため受け付けますが、HTTP契約には追加しません。
    download_info_callback = download_info_callback or compatibility_options.pop(
        "download_info", None
    )
    downloadable_speakers_callback = downloadable_speakers_callback or compatibility_options.pop(
        "downloadable_speakers", None
    )
    update_info_callback = update_info_callback or compatibility_options.pop(
        "update_info", None
    )
    if compatibility_options:
        unknown = ", ".join(sorted(compatibility_options))
        raise TypeError("unknown create_v2_router option(s): {}".format(unknown))

    router = APIRouter()
    settings = {
        "processing_algorithm": str(default_processing_algorithm),
        **_default_trim_values(default_trim_buffer),
    }

    def metadata_required() -> Any:
        if metadata_store is None:
            raise HTTPException(status_code=500, detail="speaker metadata is not configured")
        return metadata_store

    def make_wave_response(wave: np.ndarray, sampling_rate: int) -> Response:
        return Response(
            content=audio_helpers.encode_pcm_wav(wave, sampling_rate),
            media_type="audio/wav",
        )

    def request_detail(param: Union[WavMakingParam, SynthesisParam]) -> Tuple[List[str], List[List[Any]]]:
        # v2の仕様では、呼び出し元が``text``からの韻律推定を求める場合に空リストを送ります。
        if param.prosody_detail:
            detail = [list(phrase) for phrase in param.prosody_detail]
            return _detail_to_plain(detail), detail
        estimated = estimate_prosody(param.text)
        return estimated.plain, estimated.detail

    def predict_request(
        param: Union[WavMakingParam, SynthesisParam],
        with_duration: bool = False,
    ) -> Tuple[np.ndarray, List[int], List[str], List[List[Any]], int]:
        plain, detail = request_detail(param)
        wave, frames = _call_prediction(
            audio_manager,
            plain,
            param.speaker_uuid,
            param.style_id,
            param.speed_scale,
            with_duration=with_duration,
        )
        return wave, frames, plain, detail, _sampling_rate(audio_manager)

    @router.get(
        "/",
        response_model=Status,
        operation_id="read_root__get",
        tags=["その他"],
    )
    def read_root() -> Status:
        return Status(status="start")

    @router.get(
        "/v1/speakers",
        response_model=List[SpeakerMeta],
        operation_id="get_speakers_v1_speakers_get",
    )
    def get_speakers() -> List[SpeakerMeta]:
        store = metadata_required()
        try:
            result = _metadata_value(store, ("list_speakers", "speakers"), default=[])
            return list(result)
        except Exception as error:
            raise _as_http_error(error)

    @router.get(
        "/v1/speakers_path_variant",
        response_model=List[SpeakerMetaPathVariant],
        operation_id="get_speakers_v1_speakers_path_variant_get",
    )
    def get_speakers_path_variant() -> List[SpeakerMetaPathVariant]:
        store = metadata_required()
        try:
            result = _metadata_value(
                store,
                ("list_speakers_path_variant", "speakers_path_variant"),
                default=[],
            )
            return list(result)
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/estimate_prosody",
        response_model=Prosody,
        operation_id="estimate_prosody_v1_estimate_prosody_post",
    )
    def get_estimated_prosody(param: ProsodyMakingParam) -> Prosody:
        try:
            return estimate_prosody(param.text)
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/estimate_prosody_from_kana",
        response_model=Phrase,
        operation_id="estimate_prosody_from_kana_v1_estimate_prosody_from_kana_post",
    )
    def get_estimated_prosody_from_kana(param: ProsodyMakingParam) -> Phrase:
        try:
            return Phrase(detail=estimate_prosody_from_kana(param.text).detail)
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/estimate_f0",
        response_model=WorldF0,
        operation_id="estimate_f0_v1_estimate_f0_post",
    )
    def estimate_f0(param: WavWithDuration) -> WorldF0:
        try:
            wave, sampling_rate = audio_helpers.decode_pcm_wav_base64(param.wav_base64)
            get_world = _manager_audio_method(audio_manager, "get_world")
            if get_world is None:
                return audio_helpers.prepare_world_f0(
                    wave, sampling_rate, param.mora_durations
                )
            f0, _, _ = get_world(wave.astype(np.float64), sampling_rate)
            f0_array = np.asarray(f0, dtype=np.float32).reshape(-1)
            if not np.isfinite(f0_array).all():
                raise audio_helpers.AudioProcessingError("WORLD F0 contains non-finite values")
            return WorldF0(
                f0=[float(value) for value in f0_array],
                moraDurations=param.mora_durations,
            )
        except Exception as error:
            raise _as_http_error(error)

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
        except Exception as error:
            raise _as_http_error(error)

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
        except Exception as error:
            raise _as_http_error(error)

    def process_parameter(
        param: WavProcessingParam,
        wave: np.ndarray,
        sampling_rate: int,
    ) -> Tuple[np.ndarray, int]:
        start = (
            param.start_trim_buffer
            if param.start_trim_buffer is not None
            else settings["start_trim_buffer"]
        )
        end = (
            param.end_trim_buffer
            if param.end_trim_buffer is not None
            else settings["end_trim_buffer"]
        )
        algorithm = param.processing_algorithm or settings["processing_algorithm"]
        pause_start = (
            param.pause_start_trim_buffer
            if param.pause_start_trim_buffer is not None
            else settings["pause_start_trim_buffer"]
        )
        pause_end = (
            param.pause_end_trim_buffer
            if param.pause_end_trim_buffer is not None
            else settings["pause_end_trim_buffer"]
        )
        return _process_wave(
            audio_manager,
            wave,
            sampling_rate,
            volume_scale=param.volume_scale,
            pitch_scale=param.pitch_scale,
            intonation_scale=param.intonation_scale,
            pre_phoneme_length=param.pre_phoneme_length,
            post_phoneme_length=param.post_phoneme_length,
            output_sampling_rate=param.output_sampling_rate,
            start_trim_buffer=start,
            end_trim_buffer=end,
            processing_algorithm=algorithm,
            adjusted_f0=param.adjusted_f0,
            sampled_interval_value=param.sampled_interval_value,
            pause_length=param.pause_length,
            pause_start_trim_buffer=pause_start,
            pause_end_trim_buffer=pause_end,
            mora_durations=param.mora_durations,
        )

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
        except Exception as error:
            raise _as_http_error(error)

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
        # 専用処理は持たず、同じ入力契約を検証するprocessへ307で委譲します。
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
        try:
            # pause_lengthがある場合だけdurationを求め、不要なフレーム変換を省きます。
            needs_duration = param.pause_length is not None
            wave, frames, plain, detail, sampling_rate = predict_request(
                param, with_duration=needs_duration
            )
            mora_durations = (
                convert_duration(
                    plain,
                    detail,
                    frames,
                    _hop_length(
                        audio_manager, param.style_id, param.speaker_uuid
                    ),
                )
                if needs_duration
                else None
            )
            start = (
                param.start_trim_buffer
                if param.start_trim_buffer is not None
                else settings["start_trim_buffer"]
            )
            end = (
                param.end_trim_buffer
                if param.end_trim_buffer is not None
                else settings["end_trim_buffer"]
            )
            pause_start = (
                param.pause_start_trim_buffer
                if param.pause_start_trim_buffer is not None
                else settings["pause_start_trim_buffer"]
            )
            pause_end = (
                param.pause_end_trim_buffer
                if param.pause_end_trim_buffer is not None
                else settings["pause_end_trim_buffer"]
            )
            output, output_sampling_rate = _process_wave(
                audio_manager,
                wave,
                sampling_rate,
                volume_scale=param.volume_scale,
                pitch_scale=param.pitch_scale,
                intonation_scale=param.intonation_scale,
                pre_phoneme_length=param.pre_phoneme_length,
                post_phoneme_length=param.post_phoneme_length,
                output_sampling_rate=param.output_sampling_rate,
                start_trim_buffer=start,
                end_trim_buffer=end,
                processing_algorithm=param.processing_algorithm
                or settings["processing_algorithm"],
                adjusted_f0=param.adjusted_f0,
                sampled_interval_value=param.sampled_interval_value,
                pause_length=param.pause_length,
                pause_start_trim_buffer=pause_start,
                pause_end_trim_buffer=pause_end,
                mora_durations=mora_durations,
            )
            return make_wave_response(output, output_sampling_rate)
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/set_dictionary",
        response_model=None,
        operation_id="set_dictionary_v1_set_dictionary_post",
        responses={
            200: {"content": {"application/json": {"schema": {}}}},
            422: {"model": HTTPValidationError},
        },
    )
    def set_dictionary(words: DictionaryWords) -> Dict[str, Any]:
        try:
            callback = (
                dictionary_callback
                or _manager_audio_method(audio_manager, "set_dictionary")
                or _public_dictionary_callback
            )
            if callback is not None:
                _call_with_supported_kwargs(
                    callback,
                    payload=words,
                    words=words,
                    dictionary_words=words,
                )
            # 辞書更新前の解析結果を再利用しないよう、韻律キャッシュを破棄します。
            clear_prosody_cache()
            return {}
        except Exception as error:
            raise _as_http_error(error)

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
        settings["processing_algorithm"] = param.processing_algorithm
        return None

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
        return None

    @router.get(
        "/v1/download_info",
        response_model=List[DownloadableModel],
        operation_id="get_download_info_v1_download_info_get",
    )
    def download_info() -> List[DownloadableModel]:
        # カタログ未提供時はHTTPエラーではなく空配列を返します。
        result = _catalog_result(
            catalog,
            download_info_callback,
            ("download_info", "download_infos", "get_download_info"),
            [],
        )
        return list(result)

    @router.get(
        "/v1/downloadable_speakers",
        response_model=List[DownloadableSpeaker],
        operation_id="get_downloadable_speakers_v1_downloadable_speakers_get",
    )
    def downloadable_speakers() -> List[DownloadableSpeaker]:
        # ダウンロード機能を注入しない構成でもAPI形状を維持します。
        result = _catalog_result(
            catalog,
            downloadable_speakers_callback,
            ("downloadable_speakers", "get_downloadable_speakers"),
            [],
        )
        return list(result)

    @router.get(
        "/v1/speaker_folder_path",
        response_model=SpeakerFolderPath,
        operation_id="get_speaker_folder_path_v1_speaker_folder_path_get",
    )
    def speaker_folder_path(
        speaker_uuid: Optional[str] = Query(None, alias="speakerUuid"),
    ) -> SpeakerFolderPath:
        # 旧APIの未解決値は404ではなく文字列"None"で返します。
        if speaker_uuid is None:
            return SpeakerFolderPath(speakerFolderPath="None")
        try:
            store = metadata_required()
            path_getter = _raw_attribute(
                store, ("speaker_path", "lookup_speaker_folder"), default=None
            )
            path = (
                path_getter(speaker_uuid)
                if callable(path_getter)
                else path_getter
            )
            if path is None:
                raise SpeakerNotFoundError(speaker_uuid)
            return SpeakerFolderPath(speakerFolderPath=str(path))
        except SpeakerNotFoundError:
            return SpeakerFolderPath(speakerFolderPath="None")
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/query2prosody",
        response_model=Prosody,
        operation_id="speaker_folder_path_v1_query2prosody_post",
    )
    def query2prosody(query: AudioQuery) -> Prosody:
        try:
            return _query_to_prosody(query)
        except Exception as error:
            raise _as_http_error(error)

    @router.post(
        "/v1/style_id_to_speaker_meta",
        response_model=SpeakerMetaForTextBox,
        operation_id="speaker_folder_path_v1_style_id_to_speaker_meta_post",
    )
    def style_id_to_speaker_meta(
        style_id: Optional[int] = Query(None, alias="styleId"),
    ) -> SpeakerMetaForTextBox:
        # 表示用の旧APIは未指定・未解決のスタイルを互換値で返します。
        if style_id is None:
            return SpeakerMetaForTextBox(
                speakerUuid="None",
                styleId=0,
                speakerName="None",
                styleName="None",
            )
        try:
            store = metadata_required()
            result_getter = _raw_attribute(
                store,
                ("style_id_to_speaker_meta", "speaker_meta_for_style"),
                default=None,
            )
            result = result_getter(style_id) if callable(result_getter) else result_getter
            if result is None:
                raise StyleNotFoundError("<unspecified>", style_id)
            return result
        except (StyleNotFoundError, AmbiguousStyleError):
            return SpeakerMetaForTextBox(
                speakerUuid="None",
                styleId=0,
                speakerName="None",
                styleName="None",
            )
        except Exception as error:
            raise _as_http_error(error)

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
        speaker_uuid: Optional[str] = Query(None, alias="speakerUuid"),
        style_id: Optional[int] = Query(None, alias="styleId"),
        index: Optional[int] = Query(None),
    ) -> Response:
        try:
            store = metadata_required()
            resolved_uuid, resolved_style = _speaker_style(store, speaker_uuid, style_id)
            resolved_index = 0 if index is None else index
            read_sample = getattr(store, "read_sample_voice", None)
            if callable(read_sample):
                content = read_sample(resolved_uuid, resolved_style, resolved_index)
            else:
                path = store.sample_voice_path(
                    resolved_uuid, resolved_style, resolved_index
                )
                content = Path(path).read_bytes()
            return Response(content=content, media_type="audio/wav")
        except MetadataAssetNotFoundError:
            raise HTTPException(
                status_code=400, detail="Sample voice file not found"
            )
        except Exception as error:
            raise _as_http_error(error)

    @router.get(
        "/v1/speaker_policy",
        response_model=SpeakerPolicy,
        operation_id="get_speaker_policy_v1_speaker_policy_get",
    )
    def speaker_policy(
        speaker_uuid: Optional[str] = Query(None, alias="speakerUuid"),
    ) -> SpeakerPolicy:
        try:
            store = metadata_required()
            resolved_uuid = speaker_uuid or _first_speaker_uuid(store)
            result_getter = _raw_attribute(
                store, ("speaker_policy", "read_policy_license"), default=None
            )
            result = (
                result_getter(resolved_uuid)
                if callable(result_getter)
                else result_getter
            )
            if result is None:
                raise MetadataError("speaker policy is not configured")
            return result
        except Exception as error:
            raise _as_http_error(error)

    @router.get(
        "/v1/update_info",
        response_model=List[UpdateInfo],
        operation_id="get_update_info_v1_update_info_get",
    )
    def update_info() -> List[UpdateInfo]:
        result = _catalog_result(
            catalog,
            update_info_callback,
            ("update_info", "get_update_info"),
            [],
        )
        return list(result)

    @router.get(
        "/v1/engine_info",
        response_model=EngineInfo,
        operation_id="get_engine_info_v1_engine_info_get",
    )
    def engine_info() -> EngineInfo:
        return EngineInfo(device=device, version=engine_version)

    return router


__all__ = ["create_v2_router"]
