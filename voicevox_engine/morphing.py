from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import chain

import numpy as np
from coeirocore.pyworld_compat import load_pyworld

from .metas.metas import Speaker, SpeakerStyle, SpeakerSupportPermittedSynthesisMorphing
from .metas.metas_store import construct_lookup
from .model import AudioQuery, MorphableTargetInfo, SpeakerNotFoundError
from .synthesis_engine.synthesis_engine_base import SynthesisEngineBase


# FIXME: ndarrayの型ヒントをpyworldの定義に合わせる。
# https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/blob/2b64f86197573497c685c785c6e0e743f407b63e/pyworld/pyworld.pyx#L398
@dataclass(frozen=True)
class MorphingParameter:
    fs: float
    frame_period: float
    base_f0: np.ndarray
    base_aperiodicity: np.ndarray
    base_spectrogram: np.ndarray
    target_spectrogram: np.ndarray


def create_morphing_parameter(
    base_wave: np.ndarray,
    target_wave: np.ndarray,
    fs: float,
) -> MorphingParameter:
    pw = load_pyworld()
    frame_period = 1.0
    base_f0, base_time_axis = pw.harvest(base_wave, fs, frame_period=frame_period)
    base_spectrogram = pw.cheaptrick(base_wave, base_f0, base_time_axis, fs)
    base_aperiodicity = pw.d4c(base_wave, base_f0, base_time_axis, fs)

    target_f0, morph_time_axis = pw.harvest(target_wave, fs, frame_period=frame_period)
    target_spectrogram = pw.cheaptrick(target_wave, target_f0, morph_time_axis, fs)
    target_spectrogram.resize(base_spectrogram.shape)

    return MorphingParameter(
        fs=fs,
        frame_period=frame_period,
        base_f0=base_f0,
        base_aperiodicity=base_aperiodicity,
        base_spectrogram=base_spectrogram,
        target_spectrogram=target_spectrogram,
    )


def get_morphable_targets(
    speakers: list[Speaker],
    base_speakers: list[int],
) -> list[dict[int, MorphableTargetInfo]]:
    """
    ベースの話者ごとにエンジン内の各スタイルとモーフィング可能かを判定する。

    Parameters
    ----------
    speakers : list[Speaker]
        全話者の情報
    base_speakers : list[int]
        モーフィング可能か判定したいベースの話者のスタイルIDのリスト

    Returns
    -------
    list[dict[int, MorphableTargetInfo]]
        ベースの話者ごとのスタイルIDとモーフィング可否の対応表
    """
    speaker_lookup = construct_lookup(speakers)

    morphable_targets_arr = []
    for base_speaker in base_speakers:
        morphable_targets = {}
        for style in chain.from_iterable(speaker.styles for speaker in speakers):
            morphable_targets[style.id] = MorphableTargetInfo(
                is_morphable=is_synthesis_morphing_permitted(
                    speaker_lookup=speaker_lookup,
                    base_speaker=base_speaker,
                    target_speaker=style.id,
                )
            )
        morphable_targets_arr.append(morphable_targets)

    return morphable_targets_arr


# 指定されたspeakerがモーフィング可能かどうかを返す。
# speakerが見つからない場合はSpeakerNotFoundErrorを送出する。
def is_synthesis_morphing_permitted(
    speaker_lookup: dict[int, tuple[Speaker, SpeakerStyle]],
    base_speaker: int,
    target_speaker: int,
) -> bool:
    base_speaker_data = speaker_lookup.get(base_speaker)
    target_speaker_data = speaker_lookup.get(target_speaker)

    if base_speaker_data is None or target_speaker_data is None:
        raise SpeakerNotFoundError(
            base_speaker if base_speaker_data is None else target_speaker
        )

    base_speaker_info, _ = base_speaker_data
    target_speaker_info, _ = target_speaker_data

    base_speaker_uuid = base_speaker_info.speaker_uuid
    target_speaker_uuid = target_speaker_info.speaker_uuid

    base_speaker_morphing_info: SpeakerSupportPermittedSynthesisMorphing = (
        base_speaker_info.supported_features.permitted_synthesis_morphing
    )

    target_speaker_morphing_info: SpeakerSupportPermittedSynthesisMorphing = (
        target_speaker_info.supported_features.permitted_synthesis_morphing
    )

    if SpeakerSupportPermittedSynthesisMorphing.NOTHING in (
        base_speaker_morphing_info,
        target_speaker_morphing_info,
    ):
        return False
    if SpeakerSupportPermittedSynthesisMorphing.SELF_ONLY in (
        base_speaker_morphing_info,
        target_speaker_morphing_info,
    ):
        return base_speaker_uuid == target_speaker_uuid
    return (
        base_speaker_morphing_info == SpeakerSupportPermittedSynthesisMorphing.ALL
        and target_speaker_morphing_info == SpeakerSupportPermittedSynthesisMorphing.ALL
    )


# 両話者のモノラル合成結果をWORLDで解析し、モーフィング用パラメータを作る。
#
# 入力クエリは複製するため変更されない。WORLD解析のため、`outputStereo`の指定にかかわらず内部合成はモノラルで行う。
def synthesis_morphing_parameter(
    engine: SynthesisEngineBase,
    query: AudioQuery,
    base_speaker: int,
    target_speaker: int,
    enable_interrogative_upspeak: bool = False,
) -> MorphingParameter:
    query = deepcopy(query)

    # WORLDで解析するため合成はモノラルで行う。
    query.outputStereo = False

    base_wave = engine.synthesis(
        query=query,
        speaker_id=base_speaker,
        enable_interrogative_upspeak=enable_interrogative_upspeak,
    ).astype("float")
    target_wave = engine.synthesis(
        query=query,
        speaker_id=target_speaker,
        enable_interrogative_upspeak=enable_interrogative_upspeak,
    ).astype("float")

    return create_morphing_parameter(
        base_wave=base_wave,
        target_wave=target_wave,
        fs=query.outputSamplingRate,
    )


def synthesis_morphing(
    morph_param: MorphingParameter,
    morph_rate: float,
    output_stereo: bool = False,
) -> np.ndarray:
    """
    指定した割合でパラメータをもとにモーフィングした音声を生成する。

    Parameters
    ----------
    morph_param : MorphingParameter
        `synthesis_morphing_parameter`または`create_morphing_parameter`で作成したパラメータ
    morph_rate : float
        モーフィングの割合
        0.0でベースの話者、1.0でターゲットの話者に近づく。
    output_stereo : bool
        同じ波形を2チャンネルに複製してステレオで返すか

    Returns
    -------
    generated : np.ndarray
        モーフィングした音声

    Raises
    ------
    ValueError
        morph_rateが0.0から1.0の範囲外の場合
    """
    if morph_rate < 0.0 or morph_rate > 1.0:
        raise ValueError("morph_rateは0.0から1.0の範囲で指定してください")

    pw = load_pyworld()
    morph_spectrogram = (
        morph_param.base_spectrogram * (1.0 - morph_rate)
        + morph_param.target_spectrogram * morph_rate
    )

    y_h = pw.synthesize(
        morph_param.base_f0,
        morph_spectrogram,
        morph_param.base_aperiodicity,
        morph_param.fs,
        morph_param.frame_period,
    )

    if output_stereo:
        y_h = np.array([y_h, y_h]).T

    return y_h
