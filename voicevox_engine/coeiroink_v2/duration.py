"""ESPnetのトークン長をv2のモーラ長応答へ変換します。
v2の韻律応答はモーラ単位ですが、ESPnetは``plain``の各トークンに長さを返すため、両者の対応を決定的に整理します。
範囲はサンプル単位なので、音響フレーム長に``hop_length``を掛けます。
v2応答は最終的なモーラ範囲だけを公開し、トークンごとの長さは公開しません。
そのため、非音素トークンのフレーム割当は次の規則で固定します。
``^``は先頭の``pau``、``$``と``?``は末尾の``pau``、``_``は途中の``pau``になります。
``[``, ``]``, ``#``など幅のない境界・アクセント記号は次の音素または休止へ加算し、最後の音素の後なら末尾休止に含めます。
すべての入力フレームを消費し、範囲を連続させたうえで、差異が生じ得る箇所をこの割当規則に限定します。
"""

from dataclasses import dataclass
from operator import index as to_index
from typing import Iterable, List, Mapping, Sequence, Tuple

from .models import MoraDuration, PhonemeDuration, TimeRange


class DurationConversionError(ValueError):
    """トークン・韻律・長さの列が一致しない場合に発生します。"""


# API概念に合わせた短い別名であり、別の例外型ではありません。
DurationError = DurationConversionError


_ZERO_WIDTH_MARKERS = frozenset(("[", "]", "#"))
_TERMINAL_MARKERS = frozenset(("$", "?"))


@dataclass(frozen=True)
class _MoraSpec:
    name: str
    hira: str
    phonemes: Tuple[str, ...]


def _as_list(value: Iterable, field_name: str) -> List:
    if isinstance(value, (str, bytes)):
        raise DurationConversionError("{} must be a sequence".format(field_name))
    try:
        return list(value)
    except TypeError as error:
        raise DurationConversionError(
            "{} must be an iterable".format(field_name)
        ) from error


def _integer(value: object, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise DurationConversionError("{} must be an integer".format(field_name))
    try:
        result = to_index(value)
    except TypeError as error:
        raise DurationConversionError(
            "{} must be an integer".format(field_name)
        ) from error
    if result < minimum:
        raise DurationConversionError(
            "{} must be greater than or equal to {}".format(field_name, minimum)
        )
    return result


def _field(value: object, name: str, context: str) -> object:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as error:
            raise DurationConversionError(
                "{} is missing {}".format(context, name)
            ) from error
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise DurationConversionError(
            "{} has no {} field".format(context, name)
        ) from error


def _mora_specs(prosody_detail: Iterable[Iterable[object]]) -> List[_MoraSpec]:
    specs: List[_MoraSpec] = []
    phrases = _as_list(prosody_detail, "prosody_detail")
    for phrase_index, phrase in enumerate(phrases):
        moras = _as_list(phrase, "prosody_detail[{}]".format(phrase_index))
        for mora_index, mora in enumerate(moras):
            context = "prosody_detail[{}][{}]".format(phrase_index, mora_index)
            name = _field(mora, "phoneme", context)
            hira = _field(mora, "hira", context)
            if not isinstance(name, str) or not name:
                raise DurationConversionError(
                    "{}.phoneme must be a non-empty string".format(context)
                )
            if not isinstance(hira, str):
                raise DurationConversionError(
                    "{}.hira must be a string".format(context)
                )
            phonemes = tuple(name.split("-"))
            if any(not phoneme for phoneme in phonemes):
                raise DurationConversionError(
                    "{}.phoneme contains an empty phoneme".format(context)
                )
            specs.append(_MoraSpec(name=name, hira=hira, phonemes=phonemes))
    return specs


def _pause_duration(
    cursor: int, frame_count: int, hop_length: int
) -> Tuple[MoraDuration, int]:
    end = cursor + frame_count * hop_length
    wav_range = TimeRange(start=cursor, end=end)
    return (
        MoraDuration(
            mora="pau",
            hira="",
            phonemePitches=[
                PhonemeDuration(phoneme="pau", wavRange=wav_range)
            ],
            wavRange=wav_range,
        ),
        end,
    )


def convert_duration(
    plain_tokens: Sequence[str],
    prosody_detail: Sequence[Sequence[object]],
    duration_frames: Sequence[int],
    hop_length: int,
) -> List[MoraDuration]:
    """v2のplainトークンとESPnetの長さをモーラ範囲へ変換します。
    ``plain_tokens``は境界記号を含むv2の平坦なトークン列です。
    ``prosody_detail``は句・モーラ順の入れ子になったv2の``Mora``列です。
    ``duration_frames``はplain各トークンに対応する非負のESPnetフレーム長です。
    ``hop_length``は音響1フレームが表す出力サンプル数です。
    返却する範囲は音響順で、0から始まり連続し、``sum(duration_frames) * hop_length``で終わります。
    不正または不整合な列の場合は``DurationConversionError``を発生させます。
    """

    tokens = _as_list(plain_tokens, "plain_tokens")
    frames = [
        _integer(value, "duration_frames[{}]".format(i))
        for i, value in enumerate(_as_list(duration_frames, "duration_frames"))
    ]
    hop_length = _integer(hop_length, "hop_length", minimum=1)
    if len(tokens) != len(frames):
        raise DurationConversionError(
            "plain_tokens and duration_frames must have the same length"
        )
    for i, token in enumerate(tokens):
        if not isinstance(token, str) or not token:
            raise DurationConversionError(
                "plain_tokens[{}] must be a non-empty string".format(i)
            )

    specs = _mora_specs(prosody_detail)
    result: List[MoraDuration] = []
    cursor = 0
    pending_frames = 0
    spec_index = 0
    phoneme_index = 0
    current_mora_start = None
    current_phonemes: List[PhonemeDuration] = []
    saw_leading_pause = False
    saw_terminal_pause = False

    for token_index, (token, frame_count) in enumerate(zip(tokens, frames)):
        if token == "^":
            if token_index != 0 or saw_leading_pause:
                raise DurationConversionError(
                    "^ must occur exactly at the beginning of plain_tokens"
                )
            if pending_frames:
                raise DurationConversionError(
                    "a boundary marker cannot precede the leading pause"
                )
            pause, cursor = _pause_duration(cursor, frame_count, hop_length)
            result.append(pause)
            saw_leading_pause = True
            continue

        if token in _TERMINAL_MARKERS:
            if token_index != len(tokens) - 1 or saw_terminal_pause:
                raise DurationConversionError(
                    "{} must occur exactly at the end of plain_tokens".format(token)
                )
            if spec_index != len(specs) or phoneme_index:
                raise DurationConversionError(
                    "terminal pause occurred before prosody_detail was consumed"
                )
            pause, cursor = _pause_duration(
                cursor, frame_count + pending_frames, hop_length
            )
            result.append(pause)
            pending_frames = 0
            saw_terminal_pause = True
            continue

        if token == "_":
            if phoneme_index:
                raise DurationConversionError(
                    "pause marker split a mora"
                )
            pause, cursor = _pause_duration(
                cursor, frame_count + pending_frames, hop_length
            )
            result.append(pause)
            pending_frames = 0
            continue

        if token in _ZERO_WIDTH_MARKERS:
            pending_frames += frame_count
            continue

        if saw_terminal_pause:
            raise DurationConversionError(
                "phoneme token occurred after the terminal pause"
            )
        if spec_index >= len(specs):
            raise DurationConversionError(
                "plain_tokens contains more phonemes than prosody_detail"
            )
        spec = specs[spec_index]
        expected = spec.phonemes[phoneme_index]
        if token != expected:
            raise DurationConversionError(
                "plain token {} at index {} does not match prosody phoneme {}".format(
                    token, token_index, expected
                )
            )

        if phoneme_index == 0:
            current_mora_start = cursor
            current_phonemes = []
        start = cursor
        end = start + (frame_count + pending_frames) * hop_length
        current_phonemes.append(
            PhonemeDuration(
                phoneme=token,
                wavRange=TimeRange(start=start, end=end),
            )
        )
        cursor = end
        pending_frames = 0
        phoneme_index += 1

        if phoneme_index == len(spec.phonemes):
            if current_mora_start is None:
                raise DurationConversionError("internal mora range state is invalid")
            result.append(
                MoraDuration(
                    mora=spec.name,
                    hira=spec.hira,
                    phonemePitches=current_phonemes,
                    wavRange=TimeRange(start=current_mora_start, end=cursor),
                )
            )
            spec_index += 1
            phoneme_index = 0
            current_mora_start = None
            current_phonemes = []

    if phoneme_index:
        raise DurationConversionError("prosody_detail ended in the middle of a mora")
    if spec_index != len(specs):
        raise DurationConversionError(
            "prosody_detail contains phonemes absent from plain_tokens"
        )
    if pending_frames:
        raise DurationConversionError(
            "plain_tokens ended with an unallocated boundary-marker duration"
        )
    expected_end = sum(frames) * hop_length
    if cursor != expected_end:
        raise DurationConversionError(
            "duration conversion did not consume all acoustic frames"
        )
    return result


__all__ = ["DurationConversionError", "DurationError", "convert_duration"]
