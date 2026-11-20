# ESPnetのトークン継続長をv2のモーラ継続長の応答へ変換する処理
#
# v2のプロソディの応答はモーラだけを表すが、ESPnetは`plain`の各トークンに1つずつ継続長を返す。
# このモジュールはこの2つの表現の間の対応付けを決定的に行う。
# 範囲はサンプル数で表すので、音響フレーム単位の継続長に`hop_length`を掛ける。
#
# 保存してある公式の応答には最終的なモーラの範囲があるが、トークンごとの生の継続長はない。
# そのため、音素でないトークンのフレームがどこに割り当てられるかは公式の応答から確定できない。
# ここでは次の方針を明示的に採る。
#
# - `^`は先頭の`pau`モーラ、`$`と`?`は末尾の`pau`モーラになる。
# - `_`は文中の`pau`モーラになる。
# - `[`、`]`、`#`などの幅を持たない境界やアクセントの記号は、次の音素または休止に加える。
#   そのため、最後の音素より後の記号は末尾の休止に含まれる。
#
# この方針により、入力のフレームをすべて使い切り、すべての範囲を連続させ、
# 公式実装との一致が不確かな部分をこの記号の割り当て方針だけに限定する。

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import index as to_index

from .models import MoraDuration, PhonemeDuration, TimeRange


# トークン列・プロソディ・継続長が対応しない場合のエラー
class DurationConversionError(ValueError):
    pass


# APIで使う名前に合わせた別名で、例外の種類は元のまま変えない。
DurationError = DurationConversionError

_ZERO_WIDTH_MARKERS = frozenset(("[", "]", "#"))
_TERMINAL_MARKERS = frozenset(("$", "?"))


@dataclass(frozen=True)
class _MoraSpec:
    name: str
    hira: str
    phonemes: tuple[str, ...]


def _as_list(value: Iterable, field_name: str) -> list:
    if isinstance(value, (str, bytes)):
        raise DurationConversionError(f"{field_name} must be a sequence")
    try:
        return list(value)
    except TypeError as error:
        raise DurationConversionError(f"{field_name} must be an iterable") from error


def _integer(value: object, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise DurationConversionError(f"{field_name} must be an integer")
    try:
        result = to_index(value)
    except TypeError as error:
        raise DurationConversionError(f"{field_name} must be an integer") from error
    if result < minimum:
        raise DurationConversionError(
            f"{field_name} must be greater than or equal to {minimum}"
        )
    return result


def _field(value: object, name: str, context: str) -> object:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as error:
            raise DurationConversionError(f"{context} is missing {name}") from error
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise DurationConversionError(f"{context} has no {name} field") from error


def _mora_specs(prosody_detail: Iterable[Iterable[object]]) -> list[_MoraSpec]:
    specs: list[_MoraSpec] = []
    phrases = _as_list(prosody_detail, "prosody_detail")
    for phrase_index, phrase in enumerate(phrases):
        moras = _as_list(phrase, f"prosody_detail[{phrase_index}]")
        for mora_index, mora in enumerate(moras):
            context = f"prosody_detail[{phrase_index}][{mora_index}]"
            name = _field(mora, "phoneme", context)
            hira = _field(mora, "hira", context)
            if not isinstance(name, str) or not name:
                raise DurationConversionError(
                    f"{context}.phoneme must be a non-empty string"
                )
            if not isinstance(hira, str):
                raise DurationConversionError(f"{context}.hira must be a string")
            # 推論用トークンと同じ表記にそろえる。撥音のNと子音のnは区別し、応答のモーラ名は変更しない。
            phonemes = tuple(
                phoneme if phoneme == "N" else phoneme.lower()
                for phoneme in name.split("-")
            )
            if any(not phoneme for phoneme in phonemes):
                raise DurationConversionError(
                    f"{context}.phoneme contains an empty phoneme"
                )
            specs.append(_MoraSpec(name=name, hira=hira, phonemes=phonemes))
    return specs


def _pause_duration(
    cursor: int, frame_count: int, hop_length: int
) -> tuple[MoraDuration, int]:
    end = cursor + frame_count * hop_length
    wav_range = TimeRange(start=cursor, end=end)
    return (
        MoraDuration(
            mora="pau",
            hira="",
            phonemePitches=[PhonemeDuration(phoneme="pau", wavRange=wav_range)],
            wavRange=wav_range,
        ),
        end,
    )


def convert_duration(
    plain_tokens: Sequence[str],
    prosody_detail: Sequence[Sequence[object]],
    duration_frames: Sequence[int],
    hop_length: int,
) -> list[MoraDuration]:
    """
    v2のプレーンなトークン列とESPnetの継続長をモーラの範囲へ変換する。

    Parameters
    ----------
    plain_tokens : Sequence[str]
        境界記号を含む平坦なv2のトークン列
    prosody_detail : Sequence[Sequence[object]]
        句とモーラの順に入れ子になったv2の`Mora`の値
    duration_frames : Sequence[int]
        プレーンなトークンごとの0以上のESPnetのフレーム数
    hop_length : int
        1音響フレームが表す出力サンプル数

    Returns
    -------
    list[MoraDuration]
        音声中の順に並べた`MoraDuration`のリスト
        入れ子の音素とモーラの範囲は0から始まって連続し、`sum(duration_frames) * hop_length`で終わる。

    Raises
    ------
    DurationConversionError
        入力の列が不正な場合や互いに対応しない場合
    """
    tokens = _as_list(plain_tokens, "plain_tokens")
    frames = [
        _integer(value, f"duration_frames[{i}]")
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
                f"plain_tokens[{i}] must be a non-empty string"
            )

    specs = _mora_specs(prosody_detail)
    result: list[MoraDuration] = []
    cursor = 0
    pending_frames = 0
    spec_index = 0
    phoneme_index = 0
    current_mora_start = None
    current_phonemes: list[PhonemeDuration] = []
    saw_leading_pause = False
    saw_terminal_pause = False

    for token_index, (token, frame_count) in enumerate(
        zip(tokens, frames, strict=True)
    ):
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
                    f"{token} must occur exactly at the end of plain_tokens"
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
                raise DurationConversionError("pause marker split a mora")
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
                f"plain token {token} at index {token_index} does not match prosody phoneme {expected}"
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
