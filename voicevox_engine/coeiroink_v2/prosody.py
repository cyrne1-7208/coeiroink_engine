"""Public-source prosody conversion for the COEIROINK v2 HTTP API.

Text and kana are analysed by the frozen public Engine frontend, while the
frozen public Core ``query2tokens_prosody`` function creates the flat token
stream.  This module only performs the small v2 wire-format conversion.
"""

from collections.abc import Sequence
from functools import lru_cache

from coeirocore.model import AccentPhrase as CoreAccentPhrase
from coeirocore.model import AudioQuery as CoreAudioQuery
from coeirocore.model import Mora as CoreMora
from coeirocore.query_manager import query2tokens_prosody

from ..kana_parser import ParseKanaError, parse_kana
from ..model import AccentPhrase as EngineAccentPhrase
from ..model import Mora as EngineMora
from ..text_analysis import analyze_text
from .models import Mora as ProsodyMora
from .models import Prosody


class ProsodyError(ValueError):
    """Raised when a v2 prosody request cannot be analysed deterministically."""


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProsodyError(f"{field_name} must be a string")
    return value


def _to_core_mora(mora: EngineMora) -> CoreMora:
    return CoreMora(
        text=mora.text,
        consonant=mora.consonant,
        consonant_length=mora.consonant_length,
        vowel=mora.vowel,
        vowel_length=mora.vowel_length,
        pitch=mora.pitch,
    )


def _to_core_query(
    accent_phrases: Sequence[EngineAccentPhrase],
) -> CoreAudioQuery:
    """Engineの解析結果をCoreのプロソディトークン変換だけに必要なAudioQueryへ写像する。"""

    return CoreAudioQuery(
        accent_phrases=[
            CoreAccentPhrase(
                moras=[_to_core_mora(mora) for mora in phrase.moras],
                accent=phrase.accent,
                pause_mora=(
                    _to_core_mora(phrase.pause_mora)
                    if phrase.pause_mora is not None
                    else None
                ),
                is_interrogative=phrase.is_interrogative,
            )
            for phrase in accent_phrases
        ],
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.0,
        postPhonemeLength=0.0,
        outputSamplingRate=44100,
        outputStereo=False,
        kana=None,
    )


def _to_hiragana(text: str) -> str:
    """Convert the public Engine's katakana mora spelling to hiragana."""

    return "".join(
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char for char in text
    )


def _mora_accent(index: int, accent: int, mora_count: int) -> int:
    """Return the v2 high/low accent marker for one mora."""

    if not 0 <= accent <= mora_count:
        raise ProsodyError("accent must be between 0 and the number of moras")
    if accent == 0:
        return 0
    if accent == 1:
        return int(index == 0)
    return int(0 < index < accent)


def _to_prosody_moras(phrase: EngineAccentPhrase) -> list[ProsodyMora]:
    mora_count = len(phrase.moras)
    result = []
    for index, mora in enumerate(phrase.moras):
        vowel = mora.vowel if mora.vowel == "N" else mora.vowel.lower()
        phoneme = f"{mora.consonant.lower()}-{vowel}" if mora.consonant else vowel
        result.append(
            ProsodyMora(
                phoneme=phoneme,
                hira=_to_hiragana(mora.text),
                accent=_mora_accent(index, phrase.accent, mora_count),
            )
        )
    return result


def _special_detail(phoneme: str, hira: str) -> list[ProsodyMora]:
    return [ProsodyMora(phoneme=phoneme, hira=hira, accent=0)]


def _build_prosody(
    accent_phrases: Sequence[EngineAccentPhrase],
) -> Prosody:
    core_query = _to_core_query(accent_phrases)
    plain = query2tokens_prosody(core_query)
    detail: list[list[ProsodyMora]] = []
    for phrase in accent_phrases:
        detail.append(_to_prosody_moras(phrase))
        if phrase.pause_mora is not None:
            detail.append(_special_detail("_", "、"))
        if phrase.is_interrogative:
            detail.append(_special_detail("?", "？"))
    return Prosody(plain=plain, detail=detail)


@lru_cache(maxsize=256)
def _estimate_prosody_cached(text: str) -> Prosody:
    """Cache the expensive Open JTalk analysis for repeated request text."""
    try:
        accent_phrases = analyze_text(text)
    except Exception as error:
        raise ProsodyError(f"text analysis failed: {error}") from error
    return _build_prosody(accent_phrases)


def estimate_prosody(text: str) -> Prosody:
    """Estimate v2 prosody from ordinary text.

    A deep copy keeps the cache immutable from the caller's perspective.
    ``clear_prosody_cache`` is called after user-dictionary updates so cached
    readings cannot outlive the dictionary that produced them.
    """

    text = _require_string(text, "text")
    return _estimate_prosody_cached(text).model_copy(deep=True)


def clear_prosody_cache() -> None:
    """Discard text analyses made before the current user dictionary update."""

    _estimate_prosody_cached.cache_clear()


def estimate_prosody_from_kana(kana: str) -> Prosody:
    """Estimate v2 prosody from the public Engine's AquesTalk-like kana."""

    kana = _require_string(kana, "kana")
    try:
        accent_phrases = parse_kana(kana)
    except ParseKanaError as error:
        raise ProsodyError(f"invalid kana: {error.errname}: {error.text}") from error
    return _build_prosody(accent_phrases)


__all__ = [
    "ProsodyError",
    "clear_prosody_cache",
    "estimate_prosody",
    "estimate_prosody_from_kana",
]
