"""COEIROINK v2 HTTP API向けの公開ソース韻律変換です。
テキストとカナは凍結版の公開Engineフロントエンドで解析し、凍結版公開Coreの``query2tokens_prosody``が平坦なトークン列を作ります。
このモジュールはv2の通信形式への小さな変換だけを担当します。
"""

from functools import lru_cache
from typing import List, Optional, Sequence

from coeirocore.model import AccentPhrase as CoreAccentPhrase
from coeirocore.model import AudioQuery as CoreAudioQuery
from coeirocore.model import Mora as CoreMora
from coeirocore.query_manager import query2tokens_prosody

from ..kana_parser import ParseKanaError, parse_kana
from ..model import AccentPhrase as EngineAccentPhrase
from ..model import Mora as EngineMora
from ..synthesis_engine.synthesis_engine_base import SynthesisEngineBase
from .models import Mora as ProsodyMora
from .models import Prosody


class ProsodyError(ValueError):
    """v2の韻律要求を決定的に解析できない場合に発生します。"""


class _TextAnalysisEngine(SynthesisEngineBase):
    """公開テキスト解析器を再利用する音声合成なしのEngineフロントエンドです。"""

    @property
    def speakers(self) -> str:
        return ""

    @property
    def supported_devices(self) -> Optional[str]:
        return None

    def replace_phoneme_length(
        self,
        accent_phrases: List[EngineAccentPhrase],
        speaker_id: int,
    ) -> List[EngineAccentPhrase]:
        return accent_phrases

    def replace_mora_pitch(
        self,
        accent_phrases: List[EngineAccentPhrase],
        speaker_id: int,
    ) -> List[EngineAccentPhrase]:
        return accent_phrases

    def _synthesis_impl(self, query, speaker_id: int):
        raise NotImplementedError("the prosody adapter does not synthesize audio")


_TEXT_ANALYZER = _TextAnalysisEngine()


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProsodyError("{} must be a string".format(field_name))
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
    """公開Engineのカタカナ表記をひらがなへ変換します。"""

    return "".join(
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char
        for char in text
    )


def _mora_accent(index: int, accent: int, mora_count: int) -> int:
    """1モーラ分のv2高低アクセント記号を返します。"""

    if not 0 <= accent <= mora_count:
        raise ProsodyError("accent must be between 0 and the number of moras")
    if accent == 0:
        return int(index > 0)
    if accent == 1:
        return int(index == 0)
    return int(0 < index < accent)


def _to_prosody_moras(phrase: EngineAccentPhrase) -> List[ProsodyMora]:
    mora_count = len(phrase.moras)
    result = []
    for index, mora in enumerate(phrase.moras):
        vowel = mora.vowel if mora.vowel == "N" else mora.vowel.lower()
        phoneme = (
            "{}-{}".format(mora.consonant.lower(), vowel)
            if mora.consonant
            else vowel
        )
        result.append(
            ProsodyMora(
                phoneme=phoneme,
                hira=_to_hiragana(mora.text),
                accent=_mora_accent(index, phrase.accent, mora_count),
            )
        )
    return result


def _build_prosody(
    accent_phrases: Sequence[EngineAccentPhrase],
) -> Prosody:
    core_query = _to_core_query(accent_phrases)
    plain = query2tokens_prosody(core_query)
    detail = [_to_prosody_moras(phrase) for phrase in accent_phrases]
    return Prosody(plain=plain, detail=detail)


@lru_cache(maxsize=256)
def _estimate_prosody_cached(text: str) -> Prosody:
    """重いOpen JTalk解析を同じテキストの要求間でキャッシュします。"""
    try:
        accent_phrases = _TEXT_ANALYZER.create_accent_phrases(text, speaker_id=0)
    except Exception as error:
        raise ProsodyError("text analysis failed: {}".format(error)) from error
    return _build_prosody(accent_phrases)


def estimate_prosody(text: str) -> Prosody:
    """通常のテキストからv2韻律を推定します。
    深いコピーを返してキャッシュを呼び出し元から変更できないようにします。
    ユーザー辞書更新後は``clear_prosody_cache``を呼び、更新前の読みを残しません。
    """

    text = _require_string(text, "text")
    return _estimate_prosody_cached(text).model_copy(deep=True)


def clear_prosody_cache() -> None:
    """現在のユーザー辞書更新前に作られたテキスト解析を破棄します。"""

    _estimate_prosody_cached.cache_clear()


def estimate_prosody_from_kana(kana: str) -> Prosody:
    """公開EngineのAquesTalk風カナからv2韻律を推定します。"""

    kana = _require_string(kana, "kana")
    try:
        accent_phrases = parse_kana(kana)
    except ParseKanaError as error:
        raise ProsodyError(
            "invalid kana: {}: {}".format(error.errname, error.text)
        ) from error
    return _build_prosody(accent_phrases)


__all__ = [
    "ProsodyError",
    "clear_prosody_cache",
    "estimate_prosody",
    "estimate_prosody_from_kana",
]
