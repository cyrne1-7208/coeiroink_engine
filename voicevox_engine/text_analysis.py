"""Open JTalkの解析結果をEngineのアクセント句へ変換する。"""

from __future__ import annotations

from . import full_context_label
from .full_context_label import extract_full_context_label
from .model import AccentPhrase, Mora
from .mora_list import openjtalk_mora2text


def mora_to_text(mora: str) -> str:
    if mora[-1:] in ["A", "I", "U", "E", "O"]:
        # 無声化母音でも表記を引けるよう、辞書検索時だけ小文字へ正規化する。
        mora = mora[:-1] + mora[-1].lower()
    return openjtalk_mora2text.get(mora, mora)


def full_context_label_moras_to_moras(
    full_context_moras: list[full_context_label.Mora],
) -> list[Mora]:
    return [
        Mora(
            text=mora_to_text("".join(phoneme.phoneme for phoneme in mora.phonemes)),
            consonant=(mora.consonant.phoneme if mora.consonant is not None else None),
            consonant_length=0 if mora.consonant is not None else None,
            vowel=mora.vowel.phoneme,
            vowel_length=0,
            pitch=0,
        )
        for mora in full_context_moras
    ]


def analyze_text(
    text: str,
    *,
    enable_katakana_english: bool = False,
) -> list[AccentPhrase]:
    """テキストをアクセント句へ変換し、推論前の長さ・音高はゼロで返す。"""

    if not text.strip():
        return []

    utterance = extract_full_context_label(
        text,
        enable_katakana_english=enable_katakana_english,
    )
    if not utterance.breath_groups:
        return []

    return [
        AccentPhrase(
            moras=full_context_label_moras_to_moras(accent_phrase.moras),
            accent=accent_phrase.accent,
            pause_mora=(
                Mora(
                    text="、",
                    consonant=None,
                    consonant_length=None,
                    vowel="pau",
                    vowel_length=0,
                    pitch=0,
                )
                if (
                    accent_phrase_index == len(breath_group.accent_phrases) - 1
                    and breath_group_index != len(utterance.breath_groups) - 1
                )
                else None
            ),
            is_interrogative=accent_phrase.is_interrogative,
        )
        for breath_group_index, breath_group in enumerate(utterance.breath_groups)
        for accent_phrase_index, accent_phrase in enumerate(breath_group.accent_phrases)
    ]


__all__ = ["analyze_text", "full_context_label_moras_to_moras", "mora_to_text"]
