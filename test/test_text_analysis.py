"""テキスト解析（アクセント句変換）と未知英単語カタカナ読み補正の境界を検証する。"""

from __future__ import annotations

import pytest

from voicevox_engine.kana_parser import create_kana
from voicevox_engine.katakana_english import text_to_full_context_labels
from voicevox_engine.text_analysis import analyze_text


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_text_returns_empty_phrases_and_labels(text: str) -> None:
    assert analyze_text(text) == []
    assert text_to_full_context_labels(text) == []


def test_analyze_text_inserts_pause_between_breath_groups_and_marks_question() -> None:
    phrases = analyze_text("こんにちは、元気ですか？")

    assert any(phrase.pause_mora is not None for phrase in phrases[:-1])
    assert phrases[-1].pause_mora is None
    assert phrases[-1].is_interrogative is True


def test_katakana_english_handles_unknown_words_acronyms_and_fullwidth_ascii() -> None:
    word_kana = create_kana(analyze_text("synthesis", enable_katakana_english=True))
    fullwidth_kana = create_kana(
        analyze_text("ｓｙｎｔｈｅｓｉｓ", enable_katakana_english=True)
    )
    acronym_kana = create_kana(analyze_text("API", enable_katakana_english=True))

    assert "シンセシス" in word_kana.replace("'", "")
    assert fullwidth_kana == word_kana
    assert "エエピイアイ" in acronym_kana.replace("'", "")


def test_katakana_english_removes_fullwidth_space_pause_between_english_words() -> None:
    phrases = analyze_text("hello　world", enable_katakana_english=True)

    assert all(phrase.pause_mora is None for phrase in phrases)
