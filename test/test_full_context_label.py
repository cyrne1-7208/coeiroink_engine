# Phoneme・Mora・AccentPhrase・BreathGroup・UtteranceからなるOpen JTalkのフルコンテキストラベルの階層構造のテスト

from __future__ import annotations

from copy import deepcopy
from itertools import chain

import pytest

from voicevox_engine.full_context_label import (
    AccentPhrase,
    BreathGroup,
    Mora,
    Phoneme,
    Utterance,
)

# pyopenjtalk.extract_fullcontext("こんにちは、ヒホです。")の結果を固定し、外部辞書変更に依存せず検証する。
HELLO_HIHO_LABELS: tuple[str, ...] = (
    # sil: 無音
    (
        "xx^xx-sil+k=o/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:5_5%0_xx_xx/H:xx_xx/I:xx-xx"
        "@xx+xx&xx-xx|xx+xx/J:1_5/K:2+2-9"
    ),
    # k
    (
        "xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # o
    (
        "sil^k-o+N=n/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # N: ん
    (
        "k^o-N+n=i/A:-3+2+4/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # n
    (
        "o^N-n+i=ch/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # i
    (
        "N^n-i+ch=i/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # ch
    (
        "n^i-ch+i=w/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # i
    (
        "i^ch-i+w=a/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # w
    (
        "ch^i-w+a=pau/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # a
    (
        "i^w-a+pau=h/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
        "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
        "@1+2&1-2|1+9/J:1_4/K:2+2-9"
    ),
    # pau: 読点
    (
        "w^a-pau+h=i/A:xx+xx+xx/B:09-xx_xx/C:xx_xx+xx/D:09+xx_xx/E:5_5!0_xx-xx"
        "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:4_1%0_xx_xx/H:1_5/I:xx-xx"
        "@xx+xx&xx-xx|xx+xx/J:1_4/K:2+2-9"
    ),
    # h
    (
        "a^pau-h+i=h/A:0+1+4/B:09-xx_xx/C:09_xx+xx/D:22+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # i
    (
        "pau^h-i+h=o/A:0+1+4/B:09-xx_xx/C:09_xx+xx/D:22+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # h
    (
        "h^i-h+o=d/A:1+2+3/B:09-xx_xx/C:22_xx+xx/D:10+7_2/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # o
    (
        "i^h-o+d=e/A:1+2+3/B:09-xx_xx/C:22_xx+xx/D:10+7_2/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # d
    (
        "h^o-d+e=s/A:2+3+2/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # e
    (
        "o^d-e+s=U/A:2+3+2/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # s
    (
        "d^e-s+U=sil/A:3+4+1/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # U: 無声母音
    (
        "e^s-U+sil=xx/A:3+4+1/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
        "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
        "@2+1&2-1|6+4/J:xx_xx/K:2+2-9"
    ),
    # sil: 無音
    (
        "s^U-sil+xx=xx/A:xx+xx+xx/B:10-7_2/C:xx_xx+xx/D:xx+xx_xx/E:4_1!0_xx-xx"
        "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx/H:1_4/I:xx-xx"
        "@xx+xx&xx-xx|xx+xx/J:xx_xx/K:2+2-9"
    ),
)


@pytest.fixture
def hello_hiho_phonemes() -> list[Phoneme]:
    return [Phoneme.from_label(label) for label in HELLO_HIHO_LABELS]


def test_phoneme_parses_names_pauses_and_round_trips_labels(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    assert (
        " ".join(phoneme.phoneme for phoneme in hello_hiho_phonemes)
        == "sil k o N n i ch i w a pau h i h o d e s U sil"
    )
    assert [
        index for index, phoneme in enumerate(hello_hiho_phonemes) if phoneme.is_pause()
    ] == [0, 10, 19]
    assert [phoneme.label for phoneme in hello_hiho_phonemes] == list(HELLO_HIHO_LABELS)


def test_mora_exposes_phonemes_and_labels(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    cases = (
        (1, 2, "ko", 1, 3),
        (None, 3, "N", 3, 4),
        (4, 5, "ni", 4, 6),
        (6, 7, "chi", 6, 8),
        (8, 9, "wa", 8, 10),
        (11, 12, "hi", 11, 13),
        (13, 14, "ho", 13, 15),
        (15, 16, "de", 15, 17),
        (17, 18, "sU", 17, 19),
    )
    for consonant_idx, vowel_idx, expected_text, label_start, label_end in cases:
        mora = Mora(
            consonant=(
                None if consonant_idx is None else hello_hiho_phonemes[consonant_idx]
            ),
            vowel=hello_hiho_phonemes[vowel_idx],
        )
        assert "".join(phoneme.phoneme for phoneme in mora.phonemes) == expected_text
        assert mora.labels == list(HELLO_HIHO_LABELS[label_start:label_end])


def test_mora_set_context_updates_all_phonemes(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    mora = deepcopy(
        Mora(consonant=hello_hiho_phonemes[1], vowel=hello_hiho_phonemes[2])
    )
    mora.set_context("p3", "a")

    assert "".join(phoneme.phoneme for phoneme in mora.phonemes) == "aa"


def test_accent_phrase_extracts_accent_phonemes_labels_and_merges(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    hello = AccentPhrase.from_phonemes(hello_hiho_phonemes[1:10])
    hiho = AccentPhrase.from_phonemes(hello_hiho_phonemes[11:19])

    assert hello.accent == 5
    assert hiho.accent == 1
    assert " ".join(p.phoneme for p in hello.phonemes) == "k o N n i ch i w a"
    assert " ".join(p.phoneme for p in hiho.phonemes) == "h i h o d e s U"
    assert hello.labels == list(HELLO_HIHO_LABELS[1:10])
    assert hiho.labels == list(HELLO_HIHO_LABELS[11:19])

    merged = hello.merge(hiho)
    assert merged.accent == 5
    assert (
        " ".join(p.phoneme for p in merged.phonemes)
        == "k o N n i ch i w a h i h o d e s U"
    )
    assert merged.labels == list(HELLO_HIHO_LABELS[1:10] + HELLO_HIHO_LABELS[11:19])

    mutated = deepcopy(hello)
    mutated.set_context("p3", "a")
    assert "".join(p.phoneme for p in mutated.phonemes) == "aaaaaaaaa"


def test_breath_group_exposes_phonemes_labels_and_updates_context(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    hello = BreathGroup.from_phonemes(hello_hiho_phonemes[1:10])
    hiho = BreathGroup.from_phonemes(hello_hiho_phonemes[11:19])

    assert " ".join(p.phoneme for p in hello.phonemes) == "k o N n i ch i w a"
    assert " ".join(p.phoneme for p in hiho.phonemes) == "h i h o d e s U"
    assert hello.labels == list(HELLO_HIHO_LABELS[1:10])
    assert hiho.labels == list(HELLO_HIHO_LABELS[11:19])

    mutated = deepcopy(hello)
    mutated.set_context("p3", "a")
    assert "".join(p.phoneme for p in mutated.phonemes) == "aaaaaaaaa"


def test_utterance_round_trips_labels_and_recomputes_group_contexts(
    hello_hiho_phonemes: list[Phoneme],
) -> None:
    utterance = Utterance.from_phonemes(hello_hiho_phonemes)

    assert utterance.labels == list(HELLO_HIHO_LABELS)
    assert (
        " ".join(p.phoneme for p in utterance.phonemes)
        == "sil k o N n i ch i w a pau h i h o d e s U sil"
    )

    rebuilt = Utterance.from_phonemes(utterance.phonemes)
    assert len(rebuilt.breath_groups) == 2
    accent_phrases = list(
        chain.from_iterable(group.accent_phrases for group in rebuilt.breath_groups)
    )
    total_phrases = str(len(accent_phrases))

    for prev, cent, post in zip(
        [None, *accent_phrases[:-1]],
        accent_phrases,
        [*accent_phrases[1:], None],
        strict=True,
    ):
        mora_num = str(len(cent.moras))
        accent = str(cent.accent)
        if prev is not None:
            for phoneme in prev.phonemes:
                assert phoneme.contexts["g1"] == mora_num
                assert phoneme.contexts["g2"] == accent
        if post is not None:
            for phoneme in post.phonemes:
                assert phoneme.contexts["e1"] == mora_num
                assert phoneme.contexts["e2"] == accent
        for phoneme in cent.phonemes:
            assert phoneme.contexts["k2"] == total_phrases

    for prev, cent, post in zip(
        [None, *rebuilt.breath_groups[:-1]],
        rebuilt.breath_groups,
        [*rebuilt.breath_groups[1:], None],
        strict=True,
    ):
        phrase_num = str(len(cent.accent_phrases))
        if prev is not None:
            for phoneme in prev.phonemes:
                assert phoneme.contexts["j1"] == phrase_num
        if post is not None:
            for phoneme in post.phonemes:
                assert phoneme.contexts["h1"] == phrase_num
        first_index = accent_phrases.index(cent.accent_phrases[0])
        for phoneme in cent.phonemes:
            assert phoneme.contexts["i1"] == phrase_num
            assert phoneme.contexts["i5"] == str(first_index + 1)
            assert phoneme.contexts["i6"] == str(len(accent_phrases) - first_index)
