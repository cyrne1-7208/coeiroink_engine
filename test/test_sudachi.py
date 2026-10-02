"""Sudachi任意機能の境界と最小動作を確認する。"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from voicevox_engine.sudachi import SudachiAnalyzer
from voicevox_engine.sudachi.open_jtalk_dictionary import (
    OpenJTalkDictionaryEntry,
    OpenJTalkDictionaryWarning,
    _to_sudachi_row,
    load_open_jtalk_csv,
)

ENGINE_ROOT = Path(__file__).resolve().parents[1]


def _write_open_jtalk_csv(path: Path, *, surface: str, reading: str, cost: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as dictionary_file:
        csv.writer(dictionary_file, lineterminator="\n").writerow(
            [
                surface,
                1348,
                1348,
                cost,
                "名詞",
                "固有名詞",
                "一般",
                "*",
                "*",
                "*",
                "*",
                reading,
                reading,
                "0/6",
                "*",
            ]
        )


def test_cli_requires_explicit_enable_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "voicevox_engine.sudachi", "東京都へ行く"],
        cwd=ENGINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--experimental sudachi" in result.stderr


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_full_dictionary_tokenizes_in_each_mode(mode: str) -> None:
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_full")

    with SudachiAnalyzer(mode=mode) as analyzer:
        morphemes = analyzer.tokenize("東京都へ行く")

    assert morphemes
    assert "".join(morpheme.surface for morpheme in morphemes) == "東京都へ行く"
    assert all(morpheme.begin <= morpheme.end for morpheme in morphemes)


@pytest.mark.parametrize(
    ("source_cost", "expected_cost"),
    [(-40000, -32767), (40000, 32767)],
)
def test_open_jtalk_cost_outside_sudachi_range_is_limited(
    tmp_path: Path, source_cost: int, expected_cost: int
) -> None:
    source = tmp_path / "environment.csv"
    _write_open_jtalk_csv(
        source,
        surface="互換辞書語",
        reading="ゴカンジショゴ",
        cost=source_cost,
    )

    with pytest.warns(OpenJTalkDictionaryWarning, match="limited"):
        entries = load_open_jtalk_csv([source])

    assert entries[0].cost == expected_cost


@pytest.mark.parametrize(
    ("open_jtalk_pos", "expected_context_id", "expected_sudachi_pos"),
    [
        (
            ("名詞", "固有名詞", "人名", "名", "*", "*"),
            4789,
            ("名詞", "固有名詞", "人名", "名", "*", "*"),
        ),
        (
            ("名詞", "固有名詞", "人名", "姓", "*", "*"),
            4790,
            ("名詞", "固有名詞", "人名", "姓", "*", "*"),
        ),
        (
            ("名詞", "固有名詞", "一般", "*", "*", "*"),
            4786,
            ("名詞", "固有名詞", "一般", "*", "*", "*"),
        ),
        (
            ("名詞", "一般", "*", "*", "*", "*"),
            5146,
            ("名詞", "普通名詞", "一般", "*", "*", "*"),
        ),
        (
            ("動詞", "自立", "*", "*", "*", "*"),
            925,
            ("動詞", "一般", "*", "*", "サ行変格", "終止形-一般"),
        ),
        (
            ("形容詞", "自立", "*", "*", "*", "*"),
            5166,
            ("形容詞", "一般", "*", "*", "形容詞", "終止形-一般"),
        ),
        (
            ("名詞", "接尾", "一般", "*", "*", "*"),
            5771,
            ("接尾辞", "名詞的", "一般", "*", "*", "*"),
        ),
    ],
)
def test_open_jtalk_part_of_speech_uses_matching_sudachi_context(
    open_jtalk_pos: tuple[str, str, str, str, str, str],
    expected_context_id: int,
    expected_sudachi_pos: tuple[str, str, str, str, str, str],
) -> None:
    row = _to_sudachi_row(
        OpenJTalkDictionaryEntry(
            surface="互換辞書語",
            cost=0,
            part_of_speech=open_jtalk_pos,
            reading="ゴカンジショゴ",
        )
    )

    assert row[1:3] == [expected_context_id, expected_context_id]
    assert tuple(row[5:11]) == expected_sudachi_pos


def test_unknown_open_jtalk_part_of_speech_remains_usable() -> None:
    entry = OpenJTalkDictionaryEntry(
        surface="互換辞書語",
        cost=0,
        part_of_speech=("未知品詞", "*", "*", "*", "*", "*"),
        reading="ゴカンジショゴ",
    )

    with pytest.warns(OpenJTalkDictionaryWarning, match="common noun"):
        row = _to_sudachi_row(entry)

    assert row[1:3] == [5146, 5146]
    assert tuple(row[5:11]) == ("名詞", "普通名詞", "一般", "*", "*", "*")


def test_open_jtalk_user_json_has_priority_over_environment_csv(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_full")
    environment_csv = tmp_path / "environment.csv"
    user_json = tmp_path / "user_dict.json"
    _write_open_jtalk_csv(
        environment_csv,
        surface="優先辞書語",
        reading="カンキョウ",
        cost=0,
    )
    user_json.write_text(
        json.dumps(
            {
                "00000000-0000-0000-0000-000000000000": {
                    "surface": "優先辞書語",
                    "cost": 0,
                    "part_of_speech": "名詞",
                    "part_of_speech_detail_1": "固有名詞",
                    "part_of_speech_detail_2": "一般",
                    "part_of_speech_detail_3": "*",
                    "inflectional_type": "*",
                    "inflectional_form": "*",
                    "yomi": "ユーザー",
                    "pronunciation": "ユーザー",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with SudachiAnalyzer(
        open_jtalk_csv_paths=[environment_csv],
        open_jtalk_user_dict_path=user_json,
    ) as analyzer:
        morphemes = analyzer.tokenize("優先辞書語")
        dictionary_count = analyzer.open_jtalk_dictionary_count

    assert dictionary_count == 2
    assert [(morpheme.surface, morpheme.reading_form) for morpheme in morphemes] == [
        ("優先辞書語", "ユーザー")
    ]
    assert morphemes[0].dictionary_id == 2


def test_bundled_open_jtalk_dictionary_can_be_loaded() -> None:
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_full")

    with SudachiAnalyzer(
        open_jtalk_csv_paths=[ENGINE_ROOT / "default.csv"]
    ) as analyzer:
        morphemes = analyzer.tokenize("COEIROINK")

    assert [(morpheme.surface, morpheme.reading_form) for morpheme in morphemes] == [
        ("COEIROINK", "コエイロインク")
    ]
    assert morphemes[0].dictionary_id == 1
