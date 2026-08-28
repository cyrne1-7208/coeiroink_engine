"""Sudachi任意機能の境界と最小動作を確認する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voicevox_engine.sudachi import SudachiAnalyzer

ENGINE_ROOT = Path(__file__).resolve().parents[1]


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
