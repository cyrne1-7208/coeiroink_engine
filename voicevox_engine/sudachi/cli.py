"""Sudachi full辞書を個別に検証するためのCLI。"""

from __future__ import annotations

import argparse
import json
import sys

from .analyzer import SudachiAnalyzer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sudachi full dictionary analyzer")
    parser.add_argument(
        "--experimental",
        action="append",
        choices=("sudachi",),
        default=[],
        help="任意機能を明示的に有効化する",
    )
    parser.add_argument(
        "--mode",
        choices=("A", "B", "C"),
        default="C",
        help="Sudachiの分割モード（既定値: C）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="解析結果をJSONで出力する",
    )
    parser.add_argument(
        "text", nargs="?", help="解析する文字列。省略時は標準入力を読む"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """明示フラグの確認後にだけSudachiを起動する。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if "sudachi" not in args.experimental:
        parser.error("Sudachi fullを使うには --experimental sudachi が必要です")

    text = args.text if args.text is not None else sys.stdin.read()
    with SudachiAnalyzer(mode=args.mode) as analyzer:
        morphemes = [morpheme.to_dict() for morpheme in analyzer.tokenize(text)]

    result = {
        "dictionary": "full",
        "mode": args.mode,
        "morphemes": morphemes,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for morpheme in morphemes:
            print(
                f"{morpheme['surface']}\t{morpheme['reading_form']}\t"
                f"{','.join(morpheme['part_of_speech'])}"
            )
    return 0
