"""Sudachi full辞書を個別に検証するためのCLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        "--open-jtalk-dictionaries",
        action="store_true",
        help="Engineのdefault.csvと保存済みuser_dict.jsonを読み込む",
    )
    parser.add_argument(
        "--open-jtalk-csv",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="追加するOpen JTalk辞書ソースCSV（複数回指定可）",
    )
    parser.add_argument(
        "--open-jtalk-user-json",
        type=Path,
        metavar="PATH",
        help="Engine形式のOpen JTalkユーザー辞書JSON",
    )
    parser.add_argument(
        "text", nargs="?", help="解析する文字列。省略時は標準入力を読む"
    )
    return parser


def _open_jtalk_sources(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    csv_paths = list(args.open_jtalk_csv)
    user_json_path = args.open_jtalk_user_json
    if args.open_jtalk_dictionaries:
        # 通常のEngineと同じ保存場所だけを遅延参照し、Sudachi未使用時の依存や副作用を増やさない。
        from voicevox_engine.utility.path_utility import engine_root, get_save_dir

        bundled_dictionary = engine_root() / "default.csv"
        if bundled_dictionary not in csv_paths:
            csv_paths.insert(0, bundled_dictionary)
        if user_json_path is None:
            saved_user_dictionary = get_save_dir() / "user_dict.json"
            if saved_user_dictionary.is_file():
                user_json_path = saved_user_dictionary
    return csv_paths, user_json_path


def main(argv: list[str] | None = None) -> int:
    """明示フラグの確認後にだけSudachiを起動する。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if "sudachi" not in args.experimental:
        parser.error("Sudachi fullを使うには --experimental sudachi が必要です")

    text = args.text if args.text is not None else sys.stdin.read()
    csv_paths, user_json_path = _open_jtalk_sources(args)
    with SudachiAnalyzer(
        mode=args.mode,
        open_jtalk_csv_paths=csv_paths,
        open_jtalk_user_dict_path=user_json_path,
    ) as analyzer:
        morphemes = [morpheme.to_dict() for morpheme in analyzer.tokenize(text)]
        open_jtalk_dictionary_count = analyzer.open_jtalk_dictionary_count

    result = {
        "dictionary": "full",
        "mode": args.mode,
        "open_jtalk_dictionary_count": open_jtalk_dictionary_count,
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
