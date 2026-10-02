"""Open JTalkの辞書ソースをSudachiユーザー辞書へ変換する。"""

from __future__ import annotations

import csv
import json
import unicodedata
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OpenJTalkDictionaryError(ValueError):
    """Open JTalk辞書を安全に変換できない場合のエラー。"""


class OpenJTalkDictionaryWarning(UserWarning):
    """互換性を維持するため辞書値を補正した場合の警告。"""


@dataclass(frozen=True, slots=True)
class OpenJTalkDictionaryEntry:
    """Sudachiでも利用するOpen JTalk辞書の共通項目。"""

    surface: str
    cost: int
    part_of_speech: tuple[str, str, str, str, str, str]
    reading: str


_SUDACHI_MIN_WORD_COST = -(2**15) + 1
_SUDACHI_MAX_WORD_COST = 2**15 - 1
_SUDACHI_PROPER_NOUN_CONTEXT_ID = 4786
_SUDACHI_GIVEN_NAME_CONTEXT_ID = 4789
_SUDACHI_SURNAME_CONTEXT_ID = 4790
_SUDACHI_COMMON_NOUN_CONTEXT_ID = 5146
_SUDACHI_VERB_CONTEXT_ID = 925
_SUDACHI_ADJECTIVE_CONTEXT_ID = 5166
_SUDACHI_NOUN_SUFFIX_CONTEXT_ID = 5771

_SUDACHI_PROPER_NOUN_POS = ("名詞", "固有名詞", "一般", "*", "*", "*")
_SUDACHI_GIVEN_NAME_POS = ("名詞", "固有名詞", "人名", "名", "*", "*")
_SUDACHI_SURNAME_POS = ("名詞", "固有名詞", "人名", "姓", "*", "*")
_SUDACHI_COMMON_NOUN_POS = ("名詞", "普通名詞", "一般", "*", "*", "*")
_SUDACHI_VERB_POS = ("動詞", "一般", "*", "*", "サ行変格", "終止形-一般")
_SUDACHI_ADJECTIVE_POS = (
    "形容詞",
    "一般",
    "*",
    "*",
    "形容詞",
    "終止形-一般",
)
_SUDACHI_NOUN_SUFFIX_POS = ("接尾辞", "名詞的", "一般", "*", "*", "*")

# 詳細な人名を先に判定する。先頭一致にすることでOpen JTalk固有の活用欄を保持したまま代表的なSudachi文脈へ寄せられる。
_SUDACHI_POS_MAPPINGS = (
    (
        ("名詞", "固有名詞", "人名", "名"),
        _SUDACHI_GIVEN_NAME_CONTEXT_ID,
        _SUDACHI_GIVEN_NAME_POS,
    ),
    (
        ("名詞", "固有名詞", "人名", "姓"),
        _SUDACHI_SURNAME_CONTEXT_ID,
        _SUDACHI_SURNAME_POS,
    ),
    (
        ("名詞", "固有名詞"),
        _SUDACHI_PROPER_NOUN_CONTEXT_ID,
        _SUDACHI_PROPER_NOUN_POS,
    ),
    (
        ("名詞", "一般"),
        _SUDACHI_COMMON_NOUN_CONTEXT_ID,
        _SUDACHI_COMMON_NOUN_POS,
    ),
    (
        ("名詞", "普通名詞"),
        _SUDACHI_COMMON_NOUN_CONTEXT_ID,
        _SUDACHI_COMMON_NOUN_POS,
    ),
    (
        ("名詞", "接尾", "一般"),
        _SUDACHI_NOUN_SUFFIX_CONTEXT_ID,
        _SUDACHI_NOUN_SUFFIX_POS,
    ),
    (("動詞", "自立"), _SUDACHI_VERB_CONTEXT_ID, _SUDACHI_VERB_POS),
    (
        ("形容詞", "自立"),
        _SUDACHI_ADJECTIVE_CONTEXT_ID,
        _SUDACHI_ADJECTIVE_POS,
    ),
)


def _source_error(
    source: Path, location: str, message: str
) -> OpenJTalkDictionaryError:
    return OpenJTalkDictionaryError(f"{source}:{location}: {message}")


def _parse_cost(value: object, source: Path, location: str) -> int:
    try:
        cost = int(value)
    except (TypeError, ValueError) as error:
        raise _source_error(
            source, location, f"invalid Open JTalk cost: {value!r}"
        ) from error
    converted_cost = min(max(cost, _SUDACHI_MIN_WORD_COST), _SUDACHI_MAX_WORD_COST)
    if converted_cost != cost:
        # 飽和変換は大小関係を反転させないため、範囲外の辞書も読み込めることを優先する。
        warnings.warn(
            f"{source}:{location}: Open JTalk cost {cost} was limited to the Sudachi range as {converted_cost}",
            OpenJTalkDictionaryWarning,
            stacklevel=2,
        )
    return converted_cost


def _part_of_speech(values: Sequence[object]) -> tuple[str, str, str, str, str, str]:
    fields = [str(value) if value not in (None, "") else "*" for value in values[:6]]
    fields.extend("*" for _ in range(6 - len(fields)))
    return fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]


def _reading(primary: object, fallback: object, source: Path, location: str) -> str:
    for value in (primary, fallback):
        if isinstance(value, str) and value not in ("", "*"):
            return value
    raise _source_error(source, location, "reading or pronunciation is missing")


def load_open_jtalk_csv(paths: Sequence[Path]) -> list[OpenJTalkDictionaryEntry]:
    """Open JTalkのUTF-8 CSVを入力順のまま読み込む。"""

    entries: list[OpenJTalkDictionaryEntry] = []
    for source in (Path(path) for path in paths):
        if source.suffix.lower() == ".dic":
            raise OpenJTalkDictionaryError(
                f"compiled Open JTalk dictionary cannot be converted: {source}; specify its source CSV"
            )
        if not source.is_file():
            raise OpenJTalkDictionaryError(
                f"Open JTalk dictionary CSV was not found: {source}"
            )
        try:
            with source.open(encoding="utf-8-sig", newline="") as source_file:
                for line_number, row in enumerate(csv.reader(source_file), start=1):
                    if not row or not any(field for field in row):
                        continue
                    if row[0].lstrip().startswith("#"):
                        continue
                    if len(row) < 13:
                        raise _source_error(
                            source,
                            str(line_number),
                            f"expected at least 13 CSV fields, got {len(row)}",
                        )
                    surface = row[0]
                    if not surface:
                        raise _source_error(
                            source, str(line_number), "surface is empty"
                        )
                    entries.append(
                        OpenJTalkDictionaryEntry(
                            surface=surface,
                            cost=_parse_cost(row[3], source, str(line_number)),
                            part_of_speech=_part_of_speech(row[4:10]),
                            reading=_reading(
                                row[12], row[11], source, str(line_number)
                            ),
                        )
                    )
        except (UnicodeDecodeError, csv.Error) as error:
            raise OpenJTalkDictionaryError(
                f"failed to read Open JTalk dictionary CSV: {source}: {error}"
            ) from error
    return entries


def load_open_jtalk_user_json(path: Path) -> list[OpenJTalkDictionaryEntry]:
    """Engineが保存したOpen JTalkユーザー辞書JSONを読み込む。"""

    source = Path(path)
    if not source.is_file():
        raise OpenJTalkDictionaryError(
            f"Open JTalk user dictionary JSON was not found: {source}"
        )
    try:
        with source.open(encoding="utf-8-sig") as source_file:
            payload = json.load(source_file)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenJTalkDictionaryError(
            f"failed to read Open JTalk user dictionary JSON: {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise OpenJTalkDictionaryError(
            f"Open JTalk user dictionary JSON must contain an object: {source}"
        )

    entries: list[OpenJTalkDictionaryEntry] = []
    for word_id, value in payload.items():
        location = str(word_id)
        if not isinstance(value, dict):
            raise _source_error(source, location, "dictionary entry must be an object")
        surface = value.get("surface")
        if not isinstance(surface, str) or not surface:
            raise _source_error(source, location, "surface is missing or empty")
        if "cost" not in value:
            raise _source_error(source, location, "stored Open JTalk cost is missing")
        entries.append(
            OpenJTalkDictionaryEntry(
                surface=surface,
                cost=_parse_cost(value["cost"], source, location),
                part_of_speech=_part_of_speech(
                    (
                        value.get("part_of_speech"),
                        value.get("part_of_speech_detail_1"),
                        value.get("part_of_speech_detail_2"),
                        value.get("part_of_speech_detail_3"),
                        value.get("inflectional_type"),
                        value.get("inflectional_form"),
                    )
                ),
                reading=_reading(
                    value.get("pronunciation"), value.get("yomi"), source, location
                ),
            )
        )
    return entries


def _sudachi_part_of_speech(
    entry: OpenJTalkDictionaryEntry,
) -> tuple[int, tuple[str, str, str, str, str, str]]:
    """Open JTalkの代表品詞をUniDic 2.1.2由来のSudachi文脈へ対応付ける。"""

    part_of_speech = entry.part_of_speech
    for source_prefix, context_id, sudachi_part_of_speech in _SUDACHI_POS_MAPPINGS:
        if part_of_speech[: len(source_prefix)] == source_prefix:
            return context_id, sudachi_part_of_speech

    # 未知のOpen JTalk品詞でも読みを利用できることを優先し、Sudachi公式推奨の普通名詞へ寄せる。
    warnings.warn(
        f"Open JTalk part of speech {part_of_speech!r} for {entry.surface!r} was mapped to a Sudachi common noun",
        OpenJTalkDictionaryWarning,
        stacklevel=2,
    )
    return _SUDACHI_COMMON_NOUN_CONTEXT_ID, _SUDACHI_COMMON_NOUN_POS


def _to_sudachi_row(entry: OpenJTalkDictionaryEntry) -> list[str | int]:
    normalized_surface = unicodedata.normalize("NFKC", entry.surface).lower()
    context_id, part_of_speech = _sudachi_part_of_speech(entry)
    return [
        normalized_surface,
        context_id,
        context_id,
        entry.cost,
        entry.surface,
        *part_of_speech,
        entry.reading,
        entry.surface,
        "*",
        "C",
        "*",
        "*",
        "*",
    ]


def _compile_dictionary(
    sudachipy: Any,
    *,
    entries: Sequence[OpenJTalkDictionaryEntry],
    system_dictionary: Path,
    source_path: Path,
    output_path: Path,
    description: str,
) -> Path | None:
    if not entries:
        return None
    with source_path.open("w", encoding="utf-8", newline="") as source_file:
        writer = csv.writer(source_file, lineterminator="\n")
        writer.writerows(_to_sudachi_row(entry) for entry in entries)
    try:
        # SudachiPy自身のCLIと同じビルダーを直接呼び、外部コマンドやRustツールチェーンへの依存を増やさない。
        sudachipy.sudachipy.build_user_dic(
            system=system_dictionary,
            lex=[source_path],
            output=output_path,
            description=description,
        )
    except Exception as error:
        raise OpenJTalkDictionaryError(
            f"failed to build Sudachi user dictionary from {source_path}: {error}"
        ) from error
    if not output_path.is_file():
        raise OpenJTalkDictionaryError(
            f"Sudachi user dictionary was not created: {output_path}"
        )
    return output_path


def compile_open_jtalk_dictionaries(
    sudachipy: Any,
    *,
    system_dictionary: Path,
    output_directory: Path,
    csv_paths: Sequence[Path] = (),
    user_json_path: Path | None = None,
) -> tuple[Path, ...]:
    """環境CSVを先、ユーザーJSONを後にコンパイルして優先順を保持する。"""

    if not system_dictionary.is_file():
        raise OpenJTalkDictionaryError(
            f"Sudachi full system dictionary was not found: {system_dictionary}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    layers: list[Path] = []

    environment_dictionary = _compile_dictionary(
        sudachipy,
        entries=load_open_jtalk_csv(csv_paths),
        system_dictionary=system_dictionary,
        source_path=output_directory / "open_jtalk_environment.csv",
        output_path=output_directory / "open_jtalk_environment.dic",
        description="COEIROINK Open JTalk environment dictionary",
    )
    if environment_dictionary is not None:
        layers.append(environment_dictionary)

    if user_json_path is not None:
        user_dictionary = _compile_dictionary(
            sudachipy,
            entries=load_open_jtalk_user_json(user_json_path),
            system_dictionary=system_dictionary,
            source_path=output_directory / "open_jtalk_user.csv",
            output_path=output_directory / "open_jtalk_user.dic",
            description="COEIROINK Open JTalk user dictionary",
        )
        if user_dictionary is not None:
            layers.append(user_dictionary)
    return tuple(layers)
