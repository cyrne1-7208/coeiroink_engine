"""Adapter from the COEIROINK v2 dictionary payload to the public Engine.

The v2 endpoint submits the complete dictionary rather than individual UUID
operations.  This module converts that list to the frozen public Engine's
``UserDictWord`` representation and replaces the compiled user dictionary.
"""

from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from voicevox_engine.model import UserDictWord
from voicevox_engine.user_dict import (
    compiled_dict_path as default_compiled_dict_path,
)
from voicevox_engine.user_dict import create_word, update_dict, write_to_json
from voicevox_engine.user_dict import default_dict_path as bundled_dict_path
from voicevox_engine.user_dict import user_dict_path as default_user_dict_path

from .models import DictionaryWord, DictionaryWords


class DictionaryError(ValueError):
    """Raised when a v2 dictionary payload cannot be compiled safely."""


def _validated_words(
    payload: DictionaryWords | Sequence[DictionaryWord],
) -> Sequence[DictionaryWord]:
    if isinstance(payload, DictionaryWords):
        return payload.dictionary_words
    if isinstance(payload, (str, bytes)):
        raise DictionaryError("dictionary words must be a sequence")
    try:
        return [
            value
            if isinstance(value, DictionaryWord)
            else DictionaryWord.model_validate(value)
            for value in payload
        ]
    except (TypeError, ValueError) as error:
        raise DictionaryError("dictionary contains an invalid word") from error


def build_user_dictionary(
    payload: DictionaryWords | Sequence[DictionaryWord],
) -> dict[str, UserDictWord]:
    """Build stable public-Engine dictionary records from a v2 payload."""

    result: dict[str, UserDictWord] = {}
    for index, item in enumerate(_validated_words(payload)):
        try:
            record = create_word(
                surface=item.word,
                pronunciation=item.yomi,
                accent_type=item.accent,
            )
        except Exception as error:
            raise DictionaryError(
                f"invalid dictionary word at index {index}: {error}"
            ) from error
        if record.mora_count != item.num_moras:
            raise DictionaryError(
                f"numMoras does not match yomi at index {index}: {item.num_moras} != {record.mora_count}"
            )
        # v2は単語UUIDを受け取らないため、同じ辞書内容から毎回同じ公開Engine形式を作れるよう内容ベースのUUIDを生成する。
        stable_key = (
            f"{index}\0{item.word}\0{item.yomi}\0{item.accent}\0{item.num_moras}"
        )
        result[str(uuid5(NAMESPACE_URL, stable_key))] = record
    return result


def set_dictionary(
    payload: DictionaryWords | Sequence[DictionaryWord],
    *,
    user_dict_path: Path | None = None,
    compiled_dict_path: Path | None = None,
    default_dict_path: Path | None = None,
) -> None:
    """Replace and compile the public Engine user dictionary."""

    target_json = Path(user_dict_path or default_user_dict_path)
    target_compiled = Path(compiled_dict_path or default_compiled_dict_path)
    source_default = Path(default_dict_path or bundled_dict_path)
    records = build_user_dictionary(payload)
    try:
        write_to_json(records, target_json)
        update_dict(
            default_dict_path=source_default,
            user_dict_path=target_json,
            compiled_dict_path=target_compiled,
        )
    except Exception as error:
        raise DictionaryError("failed to compile dictionary") from error


__all__ = [
    "DictionaryError",
    "build_user_dictionary",
    "set_dictionary",
]
