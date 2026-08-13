"""COEIROINK v2辞書ペイロードを公開Engineへ渡すアダプターです。
v2エンドポイントはUUID単位ではなく辞書全体を受け取ります。
このモジュールは凍結版公開Engineの``UserDictWord``へ変換し、コンパイル済み辞書を置き換えます。
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union
from uuid import NAMESPACE_URL, uuid5

from voicevox_engine.model import UserDictWord
from voicevox_engine.user_dict import (
    compiled_dict_path as default_compiled_dict_path,
)
from voicevox_engine.user_dict import create_word
from voicevox_engine.user_dict import default_dict_path as bundled_dict_path
from voicevox_engine.user_dict import update_dict
from voicevox_engine.user_dict import user_dict_path as default_user_dict_path
from voicevox_engine.user_dict import write_to_json

from .models import DictionaryWord, DictionaryWords


class DictionaryError(ValueError):
    """v2辞書ペイロードを安全にコンパイルできない場合に発生します。"""


def _validated_words(
    payload: Union[DictionaryWords, Sequence[DictionaryWord]],
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
    payload: Union[DictionaryWords, Sequence[DictionaryWord]],
) -> Dict[str, UserDictWord]:
    """v2ペイロードから公開Engine用の安定した辞書レコードを構築します。"""

    result: Dict[str, UserDictWord] = {}
    for index, item in enumerate(_validated_words(payload)):
        try:
            record = create_word(
                surface=item.word,
                pronunciation=item.yomi,
                accent_type=item.accent,
            )
        except Exception as error:
            raise DictionaryError(
                "invalid dictionary word at index {}: {}".format(index, error)
            ) from error
        if record.mora_count != item.num_moras:
            raise DictionaryError(
                "numMoras does not match yomi at index {}: {} != {}".format(
                    index, item.num_moras, record.mora_count
                )
            )
        stable_key = "{}\0{}\0{}\0{}\0{}".format(
            index, item.word, item.yomi, item.accent, item.num_moras
        )
        result[str(uuid5(NAMESPACE_URL, stable_key))] = record
    return result


def set_dictionary(
    payload: Union[DictionaryWords, Sequence[DictionaryWord]],
    *,
    user_dict_path: Optional[Path] = None,
    compiled_dict_path: Optional[Path] = None,
    default_dict_path: Optional[Path] = None,
) -> None:
    """公開Engineのユーザー辞書を置き換えてコンパイルします。"""

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
