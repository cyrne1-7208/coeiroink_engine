import importlib
import json
import shutil
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

import numpy as np
import pyopenjtalk
from fastapi import HTTPException

from .model import UserDictWord, WordTypes
from .part_of_speech_data import MAX_PRIORITY, MIN_PRIORITY, part_of_speech_data
from .utility import delete_file, engine_root, get_save_dir, mutex_wrapper

root_dir = engine_root()
save_dir = get_save_dir()

if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

default_dict_path = root_dir / "default.csv"
user_dict_path = save_dir / "user_dict.json"
compiled_dict_path = save_dir / "user.dic"


mutex_user_dict = threading.Lock()
mutex_openjtalk_dict = threading.Lock()


def _create_user_dict(source_path: Path, compiled_path: Path) -> None:
    """pyopenjtalk 0.4.1の公開APIでユーザー辞書をコンパイルする。"""
    pyopenjtalk.mecab_dict_index(str(source_path), str(compiled_path))


def _set_user_dict(compiled_path: Path) -> None:
    """pyopenjtalk 0.4.1の公開APIでコンパイル済み辞書を適用する。"""
    pyopenjtalk.update_global_jtalk_with_user_dict(str(compiled_path))


def reset_user_dict() -> None:
    """ユーザー辞書更新後のpyopenjtalkを既定辞書へ戻す。"""
    if hasattr(pyopenjtalk, "unset_user_dict"):
        pyopenjtalk.unset_user_dict()
    else:
        # pyopenjtalk 0.4系には解除用公開APIがないため、非公開状態へ依存せずモジュール再読込で既定のOpenJTalkインスタンスを再生成する。
        importlib.reload(pyopenjtalk)


@mutex_wrapper(mutex_user_dict)
def write_to_json(user_dict: dict[str, UserDictWord], user_dict_path: Path):
    converted_user_dict = {}
    for word_uuid, word in user_dict.items():
        word_dict = word.model_dump()
        word_dict["cost"] = priority2cost(
            word_dict["context_id"], word_dict["priority"]
        )
        del word_dict["priority"]
        converted_user_dict[word_uuid] = word_dict
    # 予めjsonに変換できることを確かめる
    user_dict_json = json.dumps(converted_user_dict, ensure_ascii=False)
    user_dict_path.write_text(user_dict_json, encoding="utf-8")


@mutex_wrapper(mutex_openjtalk_dict)
def update_dict(
    default_dict_path: Path = default_dict_path,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    """既定辞書とユーザー辞書を一時領域でコンパイルし、成功後に実運用辞書へ置換して適用する。"""

    temporary_source_path: Path | None = None
    temporary_compiled_path: Path | None = None
    try:
        with NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, dir=save_dir
        ) as f:
            temporary_source_path = Path(f.name).resolve()
            if not default_dict_path.is_file():
                raise FileNotFoundError(
                    f"default dictionary was not found: {default_dict_path}"
                )
            default_dict = default_dict_path.read_text(encoding="utf-8")
            if default_dict == default_dict.rstrip():
                default_dict += "\n"
            f.write(default_dict)
            user_dict = read_dict(user_dict_path=user_dict_path)
            for word_uuid in user_dict:
                word = user_dict[word_uuid]
                f.write(
                    (
                        "{surface},{context_id},{context_id},{cost},{part_of_speech},"
                        + "{part_of_speech_detail_1},{part_of_speech_detail_2},"
                        + "{part_of_speech_detail_3},{inflectional_type},"
                        + "{inflectional_form},{stem},{yomi},{pronunciation},"
                        + "{accent_type}/{mora_count},{accent_associative_rule}\n"
                    ).format(
                        surface=word.surface,
                        context_id=word.context_id,
                        cost=priority2cost(word.context_id, word.priority),
                        part_of_speech=word.part_of_speech,
                        part_of_speech_detail_1=word.part_of_speech_detail_1,
                        part_of_speech_detail_2=word.part_of_speech_detail_2,
                        part_of_speech_detail_3=word.part_of_speech_detail_3,
                        inflectional_type=word.inflectional_type,
                        inflectional_form=word.inflectional_form,
                        stem=word.stem,
                        yomi=word.yomi,
                        pronunciation=word.pronunciation,
                        accent_type=word.accent_type,
                        mora_count=word.mora_count,
                        accent_associative_rule=word.accent_associative_rule,
                    )
                )
        with NamedTemporaryFile(delete=False, dir=save_dir) as compiled_file:
            temporary_compiled_path = Path(compiled_file.name).resolve()
        _create_user_dict(temporary_source_path, temporary_compiled_path)
        if temporary_source_path.is_file():
            delete_file(str(temporary_source_path))
        if not temporary_compiled_path.is_file():
            raise RuntimeError("辞書のコンパイル時にエラーが発生しました。")
        reset_user_dict()
        try:
            # 保存先が別ドライブでも更新できるよう、同一ファイルシステムを前提とするPath.replaceは使わない。
            shutil.move(temporary_compiled_path, compiled_dict_path)
        finally:
            if compiled_dict_path.is_file():
                _set_user_dict(compiled_dict_path.resolve(strict=True))
    finally:
        if temporary_source_path is not None and temporary_source_path.exists():
            delete_file(str(temporary_source_path))
        if temporary_compiled_path is not None and temporary_compiled_path.exists():
            delete_file(str(temporary_compiled_path))


@mutex_wrapper(mutex_user_dict)
def read_dict(user_dict_path: Path = user_dict_path) -> dict[str, UserDictWord]:
    if not user_dict_path.is_file():
        return {}
    with user_dict_path.open(encoding="utf-8") as f:
        result = {}
        for word_uuid, word in json.load(f).items():
            # cost2priorityで変換を行う際にcontext_idが必要となるが、
            # 0.12以前の辞書は、context_idがハードコーディングされていたためにユーザー辞書内に保管されていない
            # ハードコーディングされていたcontext_idは固有名詞を意味するものなので、固有名詞のcontext_idを補完する
            if word.get("context_id") is None:
                word["context_id"] = part_of_speech_data[
                    WordTypes.PROPER_NOUN
                ].context_id
            word["priority"] = cost2priority(word["context_id"], word["cost"])
            del word["cost"]
            result[str(UUID(word_uuid))] = UserDictWord(**word)

    return result


def create_word(
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: WordTypes | None = None,
    priority: int | None = None,
) -> UserDictWord:
    if word_type is None:
        word_type = WordTypes.PROPER_NOUN
    if word_type not in part_of_speech_data:
        raise HTTPException(status_code=422, detail="不明な品詞です")
    if priority is None:
        priority = 5
    if not MIN_PRIORITY <= priority <= MAX_PRIORITY:
        raise HTTPException(status_code=422, detail="優先度の値が無効です")
    pos_detail = part_of_speech_data[word_type]
    return UserDictWord(
        surface=surface,
        context_id=pos_detail.context_id,
        priority=priority,
        part_of_speech=pos_detail.part_of_speech,
        part_of_speech_detail_1=pos_detail.part_of_speech_detail_1,
        part_of_speech_detail_2=pos_detail.part_of_speech_detail_2,
        part_of_speech_detail_3=pos_detail.part_of_speech_detail_3,
        inflectional_type="*",
        inflectional_form="*",
        stem="*",
        yomi=pronunciation,
        pronunciation=pronunciation,
        accent_type=accent_type,
        accent_associative_rule="*",
    )


def apply_word(
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: WordTypes | None = None,
    priority: int | None = None,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> str:
    word = create_word(
        surface=surface,
        pronunciation=pronunciation,
        accent_type=accent_type,
        word_type=word_type,
        priority=priority,
    )
    user_dict = read_dict(user_dict_path=user_dict_path)
    word_uuid = str(uuid4())
    user_dict[word_uuid] = word
    write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)
    return word_uuid


def rewrite_word(
    word_uuid: str,
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: WordTypes | None = None,
    priority: int | None = None,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    word = create_word(
        surface=surface,
        pronunciation=pronunciation,
        accent_type=accent_type,
        word_type=word_type,
        priority=priority,
    )
    user_dict = read_dict(user_dict_path=user_dict_path)
    if word_uuid not in user_dict:
        raise HTTPException(
            status_code=422, detail="UUIDに該当するワードが見つかりませんでした"
        )
    user_dict[word_uuid] = word
    write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)


def delete_word(
    word_uuid: str,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    user_dict = read_dict(user_dict_path=user_dict_path)
    if word_uuid not in user_dict:
        raise HTTPException(
            status_code=422, detail="IDに該当するワードが見つかりませんでした"
        )
    del user_dict[word_uuid]
    write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)


def import_user_dict(
    dict_data: dict[str, UserDictWord],
    override: bool = False,
    user_dict_path: Path = user_dict_path,
    default_dict_path: Path = default_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    # 念のため型チェックを行う
    for word_uuid, word in dict_data.items():
        UUID(word_uuid)
        assert type(word) is UserDictWord
        for pos_detail in part_of_speech_data.values():
            if word.context_id == pos_detail.context_id:
                assert word.part_of_speech == pos_detail.part_of_speech
                assert (
                    word.part_of_speech_detail_1 == pos_detail.part_of_speech_detail_1
                )
                assert (
                    word.part_of_speech_detail_2 == pos_detail.part_of_speech_detail_2
                )
                assert (
                    word.part_of_speech_detail_3 == pos_detail.part_of_speech_detail_3
                )
                assert (
                    word.accent_associative_rule in pos_detail.accent_associative_rules
                )
                break
        else:
            raise ValueError("対応していない品詞です")
    old_dict = read_dict(user_dict_path=user_dict_path)
    if override:
        new_dict = {**old_dict, **dict_data}
    else:
        new_dict = {**dict_data, **old_dict}
    write_to_json(user_dict=new_dict, user_dict_path=user_dict_path)
    update_dict(
        default_dict_path=default_dict_path,
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_dict_path,
    )


def search_cost_candidates(context_id: int) -> list[int]:
    for value in part_of_speech_data.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise HTTPException(status_code=422, detail="品詞IDが不正です")


def cost2priority(context_id: int, cost: int) -> int:
    cost_candidates = search_cost_candidates(context_id)
    # cost_candidatesの中にある値で最も近い値を元にpriorityを返す
    # 参考: https://qiita.com/Krypf/items/2eada91c37161d17621d
    # この関数とpriority2cost関数によって、辞書ファイルのcostを操作しても最も近いpriorityのcostに上書きされる
    return MAX_PRIORITY - np.argmin(np.abs(np.array(cost_candidates) - cost))


def priority2cost(context_id: int, priority: int) -> int:
    cost_candidates = search_cost_candidates(context_id)
    return cost_candidates[MAX_PRIORITY - priority]
