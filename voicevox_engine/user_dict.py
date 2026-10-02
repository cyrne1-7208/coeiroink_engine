import importlib
import json
import os
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

import numpy as np
import pyopenjtalk
from fastapi import HTTPException

from .model import UserDictWord, WordTypes
from .part_of_speech_data import MAX_PRIORITY, MIN_PRIORITY, part_of_speech_data
from .utility import engine_root, get_save_dir

root_dir = engine_root()
save_dir = get_save_dir()

if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

default_dict_path = root_dir / "default.csv"
user_dict_path = save_dir / "user_dict.json"
compiled_dict_path = save_dir / "user.dic"


# 更新処理はOpen JTalkの切替を先にロックし、その内側でJSONのread-modify-writeを行う。
# RLockにして公開関数の組み合わせからも同じロックを安全に再利用できるようにする。
mutex_user_dict = threading.RLock()
mutex_openjtalk_dict = threading.RLock()


def _create_user_dict(source_path: Path, compiled_path: Path) -> None:
    """pyopenjtalk 0.4.1の公開APIでユーザー辞書をコンパイルする。"""
    pyopenjtalk.mecab_dict_index(str(source_path), str(compiled_path))


def _set_user_dict(compiled_path: Path) -> None:
    """pyopenjtalk 0.4.1の公開APIでコンパイル済み辞書を適用する。"""
    pyopenjtalk.update_global_jtalk_with_user_dict(str(compiled_path))


def _reset_user_dict() -> None:
    if hasattr(pyopenjtalk, "unset_user_dict"):
        pyopenjtalk.unset_user_dict()
    else:
        # pyopenjtalk 0.4系には解除用公開APIがないため、非公開状態へ依存せずモジュール再読込で既定のOpenJTalkインスタンスを再生成する。
        importlib.reload(pyopenjtalk)


def reset_user_dict() -> None:
    """ユーザー辞書更新後のpyopenjtalkを既定辞書へ戻す。"""
    with mutex_openjtalk_dict:
        _reset_user_dict()


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_json_atomic(user_dict_json: str, user_dict_path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            delete=False,
            dir=user_dict_path.parent,
            prefix=f".{user_dict_path.name}.",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(user_dict_json)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        # 同じディレクトリ内で置換するため、読み手には旧JSONか新JSONだけが見える。
        os.replace(temporary_path, user_dict_path)
    finally:
        if temporary_path is not None:
            _remove_file(temporary_path)


def _write_bytes_atomic(content: bytes, path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            _remove_file(temporary_path)


def _serialize_user_dict(user_dict: dict[str, UserDictWord]) -> str:
    converted_user_dict = {}
    for word_uuid, word in user_dict.items():
        word_dict = word.model_dump()
        word_dict["cost"] = priority2cost(
            word_dict["context_id"], word_dict["priority"]
        )
        del word_dict["priority"]
        converted_user_dict[word_uuid] = word_dict
    return json.dumps(converted_user_dict, ensure_ascii=False)


def write_to_json(user_dict: dict[str, UserDictWord], user_dict_path: Path) -> None:
    with mutex_openjtalk_dict, mutex_user_dict:
        _write_json_atomic(
            _serialize_user_dict(user_dict),
            Path(user_dict_path),
        )


def _read_dict(user_dict_path: Path) -> dict[str, UserDictWord]:
    if not user_dict_path.is_file():
        return {}
    with user_dict_path.open(encoding="utf-8") as f:
        result = {}
        for word_uuid, word in json.load(f).items():
            # 0.12以前の辞書は固有名詞のcontext_idがハードコードされており保存データに含まれないため、cost2priority変換用に固有名詞のcontext_idを補完する。
            if word.get("context_id") is None:
                word["context_id"] = part_of_speech_data[
                    WordTypes.PROPER_NOUN
                ].context_id
            word["priority"] = cost2priority(word["context_id"], word["cost"])
            del word["cost"]
            result[str(UUID(word_uuid))] = UserDictWord(**word)

    return result


def read_dict(user_dict_path: Path = user_dict_path) -> dict[str, UserDictWord]:
    with mutex_user_dict:
        return _read_dict(Path(user_dict_path))


def _restore_openjtalk_dict(compiled_dict_path: Path, had_compiled_dict: bool) -> None:
    if had_compiled_dict:
        _set_user_dict(compiled_dict_path.resolve(strict=True))
    else:
        _reset_user_dict()


def _restore_user_dict_json(
    user_dict_path: Path, previous_user_dict_json: bytes | None
) -> None:
    if previous_user_dict_json is None:
        _remove_file(user_dict_path)
    else:
        _write_bytes_atomic(previous_user_dict_json, user_dict_path)


def _rebuild_and_apply(
    user_dict: dict[str, UserDictWord],
    default_dict_path: Path,
    user_dict_path: Path,
    compiled_dict_path: Path,
    persist_json: bool,
) -> None:
    """候補辞書をコンパイルし、Open JTalk・JSON・コンパイル済み辞書を一貫して更新する。

    いずれかの適用に失敗した場合は、永続データとOpen JTalkを更新前の状態へ戻す。
    """
    previous_user_dict_json = (
        user_dict_path.read_bytes()
        if persist_json and user_dict_path.is_file()
        else None
    )
    previous_compiled_dict = (
        compiled_dict_path.read_bytes() if compiled_dict_path.is_file() else None
    )
    temporary_source_path: Path | None = None
    temporary_compiled_path: Path | None = None
    try:
        with NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            delete=False,
            dir=compiled_dict_path.parent,
            prefix=".coeiroink-user-dict-source-",
        ) as source_file:
            temporary_source_path = Path(source_file.name)
            if not default_dict_path.is_file():
                raise FileNotFoundError(
                    f"default dictionary was not found: {default_dict_path}"
                )
            default_dict = default_dict_path.read_text(encoding="utf-8")
            if not default_dict.endswith("\n"):
                default_dict += "\n"
            source_file.write(default_dict)
            for word in user_dict.values():
                source_file.write(
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
        with NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=compiled_dict_path.parent,
            prefix=".coeiroink-user-dict-compiled-",
        ) as compiled_file:
            temporary_compiled_path = Path(compiled_file.name)
        _create_user_dict(temporary_source_path, temporary_compiled_path)
        if not temporary_compiled_path.is_file():
            raise RuntimeError("辞書のコンパイル時にエラーが発生しました。")

        try:
            # 候補辞書を検証後に一度解放し、WindowsでもOpen JTalkが開いているファイルを置換しないようにする。
            _set_user_dict(temporary_compiled_path.resolve(strict=True))
            _reset_user_dict()
            os.replace(temporary_compiled_path, compiled_dict_path)
            _set_user_dict(compiled_dict_path.resolve(strict=True))
            if persist_json:
                _write_json_atomic(
                    _serialize_user_dict(user_dict),
                    user_dict_path,
                )
        except Exception as update_error:
            rollback_errors: list[Exception] = []
            try:
                _reset_user_dict()
            except Exception as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
            try:
                if previous_compiled_dict is None:
                    _remove_file(compiled_dict_path)
                else:
                    _write_bytes_atomic(previous_compiled_dict, compiled_dict_path)
            except Exception as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
            if persist_json:
                try:
                    _restore_user_dict_json(user_dict_path, previous_user_dict_json)
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(rollback_error)
            try:
                _restore_openjtalk_dict(
                    compiled_dict_path,
                    previous_compiled_dict is not None,
                )
            except Exception as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    "辞書更新に失敗し、更新前の状態へ復元できませんでした。"
                ) from ExceptionGroup(
                    "辞書更新とロールバックで発生したエラー",
                    [update_error, *rollback_errors],
                )
            raise
    finally:
        if temporary_source_path is not None:
            _remove_file(temporary_source_path)
        if temporary_compiled_path is not None:
            _remove_file(temporary_compiled_path)


def update_dict(
    default_dict_path: Path = default_dict_path,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    """既定辞書とユーザー辞書を再構築し、成功後にOpen JTalkへ適用する。"""
    with mutex_openjtalk_dict, mutex_user_dict:
        user_dict_path = Path(user_dict_path)
        _rebuild_and_apply(
            _read_dict(user_dict_path),
            Path(default_dict_path),
            user_dict_path,
            Path(compiled_dict_path),
            persist_json=False,
        )


def replace_user_dict(
    user_dict: dict[str, UserDictWord],
    *,
    default_dict_path: Path = default_dict_path,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    """辞書全体の保存・コンパイル・適用を一つの更新処理として行う。"""

    with mutex_openjtalk_dict, mutex_user_dict:
        _rebuild_and_apply(
            user_dict,
            Path(default_dict_path),
            Path(user_dict_path),
            Path(compiled_dict_path),
            persist_json=True,
        )


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
    with mutex_openjtalk_dict, mutex_user_dict:
        user_dict_path = Path(user_dict_path)
        user_dict = _read_dict(user_dict_path)
        word_uuid = str(uuid4())
        user_dict[word_uuid] = word
        _rebuild_and_apply(
            user_dict,
            default_dict_path,
            user_dict_path,
            Path(compiled_dict_path),
            persist_json=True,
        )
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
    with mutex_openjtalk_dict, mutex_user_dict:
        user_dict_path = Path(user_dict_path)
        user_dict = _read_dict(user_dict_path)
        if word_uuid not in user_dict:
            raise HTTPException(
                status_code=422,
                detail="指定されたUUIDに該当する単語が見つかりませんでした",
            )
        user_dict[word_uuid] = word
        _rebuild_and_apply(
            user_dict,
            default_dict_path,
            user_dict_path,
            Path(compiled_dict_path),
            persist_json=True,
        )


def delete_word(
    word_uuid: str,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    with mutex_openjtalk_dict, mutex_user_dict:
        user_dict_path = Path(user_dict_path)
        user_dict = _read_dict(user_dict_path)
        if word_uuid not in user_dict:
            raise HTTPException(
                status_code=422,
                detail="指定されたUUIDに該当する単語が見つかりませんでした",
            )
        del user_dict[word_uuid]
        _rebuild_and_apply(
            user_dict,
            default_dict_path,
            user_dict_path,
            Path(compiled_dict_path),
            persist_json=True,
        )


def import_user_dict(
    dict_data: dict[str, UserDictWord],
    override: bool = False,
    user_dict_path: Path = user_dict_path,
    default_dict_path: Path = default_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
):
    # インポートデータはassertではなく、実行設定に依存しない例外で検証する。
    for word_uuid, word in dict_data.items():
        try:
            UUID(word_uuid)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"UUIDが不正です: {word_uuid}") from error
        if not isinstance(word, UserDictWord):
            raise TypeError("ユーザー辞書の単語はUserDictWordである必要があります")
        for pos_detail in part_of_speech_data.values():
            if word.context_id == pos_detail.context_id:
                if word.part_of_speech != pos_detail.part_of_speech:
                    raise ValueError("品詞が文脈IDと一致しません")
                if word.part_of_speech_detail_1 != pos_detail.part_of_speech_detail_1:
                    raise ValueError("品詞細分類1が文脈IDと一致しません")
                if word.part_of_speech_detail_2 != pos_detail.part_of_speech_detail_2:
                    raise ValueError("品詞細分類2が文脈IDと一致しません")
                if word.part_of_speech_detail_3 != pos_detail.part_of_speech_detail_3:
                    raise ValueError("品詞細分類3が文脈IDと一致しません")
                if (
                    word.accent_associative_rule
                    not in pos_detail.accent_associative_rules
                ):
                    raise ValueError("アクセント結合規則が不正です")
                break
        else:
            raise ValueError("対応していない品詞です")
    with mutex_openjtalk_dict, mutex_user_dict:
        user_dict_path = Path(user_dict_path)
        old_dict = _read_dict(user_dict_path)
        if override:
            new_dict = {**old_dict, **dict_data}
        else:
            new_dict = {**dict_data, **old_dict}
        _rebuild_and_apply(
            new_dict,
            Path(default_dict_path),
            user_dict_path,
            Path(compiled_dict_path),
            persist_json=True,
        )


def search_cost_candidates(context_id: int) -> list[int]:
    for value in part_of_speech_data.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise HTTPException(status_code=422, detail="文脈IDが不正です")


def cost2priority(context_id: int, cost: int) -> int:
    cost_candidates = search_cost_candidates(context_id)
    # cost_candidatesの中にある値で最も近い値を元にpriorityを返す
    # 参考: https://qiita.com/Krypf/items/2eada91c37161d17621d
    # この関数とpriority2cost関数によって、辞書ファイルのcostを操作しても最も近いpriorityのcostに上書きされる
    return MAX_PRIORITY - np.argmin(np.abs(np.array(cost_candidates) - cost))


def priority2cost(context_id: int, priority: int) -> int:
    cost_candidates = search_cost_candidates(context_id)
    return cost_candidates[MAX_PRIORITY - priority]
