"""プリセットの永続化と検証を管理する。"""

from pathlib import Path
from threading import RLock

import yaml
from pydantic import TypeAdapter, ValidationError

from ..utility import atomic_write_text
from .preset import Preset
from .preset_error import PresetError

_PRESET_LOCK = RLock()


def _atomic_write_yaml(path: Path, data: object) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    )


class PresetManager:
    def __init__(
        self,
        preset_path: Path,
    ):
        self.presets = []
        self.last_modified_time = 0
        self.preset_path = Path(preset_path)

    def load_presets(self):
        """
        プリセットのYAMLファイルを読み込む

        Returns
        -------
        ret: List[Preset]
            プリセットのリスト
        """

        with _PRESET_LOCK:
            # 外部編集がなければ検証済みのプリセットを再利用する。
            try:
                _last_modified_time = self.preset_path.stat().st_mtime_ns
                if _last_modified_time == self.last_modified_time:
                    return self.presets
            except OSError as error:
                raise PresetError("プリセットの設定ファイルが見つかりません") from error

            with self.preset_path.open(encoding="utf-8") as file:
                obj = yaml.safe_load(file)
                if obj is None:
                    raise PresetError("プリセットの設定ファイルの内容が空です")

            try:
                _presets = TypeAdapter(list[Preset]).validate_python(obj)
            except ValidationError as error:
                raise PresetError(
                    "プリセットの設定ファイルの形式に誤りがあります"
                ) from error

            if len(_presets) != len({preset.id for preset in _presets}):
                raise PresetError("プリセットIDに重複があります")

            self.presets = _presets
            self.last_modified_time = _last_modified_time
            return self.presets

    def _write_presets(self) -> None:
        _atomic_write_yaml(
            self.preset_path,
            [stored_preset.model_dump() for stored_preset in self.presets],
        )

    def add_preset(self, preset: Preset):
        """プリセットを追加してYAMLファイルへ永続化し、登録されたIDを返す。

        IDが負数または既存IDと重複する場合は、上書きを防ぐため`preset.id`を未使用IDへ書き換える。

        Parameters
        ----------
        preset : Preset
            追加するプリセット

        Returns
        -------
        int
            登録されたプリセットID
        """

        with _PRESET_LOCK:
            # 手動でファイルが更新されているかもしれないので、最新のYAMLファイルを読み直す
            self.load_presets()

            # 負のIDと重複IDは自動採番し、既存プリセットの上書きを防ぐ。
            preset_ids = {stored_preset.id for stored_preset in self.presets}
            if preset.id < 0 or preset.id in preset_ids:
                preset.id = max(preset_ids, default=0) + 1
            self.presets.append(preset)

            try:
                self._write_presets()
            except Exception as err:
                self.presets.pop()
                if isinstance(err, FileNotFoundError):
                    raise PresetError(
                        "プリセットの設定ファイルへの書き込みに失敗しました"
                    ) from err
                raise

            return preset.id

    def update_preset(self, preset: Preset):
        """
        YAMLファイルのプリセットを更新する

        Parameters
        ----------
        preset : Preset
            更新するプリセットを渡す

        Returns
        -------
        ret: int
            更新したプリセットのプリセットID
        """

        with _PRESET_LOCK:
            # 手動でファイルが更新されているかもしれないので、最新のYAMLファイルを読み直す
            self.load_presets()

            prev_preset = (-1, None)
            for i in range(len(self.presets)):
                if self.presets[i].id == preset.id:
                    prev_preset = (i, self.presets[i])
                    self.presets[i] = preset
                    break
            else:
                raise PresetError("更新先のプリセットが存在しません")

            try:
                self._write_presets()
            except Exception as err:
                if prev_preset != (-1, None):
                    self.presets[prev_preset[0]] = prev_preset[1]
                if isinstance(err, FileNotFoundError):
                    raise PresetError(
                        "プリセットの設定ファイルへの書き込みに失敗しました"
                    ) from err
                raise

            return preset.id

    def delete_preset(self, id: int):
        """
        YAMLファイルのプリセットを削除する

        Parameters
        ----------
        id: int
            削除するプリセットのプリセットIDを渡す

        Returns
        -------
        ret: int
            削除したプリセットのプリセットID
        """

        with _PRESET_LOCK:
            # 手動でファイルが更新されているかもしれないので、最新のYAMLファイルを読み直す
            self.load_presets()

            buf = None
            buf_index = -1
            for i in range(len(self.presets)):
                if self.presets[i].id == id:
                    buf = self.presets.pop(i)
                    buf_index = i
                    break
            else:
                raise PresetError("削除対象のプリセットが存在しません")

            try:
                self._write_presets()
            except Exception as err:
                self.presets.insert(buf_index, buf)
                if isinstance(err, FileNotFoundError):
                    raise PresetError(
                        "プリセットの設定ファイルへの書き込みに失敗しました"
                    ) from err
                raise

            return id
