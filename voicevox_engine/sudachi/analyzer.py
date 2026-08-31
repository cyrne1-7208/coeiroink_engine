"""Sudachi full辞書を明示的に利用するための独立した解析API。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self

from .open_jtalk_dictionary import compile_open_jtalk_dictionaries


class SudachiError(RuntimeError):
    """Sudachi解析で発生したエラー。"""


class SudachiUnavailableError(SudachiError):
    """Sudachiの任意依存がインストールされていない。"""


@dataclass(frozen=True, slots=True)
class SudachiMorpheme:
    """Sudachiの形態素をPythonの値として保持する。"""

    surface: str
    normalized_form: str
    dictionary_form: str
    reading_form: str
    part_of_speech: tuple[str, ...]
    begin: int
    end: int
    dictionary_id: int
    is_oov: bool

    def to_dict(self) -> dict[str, Any]:
        """CLIや外部検証で扱いやすい辞書へ変換する。"""
        return {
            "surface": self.surface,
            "normalized_form": self.normalized_form,
            "dictionary_form": self.dictionary_form,
            "reading_form": self.reading_form,
            "part_of_speech": list(self.part_of_speech),
            "begin": self.begin,
            "end": self.end,
            "dictionary_id": self.dictionary_id,
            "is_oov": self.is_oov,
        }


def _load_sudachipy() -> Any:
    """任意依存を必要な時だけ読み込む。"""
    try:
        import sudachipy
    except ModuleNotFoundError as error:
        if error.name != "sudachipy":
            raise
        raise SudachiUnavailableError(
            "Sudachiは未インストールです。"
            " `uv sync --locked --extra sudachi` を実行してください。"
        ) from error
    return sudachipy


class SudachiAnalyzer:
    """Sudachi full辞書を一度ロードして繰り返し解析する。"""

    _MODES = frozenset({"A", "B", "C"})

    def __init__(
        self,
        mode: str = "C",
        *,
        open_jtalk_csv_paths: Sequence[Path] = (),
        open_jtalk_user_dict_path: Path | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of A, B, C: {mode!r}")

        sudachipy = _load_sudachipy()
        self.mode = mode
        self.dictionary_type = "full"
        self._resources = ExitStack()
        self._dictionary: Any | None = None
        try:
            if open_jtalk_csv_paths or open_jtalk_user_dict_path is not None:
                system_resource = files("sudachidict_full").joinpath(
                    "resources", "system.dic"
                )
                system_dictionary = self._resources.enter_context(
                    as_file(system_resource)
                )
                temporary_directory = TemporaryDirectory(prefix="coeiroink-sudachi-")
                self._resources.callback(temporary_directory.cleanup)
                user_dictionaries = compile_open_jtalk_dictionaries(
                    sudachipy,
                    system_dictionary=system_dictionary,
                    output_directory=Path(temporary_directory.name),
                    csv_paths=open_jtalk_csv_paths,
                    user_json_path=open_jtalk_user_dict_path,
                )
                self._dictionary = sudachipy.Dictionary(
                    config=sudachipy.Config(
                        system=str(system_dictionary),
                        user=[str(path) for path in user_dictionaries],
                    )
                )
            else:
                self._dictionary = sudachipy.Dictionary(dict="full")
                user_dictionaries = ()
            self.open_jtalk_dictionary_count = len(user_dictionaries)
            self._tokenizer = self._dictionary.create(
                mode=getattr(sudachipy.SplitMode, mode)
            )
        except Exception:
            if self._dictionary is not None:
                self._dictionary.close()
            self._resources.close()
            raise

    def tokenize(self, text: str) -> list[SudachiMorpheme]:
        """入力を解析し、元文字列上の形態素境界と属性を返す。"""
        if not isinstance(text, str):
            raise TypeError("text must be str")

        return [
            SudachiMorpheme(
                surface=morpheme.surface(),
                normalized_form=morpheme.normalized_form(),
                dictionary_form=morpheme.dictionary_form(),
                reading_form=morpheme.reading_form(),
                part_of_speech=tuple(morpheme.part_of_speech()),
                begin=morpheme.begin(),
                end=morpheme.end(),
                dictionary_id=morpheme.dictionary_id(),
                is_oov=morpheme.is_oov(),
            )
            for morpheme in self._tokenizer.tokenize(text)
        ]

    def close(self) -> None:
        """mmap辞書を明示的に閉じて保持資源を解放する。"""
        if self._dictionary is None:
            return
        try:
            self._dictionary.close()
        finally:
            self._dictionary = None
            self._resources.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.close()
