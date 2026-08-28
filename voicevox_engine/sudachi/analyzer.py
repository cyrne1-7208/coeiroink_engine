"""Sudachi full辞書を明示的に利用するための独立した解析API。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self


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

    def __init__(self, mode: str = "C") -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of A, B, C: {mode!r}")

        sudachipy = _load_sudachipy()
        self.mode = mode
        self.dictionary_type = "full"
        self._dictionary = sudachipy.Dictionary(dict="full")
        self._tokenizer = self._dictionary.create(
            mode=getattr(sudachipy.SplitMode, mode)
        )

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
        self._dictionary.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
