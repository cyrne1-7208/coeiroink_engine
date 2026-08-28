"""COEIROINK本体から分離したSudachi full辞書解析機能。"""

from .analyzer import (
    SudachiAnalyzer,
    SudachiError,
    SudachiMorpheme,
    SudachiUnavailableError,
)

__all__ = [
    "SudachiAnalyzer",
    "SudachiError",
    "SudachiMorpheme",
    "SudachiUnavailableError",
]
