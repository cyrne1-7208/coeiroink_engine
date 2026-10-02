"""COEIROINK本体から分離したSudachi full辞書解析機能。"""

from .analyzer import (
    SudachiAnalyzer,
    SudachiError,
    SudachiMorpheme,
    SudachiUnavailableError,
)
from .open_jtalk_dictionary import (
    OpenJTalkDictionaryError,
    OpenJTalkDictionaryWarning,
)

__all__ = [
    "OpenJTalkDictionaryError",
    "OpenJTalkDictionaryWarning",
    "SudachiAnalyzer",
    "SudachiError",
    "SudachiMorpheme",
    "SudachiUnavailableError",
]
