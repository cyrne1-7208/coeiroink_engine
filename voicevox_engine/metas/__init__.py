import sys

from . import metas as Metas
from . import metas_store as MetasStore

# 公開済みのモジュール名は互換エイリアスとして維持し、新規コードはsnake_caseを使う。
sys.modules[f"{__name__}.Metas"] = Metas
sys.modules[f"{__name__}.MetasStore"] = MetasStore

__all__ = [
    "Metas",
    "MetasStore",
]
