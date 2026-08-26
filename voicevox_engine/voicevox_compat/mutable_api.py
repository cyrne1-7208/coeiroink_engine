"""VOICEVOX互換の可変APIを起動設定で無効化する。"""

import os
import warnings
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException


def boolean_from_env(env_name: str) -> bool:
    """VOICEVOXと同じく、空文字と0を偽、1を真として環境変数を読む。"""

    value = os.getenv(env_name, "")
    if value == "1":
        return True
    if value not in ("", "0"):
        warnings.warn(
            f"Invalid environment variable value: {env_name}={value}",
            stacklevel=2,
        )
    return False


def mutability_guard(disabled: bool) -> Callable[[], Coroutine[Any, Any, None]]:
    """FastAPI dependencyとして利用する可変APIの共通ガードを返す。"""

    async def verify() -> None:
        if disabled:
            raise HTTPException(
                status_code=403,
                detail="エンジンの静的なデータを変更するAPIは無効化されています",
            )

    return verify
