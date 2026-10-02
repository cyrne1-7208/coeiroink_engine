import os
import sys
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from platformdirs import user_data_dir


def engine_root() -> Path:
    if is_development():
        root_dir = Path(__file__).parents[2]
    else:
        root_dir = Path(sys.argv[0]).parent

    return root_dir.resolve(strict=True)


def is_development() -> bool:
    """凍結環境（PyInstallerまたはNuitka）でなければ開発環境と判定する。"""
    return not ("__compiled__" in globals() or getattr(sys, "frozen", False))


def get_save_dir():
    app_name = "coeiroink-engine-dev" if is_development() else "coeiroink-engine"
    return Path(user_data_dir(app_name))


def delete_file(file_path: str) -> None:
    try:
        os.remove(file_path)
    except OSError:
        traceback.print_exc()


def atomic_write_text(path: Path, content: str) -> None:
    """同じディレクトリの一時ファイルを同期してから、既存テキストを原子的に置換する。"""

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
