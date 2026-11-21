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


# PyInstallerやNuitkaで凍結した環境でなければ開発環境と判定する。
def is_development() -> bool:
    return not ("__compiled__" in globals() or getattr(sys, "frozen", False))


def get_save_dir() -> Path:
    app_name = "coeiroink-engine-dev" if is_development() else "coeiroink-engine"
    return Path(user_data_dir(app_name))


def delete_file(file_path: str) -> None:
    try:
        os.remove(file_path)
    except OSError:
        traceback.print_exc()


# 同じディレクトリの一時ファイルを同期し、既存のファイルを原子的に置き換える。
def atomic_write_text(path: Path, content: str) -> None:
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
            temporary_path.unlink(missing_ok=True)
