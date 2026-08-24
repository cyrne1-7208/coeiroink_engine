"""依存込みRelease成果物を外部Python環境に頼らず起動確認する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def executable_path(dist_dir: Path) -> Path:
    """対象OSのEngine実行ファイルを返す。"""

    for name in ("run", "run.exe"):
        candidate = dist_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Engine executable was not found in {dist_dir}")


def check_release_build(dist_dir: Path, expected_version: str) -> None:
    """マニフェストとHTTP起動を検査する。"""

    dist_dir = dist_dir.resolve(strict=True)
    executable = executable_path(dist_dir)
    manifest = json.loads(
        (dist_dir / "engine_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["version"] != expected_version:
        raise AssertionError(
            f"manifest version {manifest['version']!r} != {expected_version!r}"
        )
    if not (dist_dir / manifest["command"]).is_file():
        raise FileNotFoundError(
            f"Manifest command was not found: {manifest['command']}"
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        log_path = temporary_path / "engine.log"
        speaker_info_dir = temporary_path / "speaker_info"
        speaker_info_dir.mkdir()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(executable),
                    "--enable_mock",
                    "--device",
                    "cpu",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "50032",
                    "--speaker_info_dir",
                    str(speaker_info_dir),
                    "--output_log_utf8",
                ],
                cwd=dist_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(30):
                    if process.poll() is not None:
                        break
                    try:
                        with urlopen(
                            "http://127.0.0.1:50032/voicevox/version", timeout=2
                        ) as response:
                            if json.loads(response.read()) == expected_version:
                                return
                    except (OSError, URLError, TimeoutError):
                        pass
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"Packaged Engine did not start successfully:\n{log_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    check_release_build(args.dist_dir, args.expected_version)


if __name__ == "__main__":
    main()
