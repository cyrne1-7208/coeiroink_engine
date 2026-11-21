# リリース用の配布パッケージが単体で起動し、音声を合成できるか確認するスクリプト

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from make_ci_speaker import STYLE_ID, write_speaker

ENGINE_PORT = 50032
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
# 初回の合成ではESPnetとモデルの読み込みを待つ。
SYNTHESIS_TIMEOUT_SECONDS = 300


# 対象OSのEngine実行ファイルを返す。
def executable_path(dist_dir: Path) -> Path:
    for name in ("run", "run.exe"):
        candidate = dist_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Engine executable was not found in {dist_dir}")


# Engineが期待したバージョンを返すまで待ち、起動できたかを返す。
def wait_for_version(process: subprocess.Popen[bytes], expected_version: str) -> bool:
    for _ in range(30):
        if process.poll() is not None:
            return False
        with (
            suppress(OSError, URLError, TimeoutError),
            urlopen(f"{ENGINE_URL}/voicevox/version", timeout=2) as response,
        ):
            if json.loads(response.read()) == expected_version:
                return True
        time.sleep(2)
    return False


# テスト用の話者でaudio_queryとsynthesisを呼び、失敗した場合は理由を返す。
def synthesis_failure() -> str | None:
    query_url = f"{ENGINE_URL}/voicevox/audio_query?" + urlencode(
        {"text": "こんにちは", "speaker": STYLE_ID}
    )
    synthesis_url = f"{ENGINE_URL}/voicevox/synthesis?" + urlencode(
        {"speaker": STYLE_ID}
    )
    try:
        with urlopen(
            Request(query_url, method="POST"), timeout=SYNTHESIS_TIMEOUT_SECONDS
        ) as response:
            query = response.read()
        request = Request(
            synthesis_url,
            data=query,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=SYNTHESIS_TIMEOUT_SECONDS) as response:
            wave = response.read()
    except OSError as error:
        return f"Packaged Engine could not synthesize speech: {error}"
    if not wave.startswith(b"RIFF"):
        return f"Packaged Engine did not return a WAV file: {wave[:64]!r}"
    return None


# マニフェスト、HTTP起動、テスト用の話者での合成を検査する。
def check_release_build(
    dist_dir: Path,
    expected_version: str,
    *,
    experimental_voice_smoothing: bool = False,
) -> None:
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
        write_speaker(speaker_info_dir)
        with log_path.open("w", encoding="utf-8") as log:
            command = [
                str(executable),
                "--device",
                "cpu",
                "--host",
                "127.0.0.1",
                "--port",
                str(ENGINE_PORT),
                "--speaker_info_dir",
                str(speaker_info_dir),
                "--output_log_utf8",
            ]
            if experimental_voice_smoothing:
                command.extend(("--experimental", "voice-smoothing"))
            process = subprocess.Popen(
                command,
                cwd=dist_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                failure = "Packaged Engine did not start successfully"
                if wait_for_version(process, expected_version):
                    failure = synthesis_failure()
                    if failure is None:
                        return
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"{failure}:\n{log_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--experimental-voice-smoothing", action="store_true")
    args = parser.parse_args()
    check_release_build(
        args.dist_dir,
        args.expected_version,
        experimental_voice_smoothing=args.experimental_voice_smoothing,
    )


if __name__ == "__main__":
    main()
