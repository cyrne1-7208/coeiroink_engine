"""依存パッケージのライセンス文書収集・例外規定・ソース同梱ハッシュ検証を確認する。"""

from __future__ import annotations

import hashlib
import json
from email.message import Message
from pathlib import Path

import pytest

import generate_licenses

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _Distribution:
    def __init__(self, root: Path, files: list[Path], declared: list[str]) -> None:
        self.root = root
        self.files = files
        self.metadata = Message()
        for path in declared:
            self.metadata["License-File"] = path

    def locate_file(self, path: Path) -> Path:
        return self.root / path


def test_legal_documents_include_metadata_and_conventional_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = [
        Path("demo-1.0.dist-info/licenses/LICENSE"),
        Path("demo-1.0.dist-info/licenses/AUTHORS.md"),
        Path("demo-1.0.dist-info/GPL-3.0.txt"),
        Path("demo/LICENSES.third-party"),
        Path("demo/COPYRIGHT"),
    ]
    for path in files:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"contents of {path.name}", encoding="utf-8")
    distribution = _Distribution(
        tmp_path,
        files,
        ["LICENSE", "AUTHORS.md", "licenses/GPL-3.0.txt"],
    )
    monkeypatch.setattr(
        generate_licenses.metadata, "distribution", lambda _name: distribution
    )

    documents = generate_licenses._legal_documents("demo")

    assert [path for path, _text in documents] == sorted(
        (str(path) for path in files), key=str.lower
    )


def test_legal_documents_reject_missing_metadata_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = _Distribution(tmp_path, [], ["LICENSE"])
    monkeypatch.setattr(
        generate_licenses.metadata, "distribution", lambda _name: distribution
    )

    with pytest.raises(
        generate_licenses.LicenseGenerationError,
        match="License-File metadata points to missing files",
    ):
        generate_licenses._legal_documents("demo")


def test_cuda_toolkit_uses_nvidia_cuda_runtime_eula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    license_path = Path("nvidia_cuda_runtime-13.0.96.dist-info/licenses/License.txt")
    target = tmp_path / license_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("NVIDIA CUDA Toolkit EULA", encoding="utf-8")

    toolkit_dist = _Distribution(tmp_path, [], [])
    runtime_dist = _Distribution(tmp_path, [license_path], ["License.txt"])

    def fake_distribution(name: str) -> _Distribution:
        if name == "cuda-toolkit":
            return toolkit_dist
        if name == "nvidia-cuda-runtime":
            return runtime_dist
        raise generate_licenses.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(generate_licenses.metadata, "distribution", fake_distribution)

    license_info = generate_licenses._package_license(
        {"Name": "cuda-toolkit", "Version": "13.0.3.0", "License": "UNKNOWN"},
        "cuda-toolkit",
    )

    assert license_info.license == "LicenseRef-NVIDIA-Proprietary"
    assert license_info.text == "NVIDIA CUDA Toolkit EULA"


def test_distance_notice_preserves_original_license(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    documents = generate_licenses._legal_documents("Distance")
    license_info = generate_licenses._package_license(
        {"Name": "Distance", "Version": "0.1.3", "License": "GPL"}, "distance"
    )

    assert "notices differ" in license_info.license
    assert all(text in license_info.text for _path, text in documents)
    assert "does not relicense Distance" in license_info.text
    assert (REPOSITORY_ROOT / "licenses" / "GPL-3.0.txt").read_text(
        encoding="utf-8"
    ) in license_info.text


@pytest.mark.parametrize("valid_hash", [True, False])
def test_collect_sources_uses_locked_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_hash: bool,
) -> None:
    # 小さなローカル配布物を使い、外部通信なしで同梱とハッシュ不一致時の停止を確認する。
    source = tmp_path / "library-1.0.tar.gz"
    source.write_bytes(b"source archive contents")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "uv.lock").write_text(
        "\n".join(
            [
                "[[package]]",
                'name = "library"',
                'version = "1.0"',
                'source = { registry = "https://pypi.org/simple" }',
                f'sdist = {{ url = {json.dumps(source.as_uri())}, hash = "sha256:{digest if valid_hash else "0" * 64}" }}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "bundled"
    licenses = [generate_licenses.License("library", "1.0", "GPL-3.0-only", "license")]

    if not valid_hash:
        with pytest.raises(
            generate_licenses.LicenseGenerationError, match="checksum mismatch"
        ):
            generate_licenses.collect_sources(licenses, destination)
        return

    generate_licenses.collect_sources(licenses, destination)
    assert (destination / source.name).read_bytes() == source.read_bytes()
    assert "[library-1.0.tar.gz]" in (destination / "README.md").read_text(
        encoding="utf-8"
    )
