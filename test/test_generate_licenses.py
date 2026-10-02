from email.message import Message
from pathlib import Path

import pytest

import generate_licenses


class _Distribution:
    def __init__(self, root: Path, files: list[Path], declared: list[str]):
        self.root = root
        self.files = files
        self.metadata = Message()
        for path in declared:
            self.metadata["License-File"] = path

    def locate_file(self, path: Path) -> Path:
        return self.root / path


def test_legal_documents_include_metadata_and_conventional_files(tmp_path, monkeypatch):
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


def test_legal_documents_reject_missing_metadata_file(tmp_path, monkeypatch):
    distribution = _Distribution(tmp_path, [], ["LICENSE"])
    monkeypatch.setattr(
        generate_licenses.metadata, "distribution", lambda _name: distribution
    )

    with pytest.raises(
        generate_licenses.LicenseGenerationError,
        match="License-File metadata points to missing files",
    ):
        generate_licenses._legal_documents("demo")
