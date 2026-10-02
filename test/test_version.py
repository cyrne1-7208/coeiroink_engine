import json
import tomllib
from pathlib import Path

import voicevox_engine

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_toml(path: Path) -> dict:
    with path.open("rb") as toml_file:
        return tomllib.load(toml_file)


def _locked_versions(lock: dict, package_name: str) -> set[str]:
    return {
        package["version"]
        for package in lock["package"]
        if package["name"] == package_name and "version" in package
    }


def _pinned_versions(project: dict, package_name: str) -> set[str]:
    requirement_groups = [
        project["project"]["dependencies"],
        *project["project"]["optional-dependencies"].values(),
    ]
    prefix = f"{package_name}=="
    return {
        requirement.removeprefix(prefix).split(";", 1)[0]
        for requirements in requirement_groups
        for requirement in requirements
        if requirement.startswith(prefix)
    }


def test_engine_version_sources_match() -> None:
    """配布物とAPIで公開するEngineバージョンの同期漏れを検出する。"""

    project = _load_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _load_toml(REPOSITORY_ROOT / "uv.lock")
    manifest = json.loads(
        (REPOSITORY_ROOT / "engine_manifest.json").read_text(encoding="utf-8")
    )
    version = project["project"]["version"]

    assert voicevox_engine.__version__ == version
    assert manifest["version"] == version
    assert _locked_versions(lock, "coeiroink-engine") == {version}


def test_core_dependency_versions_match_engine() -> None:
    """Engine、隣接Core、OpenCL拡張、lockの片側だけが更新される状態を防ぐ。"""

    project = _load_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _load_toml(REPOSITORY_ROOT / "uv.lock")
    version = project["project"]["version"]
    opencl_metadata = next(
        metadata
        for metadata in project["tool"]["uv"]["dependency-metadata"]
        if metadata["name"] == "coeiroink-opencl"
    )

    assert _pinned_versions(project, "coeirocore") == {version}
    assert _pinned_versions(project, "coeiroink-opencl") == {version}
    assert opencl_metadata["version"] == version
    assert _locked_versions(lock, "coeirocore") == {version}
    assert _locked_versions(lock, "coeiroink-opencl") == {version}

    adjacent_core = REPOSITORY_ROOT.parent / "coeiroink_core"
    if adjacent_core.is_dir():
        assert (
            _load_toml(adjacent_core / "pyproject.toml")["project"]["version"]
            == version
        )
        assert (
            _load_toml(adjacent_core / "native" / "opencl" / "pyproject.toml")[
                "project"
            ]["version"]
            == version
        )
