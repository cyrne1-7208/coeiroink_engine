import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

URL_TIMEOUT_SECONDS = 30
UNKNOWN_LICENSE = "UNKNOWN"
LEGAL_FILE_PATTERN = re.compile(
    r"^(?:licen[cs]e|copying|notice)(?:$|[._-].*)", re.IGNORECASE
)

# パッケージ側のメタデータが空でも、一次配布元から識別できるものだけSPDX名を補う。
KNOWN_LICENSE_NAMES = {
    "coeiroink-engine": "LGPL-3.0-only",
    "coeirocore": "LGPL-3.0-only",
    "coeiroink-opencl": "LGPL-3.0-only",
    "kaldiio": "LGPL-3.0-only",
    "nvidia-nvshmem-cu12": "LicenseRef-NVIDIA-NVSHMEM",
    "pyworld": "MIT",
    "sentencepiece": "Apache-2.0",
    "sudachidict-full": "Apache-2.0",
    "sudachipy": "Apache-2.0",
    "torch-complex": "Apache-2.0",
    "torch-directml": "MIT",
    "triton": "MIT",
}


@dataclass
class License:
    name: str
    version: str | None
    license: str | None
    text: str


class LicenseGenerationError(RuntimeError):
    """ライセンス一覧の生成に失敗した場合に発生するエラー。"""


def _canonicalize_package_name(name: str) -> str:
    """PEP 503と同じ規則で依存名を比較する。"""

    return re.sub(r"[-_.]+", "-", name).lower()


def _read_package_snapshot(path: Path) -> dict[str, tuple[str, str]]:
    """`uv pip list --format json`で保存した実行時依存を読み込む。"""

    try:
        raw_packages: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LicenseGenerationError(
            f"Could not read runtime package snapshot: {path}"
        ) from error
    if not isinstance(raw_packages, list):
        raise LicenseGenerationError("Runtime package snapshot must be a JSON list")

    packages: dict[str, tuple[str, str]] = {}
    for package in raw_packages:
        if not isinstance(package, dict):
            raise LicenseGenerationError("Invalid package entry in runtime snapshot")
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise LicenseGenerationError(
                "Every runtime package must have string name and version fields"
            )
        canonical_name = _canonicalize_package_name(name)
        previous = packages.get(canonical_name)
        if previous is not None and previous[1] != version:
            raise LicenseGenerationError(
                f"Conflicting versions in runtime snapshot: {name} {previous[1]} / {version}"
            )
        packages[canonical_name] = (name, version)
    if not packages:
        raise LicenseGenerationError("Runtime package snapshot is empty")
    return packages


def _read_url(url: str) -> str:
    """ネットワーク障害で生成処理が無期限に停止しないよう、上限時間付きで本文を取得する。"""

    with urllib.request.urlopen(url, timeout=URL_TIMEOUT_SECONDS) as response:
        return response.read().decode()


def _unknown_license_text(name: str, version: str | None) -> str:
    """パッケージメタデータに本文がない既知の依存だけ、一次配布元から補完する。"""

    normalized_name = _canonicalize_package_name(name)
    if normalized_name == "torch-complex":
        return _combine_legal_documents(
            [
                (
                    "licenses/Apache-2.0.txt",
                    Path("licenses/Apache-2.0.txt").read_text(encoding="utf-8"),
                ),
                (
                    "licenses/torch_complex/ATTRIBUTION.txt",
                    Path("licenses/torch_complex/ATTRIBUTION.txt").read_text(
                        encoding="utf-8"
                    ),
                ),
            ]
        )
    if normalized_name == "coeirocore":
        local_license = Path("../coeiroink_core/LICENSE")
        if local_license.is_file():
            return local_license.read_text(encoding="utf-8")
        return _read_url(
            "https://raw.githubusercontent.com/cyrne1-7208/coeiroink_core/main/LICENSE"
        )
    if normalized_name == "kaldiio" and version is not None:
        # 評価用途に限定された上流kaldiioを公開配布物へ混入させないため、ローカル互換ガード以外は生成時に拒否する。
        if not version.endswith("+coeiroink.guard1"):
            raise LicenseGenerationError(
                f"Unexpected external kaldiio distribution: {version}"
            )
        local_license = Path("../coeiroink_core/LICENSE")
        if local_license.is_file():
            return local_license.read_text(encoding="utf-8")
        return _read_url(
            "https://raw.githubusercontent.com/cyrne1-7208/coeiroink_core/main/LICENSE"
        )

    fixed_urls = {
        "future": "https://raw.githubusercontent.com/PythonCharmers/python-future/master/LICENSE.txt",
        "nvidia-nvshmem-cu12": "https://raw.githubusercontent.com/NVIDIA/nvshmem/131da55f643ac87c810ba0bc51d359258bf433a1/License.txt",
        "pefile": "https://raw.githubusercontent.com/erocarrera/pefile/master/LICENSE",
        "pyopenjtalk": "https://raw.githubusercontent.com/r9y9/pyopenjtalk/v0.4.1/LICENSE.md",
        "tensorboard-data-server": "https://raw.githubusercontent.com/tensorflow/tensorboard/master/LICENSE",
        "python-multipart": "https://raw.githubusercontent.com/Kludex/python-multipart/master/LICENSE.txt",
        "romkan": "https://raw.githubusercontent.com/soimort/python-romkan/master/LICENSE",
        "distlib": "https://bitbucket.org/pypa/distlib/raw/7d93712134b28401407da27382f2b6236c87623a/LICENSE.txt",
        "jsonschema": "https://raw.githubusercontent.com/python-jsonschema/jsonschema/dbc398245a583cb2366795dc529ae042d10c1577/COPYING",
        "lockfile": "https://opendev.org/openstack/pylockfile/raw/tag/0.12.2/LICENSE",
        "platformdirs": "https://raw.githubusercontent.com/platformdirs/platformdirs/aa671aaa97913c7b948567f4d9c77d4f98bfa134/LICENSE",
        "webencodings": "https://raw.githubusercontent.com/gsnedders/python-webencodings/fa2cb5d75ab41e63ace691bc0825d3432ba7d694/LICENSE",
        "espnet": "https://raw.githubusercontent.com/espnet/espnet/v.202604-patch1/LICENSE",
        "sentencepiece": "https://raw.githubusercontent.com/google/sentencepiece/v0.2.1/LICENSE",
        "sudachipy": "https://raw.githubusercontent.com/WorksApplications/sudachi.rs/90fd6068c80c2fc3b63e0dbab0e341475bad4d8f/LICENSE",
        "torch-directml": "https://raw.githubusercontent.com/microsoft/DirectML/8700779fe7a09ea7a007cf3d7ab4293c78e41017/LICENSE",
        "triton": "https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/LICENSE",
    }
    versioned_urls = {
        "cython": "https://raw.githubusercontent.com/cython/cython/{version}/LICENSE.txt",
        "antlr4-python3-runtime": "https://raw.githubusercontent.com/antlr/antlr4/{version}/LICENSE.txt",
        "pyworld": "https://raw.githubusercontent.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/v{version}/LICENSE",
    }

    if normalized_name in fixed_urls:
        return _read_url(fixed_urls[normalized_name])
    if normalized_name in versioned_urls and version is not None:
        return _read_url(versioned_urls[normalized_name].format(version=version))
    raise LicenseGenerationError(
        f"No license text provided for {name} {version or ''}".rstrip()
    )


def _bundled_licenses() -> list[License]:
    """Pythonパッケージのメタデータだけでは表現できない同梱物のライセンスを返す。"""

    python_version = ".".join(
        str(component)
        for component in (
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        )
    )
    return [
        License(
            name="GNU General Public License",
            version="3.0",
            license="GPL-3.0-only",
            text=Path("licenses/GPL-3.0.txt").read_text(encoding="utf-8"),
        ),
        License(
            name="Open JTalk",
            version="1.11",
            license="Modified BSD license",
            text=Path("licenses/open_jtalk/COPYING").read_text(encoding="utf-8"),
        ),
        License(
            name="MeCab",
            version=None,
            license="Modified BSD license",
            text=Path("licenses/open_jtalk/mecab/COPYING").read_text(encoding="utf-8"),
        ),
        License(
            name="NAIST Japanese Dictionary",
            version=None,
            license="Modified BSD license",
            text=Path("licenses/open_jtalk/mecab-naist-jdic/COPYING").read_text(
                encoding="utf-8"
            ),
        ),
        License(
            name='HTS Voice "Mei"',
            version=None,
            license="Creative Commons Attribution 3.0 license",
            text=_read_url(
                "https://raw.githubusercontent.com/r9y9/pyopenjtalk/v0.4.1/pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice"
            ),
        ),
        License(
            name="VOICEVOX",
            version="0.14.5-modified-by-shirowanisan",
            license="LGPL license",
            text=_read_url(
                "https://raw.githubusercontent.com/VOICEVOX/voicevox/main/LGPL_LICENSE"
            ),
        ),
        License(
            name="VOICEVOX ENGINE",
            version="0.14.3-modified-by-shirowanisan",
            license="LGPL license",
            text=_read_url(
                "https://raw.githubusercontent.com/VOICEVOX/voicevox_engine/master/LGPL_LICENSE"
            ),
        ),
        License(
            name="world",
            version=None,
            license="Modified BSD license",
            text=_read_url(
                "https://raw.githubusercontent.com/mmorise/World/master/LICENSE.txt"
            ),
        ),
        License(
            name="Python",
            version=python_version,
            license="Python Software Foundation License",
            text=_read_url(
                f"https://raw.githubusercontent.com/python/cpython/v{python_version}/LICENSE"
            ),
        ),
    ]


def _legal_documents(package_name: str) -> list[tuple[str, str]]:
    """wheelまたはeditable metadataに含まれる全ライセンス・NOTICE本文を返す。"""

    try:
        distribution = metadata.distribution(package_name)
    except metadata.PackageNotFoundError as error:
        raise LicenseGenerationError(
            f"Installed distribution metadata not found: {package_name}"
        ) from error

    documents: list[tuple[str, str]] = []
    seen_texts: set[str] = set()
    for relative_path in distribution.files or ():
        if LEGAL_FILE_PATTERN.fullmatch(relative_path.name) is None:
            continue
        absolute_path = Path(distribution.locate_file(relative_path))
        if not absolute_path.is_file():
            continue
        text = absolute_path.read_text(encoding="utf-8", errors="backslashreplace")
        if not text.strip() or text in seen_texts:
            continue
        seen_texts.add(text)
        documents.append((str(relative_path), text))
    return sorted(documents, key=lambda item: item[0].lower())


def _combine_legal_documents(documents: list[tuple[str, str]]) -> str:
    """複数の法的文書を改変せず、元ファイル名が分かる区切りだけ付けて収録する。"""

    if len(documents) == 1:
        return documents[0][1]
    return "\n\n".join(f"===== {path} =====\n{text}" for path, text in documents)


def _python_package_licenses(
    runtime_packages: dict[str, tuple[str, str]] | None = None,
) -> list[License]:
    command = [
        sys.executable,
        "-m",
        "piplicenses",
        "--from=mixed",
        "--format=json",
        "--with-urls",
    ]
    if runtime_packages is not None:
        # pip-licenses自身など生成専用ツールを除き、配布する実行時依存だけを列挙する。
        command.extend(
            [
                "--with-system",
                "--packages",
                *(package[0] for package in runtime_packages.values()),
            ]
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise LicenseGenerationError(f"pip-licenses failed:\n{error.stderr}") from error

    package_rows = json.loads(completed.stdout)
    if not isinstance(package_rows, list):
        raise LicenseGenerationError("pip-licenses returned invalid JSON")

    licenses: list[License] = []
    seen_package_keys: set[tuple[str, str]] = set()
    generated_packages: set[str] = set()
    for package in package_rows:
        canonical_name = _canonicalize_package_name(package["Name"])
        package_key = (canonical_name, package["Version"])
        if package_key in seen_package_keys:
            continue
        seen_package_keys.add(package_key)
        generated_packages.add(canonical_name)

        if runtime_packages is not None:
            expected = runtime_packages.get(canonical_name)
            if expected is None:
                raise LicenseGenerationError(
                    f"Unexpected package in license output: {package['Name']}"
                )
            if expected[1] != package["Version"]:
                raise LicenseGenerationError(
                    f"Runtime/license version mismatch for {package['Name']}: "
                    f"{expected[1]} != {package['Version']}"
                )

        documents = _legal_documents(package["Name"])
        if documents:
            license_text = _combine_legal_documents(documents)
        else:
            license_text = _unknown_license_text(package["Name"], package["Version"])
        license_name = package["License"]
        if canonical_name in KNOWN_LICENSE_NAMES:
            license_name = KNOWN_LICENSE_NAMES[canonical_name]
        if not license_name or license_name == UNKNOWN_LICENSE:
            raise LicenseGenerationError(
                f"No license identifier provided for {package['Name']} {package['Version']}"
            )
        licenses.append(
            License(
                name=package["Name"],
                version=package["Version"],
                license=license_name,
                text=license_text,
            )
        )

    if runtime_packages is not None:
        missing_packages = sorted(set(runtime_packages) - generated_packages)
        if missing_packages:
            details = ", ".join(
                f"{runtime_packages[name][0]} {runtime_packages[name][1]}"
                for name in missing_packages
            )
            raise LicenseGenerationError(
                f"No license metadata generated for runtime packages: {details}"
            )
    return licenses


def generate_licenses(
    runtime_packages: dict[str, tuple[str, str]] | None = None,
) -> list[License]:
    return [*_bundled_licenses(), *_python_package_licenses(runtime_packages)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=Path)
    parser.add_argument(
        "--package-snapshot",
        type=Path,
        help="JSON produced by `uv pip list --format json` before license tooling is installed",
    )
    args = parser.parse_args()
    runtime_packages = (
        _read_package_snapshot(args.package_snapshot)
        if args.package_snapshot is not None
        else None
    )
    serialized = [asdict(license) for license in generate_licenses(runtime_packages)]

    if args.output_path is None:
        json.dump(serialized, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output:
        json.dump(serialized, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
