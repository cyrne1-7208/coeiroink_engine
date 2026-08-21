import argparse
import json
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

URL_TIMEOUT_SECONDS = 30


@dataclass
class License:
    name: str
    version: str | None
    license: str | None
    text: str


class LicenseGenerationError(RuntimeError):
    """ライセンス一覧の生成に失敗した場合に発生するエラー。"""


def _read_url(url: str) -> str:
    """ネットワーク障害で生成処理が無期限に停止しないよう、上限時間付きで本文を取得する。"""

    with urllib.request.urlopen(url, timeout=URL_TIMEOUT_SECONDS) as response:
        return response.read().decode()


def _unknown_license_text(name: str, version: str | None) -> str:
    """パッケージメタデータに本文がない既知の依存だけ、一次配布元から補完する。"""

    normalized_name = name.lower()
    if normalized_name == "torch-complex":
        return Path(
            "docs/licenses/torch_complex/for-torch-complex-license.txt"
        ).read_text(encoding="utf-8")
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
            name="Open JTalk",
            version="1.11",
            license="Modified BSD license",
            text=Path("docs/licenses/open_jtalk/COPYING").read_text(encoding="utf-8"),
        ),
        License(
            name="MeCab",
            version=None,
            license="Modified BSD license",
            text=Path("docs/licenses/open_jtalk/mecab/COPYING").read_text(
                encoding="utf-8"
            ),
        ),
        License(
            name="NAIST Japanese Dictionary",
            version=None,
            license="Modified BSD license",
            text=Path("docs/licenses/open_jtalk/mecab-naist-jdic/COPYING").read_text(
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


def _python_package_licenses() -> list[License]:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "piplicenses",
                "--from=mixed",
                "--format=json",
                "--with-urls",
                "--with-license-file",
                "--no-license-path",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise LicenseGenerationError(f"pip-licenses failed:\n{error.stderr}") from error

    licenses: list[License] = []
    seen_package_keys: set[tuple[str, str]] = set()
    for package in json.loads(completed.stdout):
        package_key = (package["Name"].lower(), package["Version"])
        if package_key in seen_package_keys:
            continue
        seen_package_keys.add(package_key)

        license_text = package["LicenseText"]
        if not license_text or license_text == "UNKNOWN":
            license_text = _unknown_license_text(package["Name"], package["Version"])
        licenses.append(
            License(
                name=package["Name"],
                version=package["Version"],
                license=package["License"],
                text=license_text,
            )
        )
    return licenses


def generate_licenses() -> list[License]:
    return [*_bundled_licenses(), *_python_package_licenses()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=Path)
    args = parser.parse_args()
    serialized = [asdict(license) for license in generate_licenses()]

    if args.output_path is None:
        json.dump(serialized, sys.stdout, ensure_ascii=False)
        return
    with args.output_path.open("w", encoding="utf-8") as output:
        json.dump(serialized, output, ensure_ascii=False)


if __name__ == "__main__":
    main()
