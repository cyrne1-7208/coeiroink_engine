# ソースコードと再ビルド

Core・Engine本体のソースコードには、個別に示されているものを除きLGPL-3.0-onlyが適用されます。依存ライブラリにはそれぞれのライセンスが適用され、NVIDIAなどのライブラリをLGPLやGPLに変更するものではありません。

## リリースパッケージ

`sources/coeiroink_core.zip` と `sources/coeiroink_engine.zip` に、そのパッケージをビルドした際の本体ソースを収録しています。ZIPを同じディレクトリへ展開すると、CoreとEngineが隣同士に配置されます。

GPL・LGPL・MPL依存のソースアーカイブも `sources/` に収録しています。ライブラリ名とバージョンの対応は [sources/README.md](./sources/README.md)、ライセンス本文はパッケージ内の `engine_manifest_assets/dependency_licenses.json` を参照してください。Distance 0.1.3の上流表記の不整合は [distance/ATTRIBUTION.txt](./distance/ATTRIBUTION.txt) に記載しています。

その他の依存も、使用したバージョンとソースの配布元をZIP内の `uv.lock` に記録しています。Pythonパッケージのソースは `sdist.url`、Gitから取得するパッケージのソースは `source.git` を参照してください。OpenCL拡張の上流コミットとビルド設定は、Coreの `native/opencl/CMakeLists.txt` にあります。

再ビルドにはPython 3.12、uv、Git、対象OSのC/C++ビルド環境が必要です。GPU版の追加要件は本体のREADMEを参照してください。Engineのディレクトリで、配布物と同じバックエンドを指定します。

```bash
uv sync --locked --extra cpu --no-dev
uv pip list --format json > runtime-packages.json
uv sync --locked --extra cpu --group build --group licenses --no-dev
uv run --no-sync python generate_licenses.py --package-snapshot runtime-packages.json --output-path engine_manifest_assets/dependency_licenses.json --sources-dir build/licenses/sources
uv run --no-sync pyinstaller --noconfirm --clean run.spec -- --backend cpu
```

GPU版では、上の3か所の `cpu` を `cuda`、`opencl`、`directml` のいずれかへ置き換えます。依存ライブラリを変更してビルドする場合は、それぞれのソースアーカイブにあるビルド手順に従ってください。

## Dockerイメージ

本体ソースはイメージ内の `/opt/coeiroink/coeiroink_core` と `/opt/coeiroink/coeiroink_engine` にあります。依存のソースアーカイブは `/opt/coeiroink/coeiroink_engine/licenses/sources` に収録しています。再ビルドには、両リポジトリを含む親ディレクトリをコンテキストにして、EngineのDockerfileを使用してください。

## 公開ソース

- [COEIROINK Core](https://github.com/cyrne1-7208/coeiroink_core)
- [COEIROINK Engine](https://github.com/cyrne1-7208/coeiroink_engine)

再配布するときは、使用したバージョンのソースとビルド用ファイルを、ライセンス本文とともに提供してください。本体の公開リポジトリだけでは、同梱した依存ライブラリのソースを提供したことにはなりません。
