# ソースコードと再ビルド

CoreとEngine本体のソースコードには、個別表記があるものを除きLGPL-3.0-onlyが適用されます。各依存ライブラリには固有のライセンスが適用され、NVIDIAなどのライブラリをLGPLやGPLに変更するものではありません。

## 配布物に含まれるソースとライセンス

| 対象 | パスと参照先 |
| --- | --- |
| 本体ソースアーカイブ | `sources/coeiroink_core.zip`、`sources/coeiroink_engine.zip` |
| GPL、LGPL、MPL依存ソース | `sources/`配下のアーカイブ、[sources/README.md](./sources/README.md) |
| 依存ライブラリのライセンス本文 | `engine_manifest_assets/dependency_licenses.json` |
| その他の依存とOpenCL拡張設定 | `uv.lock`の`sdist.url`と`source.git`、Coreの`native/opencl/CMakeLists.txt` |

`sources/coeiroink_core.zip`と`sources/coeiroink_engine.zip`を同じディレクトリへ展開すると、CoreとEngineが隣同士に配置されます。

## リリースパッケージの再ビルド

Python 3.14、uv、Git、対象OSのC/C++ビルド環境が必要です。GPU版の追加要件は本体のREADMEを参照してください。

Engineのディレクトリで、配布物と同じバックエンドを指定して順に実行します。

```bash
uv sync --locked --extra cpu --no-dev
uv pip list --format json > runtime-packages.json
uv sync --locked --extra cpu --group build --group licenses --no-dev
uv run --no-sync python generate_licenses.py --package-snapshot runtime-packages.json --output-path engine_manifest_assets/dependency_licenses.json --sources-dir build/licenses/sources
uv run --locked --extra cpu --group build --no-dev pyinstaller --noconfirm --clean run.spec -- --backend cpu
```

GPU版をビルドする場合は、コマンド内の`cpu`を`cuda`、`opencl`、`directml`のいずれかへ置き換えてください。依存ライブラリを改変する場合は、それぞれのソースアーカイブ内のビルド手順に従ってください。

## Dockerイメージ

イメージ内では`/opt/coeiroink/coeiroink_core`と`/opt/coeiroink/coeiroink_engine`に本体ソースを、`/opt/coeiroink/coeiroink_engine/licenses/sources`に依存のソースアーカイブを収録しています。再ビルド時は両リポジトリを含む親ディレクトリをビルドコンテキストにして、EngineのDockerfileを使用してください。

## 公開ソースと再配布条件

- **公開リポジトリ**：[COEIROINK Core](https://github.com/cyrne1-7208/coeiroink_core)、[COEIROINK Engine](https://github.com/cyrne1-7208/coeiroink_engine)
- **再配布時の注意**：再配布時は、使用したバージョンのソースとビルド用ファイルをライセンス本文とともに提供してください。本体の公開リポジトリだけでは、同梱した依存ライブラリのソースを提供したことにはなりません。
