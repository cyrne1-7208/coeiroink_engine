# COEIROINK Engine (Forked by Cyrne1)

Cyrne1によるCOEIROINK Engineのフォークです。HTTP APIを担当し、音声合成は隣接する[coeiroink_core](https://github.com/cyrne1-7208/coeiroink_core)が行います。

## 動作環境

64bit環境のPython 3.14で動作します。[uv](https://docs.astral.sh/uv/)、[Git](https://git-scm.com/)、C/C++ビルド環境が必要です。Python 3.14が未導入の場合は`uv`が自動で取得します。

| 環境 | Linux x64 | Windows x64 |
| --- | --- | --- |
| CPU | `cpu` | `cpu` |
| NVIDIA GPU | `cuda` | `cuda` |
| GPU全般 | `opencl` | `directml` |

- **`cuda`**：CUDA 13.0対応のNVIDIAドライバが必要です。
- **`opencl`**：OpenCL対応GPUとドライバに加え、OpenCL C++ヘッダー、ICDローダー、SQLite 3開発用ヘッダーが必要です。
- **`directml`**：Windows 10 バージョン1709以降、DirectX 12対応GPU、および[Microsoft Visual C++ x64再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)が必要です。[torch-directml-backwards](https://github.com/cyrne1-7208/torch-directml-backwards)を使用するため、旧`torch-directml`とは併用しないでください。

## セットアップ

CoreとEngineを同じ親ディレクトリに並べ、Engine配下の`speaker_info`にMYCOEIROINKのZIPを**展開したフォルダ**を名前を変えずに入れます。

```text
親ディレクトリ/
├── coeiroink_core/
└── coeiroink_engine/
    ├── run.py
    └── speaker_info/
        └── 展開したモデルのフォルダ/
```

上の表からバックエンドを1つ選び、Engineのディレクトリで導入します。`libsoxr`を使う場合は`--extra soxr`も併せて指定してください。

```bash
uv sync --locked --extra cpu
mkdir speaker_info
```

## 起動とオプション

`--device`にもセットアップ時と同じバックエンドを指定します。既定の待受URLは`http://127.0.0.1:50032`です。

```bash
# Linux
.venv/bin/python run.py --speaker_info_dir speaker_info --device cpu

# Windows
.venv\Scripts\python.exe run.py --speaker_info_dir speaker_info --device cpu
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `50032` | 待受アドレスとポート番号 |
| `--device` | `cpu` | 推論デバイス。`cpu`、`cuda`、`opencl`、`directml`から指定 |
| `--speaker_info_dir` | `./speaker_info` | 展開済みモデルの配置先 |
| `--max-loaded-models` | `1` | LRU保持数。値省略または`all`で起動時に全モデルを読込 |
| `--dangerous-skip-check` | 無効 | 容量不足を警告だけにとどめ、拒否と自動解放を省略 |
| `--experimental` | なし | 実験的機能を有効化。複数指定する場合はオプションを繰り返す |

### メモリ管理

- 起動時は最大モデル、切替時は対象モデルのサイズで空きを確認し、不足時は古いモデルから解放します。
- 最大1個分の空き不足や読込失敗時は起動を中止し、`all`で同時保持できない場合のみ警告して続行します。
- `--dangerous-skip-check`指定時も保持数の上限は維持され、実メモリ不足で異常終了する危険があります。

### 実験的な機能

`--experimental`に以下を指定します。既定ではすべて無効で、併用する場合はオプションを繰り返します。

| 指定値 | 内容 | 条件と対象外 |
| --- | --- | --- |
| `chunked-decoding` | 長い音声の波形生成を分割して実行 | 約9秒を超える音声だけが対象 |
| `skip-trailing-silence` | `/v1/synthesis`とVOICEVOX互換合成で、文末の無音の波形生成を省略 | `/v1/predict`系は対象外 |
| `soxr` | libsoxr VHQでリサンプリング | 導入時に`--extra soxr`が必要 |
| `voice-smoothing` | `/v1/synthesis`とVOICEVOX互換合成の音声を補正 | `/v1/predict`系は対象外 |
| `warmup` | ロード済みモデルを短い合成で事前準備 | `--max-loaded-models all`または値省略と併用し、未読込モデルと`--enable_cancellable_synthesis`は対象外 |

## Docker

Linux版はDockerでも起動できます。CoreとEngineの親ディレクトリでビルドして起動します。`COEIROINK_BACKEND`には既定値の`cpu`のほか、`cuda`や`opencl`を指定できます。

```bash
docker build --build-arg COEIROINK_BACKEND=cpu -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
docker run --rm -p 127.0.0.1:50032:50032 -v "$(pwd)/coeiroink_engine/speaker_info:/opt/coeiroink/speaker_info:ro" coeiroink-engine:cpu
```

CUDA版では`docker run`に`--gpus all`を追加し、OpenCL版ではホストのGPUデバイスとICDをコンテナへ渡してください。

## API

COEIROINK APIは`/v1`、VOICEVOX互換APIは`/voicevox`配下にあります。ユーザー辞書は`/voicevox/user_dict`と`/voicevox/user_dict_word`で操作でき、起動後に`http://127.0.0.1:50032/docs`で仕様を確認できます。未対応の機能は`501 Not Implemented`を返します。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
uv run --locked --extra cpu --group dev ruff check .
uv run --locked --extra cpu --group dev ruff format --check .
```

`sudachi`マーカーが付いたテストは、`--extra sudachi`未導入環境では自動でスキップされます。手動で除外する場合は`-m "not sudachi"`を指定してください。

SudachiはHTTP APIや音声合成では使用しない実験的な形態素解析CLI機能です。使い方は[Sudachi README](https://github.com/cyrne1-7208/coeiroink_engine/blob/main/voicevox_engine/sudachi/README.md)を参照してください。

## ライセンス

個別表記があるものを除き、ソースコードはLGPL-3.0-onlyです。

- **本体ライセンス**：[LICENSE](./LICENSE)、[GPL-3.0.txt](./licenses/GPL-3.0.txt)
- **依存ライブラリ**：[dependency_licenses.json](./engine_manifest_assets/dependency_licenses.json)、[licenses/](./licenses/)
- **ソースと再ビルド**：[licenses/SOURCES.md](./licenses/SOURCES.md)
- **利用規約**：[terms_of_service.md](./engine_manifest_assets/terms_of_service.md)

## 謝辞

本プロジェクトは[COEIROINK](https://coeiroink.com/)、[shirowanisan/voicevox_engine](https://github.com/shirowanisan/voicevox_engine)、[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[FastAPI](https://github.com/fastapi/fastapi)、[ESPnet](https://github.com/espnet/espnet)などのOSSを利用しています。開発者と貢献者の皆様に感謝します。
