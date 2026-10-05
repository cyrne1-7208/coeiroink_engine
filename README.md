# COEIROINK Engine (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Engineです。音声合成は、[coeiroink_core](https://github.com/cyrne1-7208/coeiroink_core)が行います。

## 動作環境

Python 3.12を使用します。

| バックエンド | OS（x64） | GPUの要件 |
| --- | --- | --- |
| CPU | Linux・Windows | 不要 |
| CUDA | Linux・Windows | CUDA 12.8対応のNVIDIAドライバ |
| OpenCL | Linux | OpenCL対応GPUとドライバ |
| DirectML | Windows | Windows 10 バージョン1709以降・DirectX 12対応GPUとドライバ |

## セットアップ

ソースから実行するには、[uv](https://docs.astral.sh/uv/)、[Git](https://git-scm.com/)、C/C++のビルド環境が必要です。Python 3.12は、インストールされていなければuvが取得します。

OpenCL版には追加で、OpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダーも必要です。

CoreとEngineを同じ親ディレクトリに配置し、Engineのディレクトリで実行してください。

```bash
uv sync --locked --extra cpu
```

GPU版では、`cpu`を`cuda`、`opencl`、`directml`のいずれかに置き換えます。バックエンドは1つだけ選択してください。

## 起動

Engineのディレクトリで、モデルを入れる`speaker_info`フォルダを作成してください。

```bash
mkdir speaker_info
```

MYCOEIROINKのZIPを展開し、モデルのフォルダを名前を変えずに入れてください。

```text
coeiroink_engine/
├── run.py
└── speaker_info/
    └── 展開したモデルのフォルダ/
```

Engineのディレクトリで起動します。

```bash
.venv/bin/python run.py --speaker_info_dir speaker_info --device cpu
```

Windowsでは`.venv\Scripts\python.exe`を使います。GPU版では、`--device`にもセットアップ時と同じバックエンドを指定してください。

既定の接続先は`http://127.0.0.1:50032`です。`--host`と`--port`で変更できます。

### モデルの保持数

既定では、最後に使った1モデルを保持します。

- `--max-loaded-models 3`：最近使ったモデルを最大3つ保持します。
- `--max-loaded-models`または`--max-loaded-models all`：起動時に全モデルを読み込みます。

空きメモリが足りなくなると、使用していない期間が長いモデルから解放します。

### 実験的な機能

- `--experimental soxr`：リサンプラーにlibsoxr VHQを使います。セットアップ時に`--extra soxr`も指定してください。
- `--experimental voice-smoothing`：母音の音色や周期の細かな揺れを抑えます。`/v1/synthesis`とVOICEVOX互換の合成に適用されます。

どちらも既定では無効です。併用する場合は、`--experimental soxr --experimental voice-smoothing`と指定します。

## Docker

Linux版はDockerでも起動できます。Docker Engineを用意し、CoreとEngineの親ディレクトリで実行してください。

モデルは、上で作成した`coeiroink_engine/speaker_info`から読み込みます。

```bash
docker build --build-arg COEIROINK_BACKEND=cpu -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
docker run --rm -p 127.0.0.1:50032:50032 -v "$(pwd)/coeiroink_engine/speaker_info:/opt/coeiroink/speaker_info:ro" coeiroink-engine:cpu
```

GPU版では、`COEIROINK_BACKEND`に`cuda`または`opencl`を指定します。CUDAでは起動時に`--gpus all`を追加してください。OpenCLでは、ホストのGPUデバイスとICDをコンテナに渡す必要があります。

## API

COEIROINK APIは`/v1`、VOICEVOX互換APIは`/voicevox`配下にあります。起動後の[APIドキュメント](http://127.0.0.1:50032/docs)で、各APIの使い方を確認できます。未対応の機能は`501 Not Implemented`を返します。

ユーザー辞書は`/voicevox/user_dict`と`/voicevox/user_dict_word`から管理できます。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
uv run --locked --extra cpu --group dev ruff check .
uv run --locked --extra cpu --group dev ruff format --check .
```

## ライセンス

個別にライセンスが示されているものを除き、ソースコードはLGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。

依存ライブラリのライセンス原文は[licenses](./licenses/)、ライブラリの一覧とライセンスは[dependency_licenses.json](./engine_manifest_assets/dependency_licenses.json)にあります。

同梱ソースと再ビルド方法は [licenses/SOURCES.md](./licenses/SOURCES.md) を参照してください。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)、[shirowanisan/voicevox_engine](https://github.com/shirowanisan/voicevox_engine)、[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[FastAPI](https://github.com/fastapi/fastapi)、[ESPnet](https://github.com/espnet/espnet)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
