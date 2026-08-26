# COEIROINK Engine (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Engineです。HTTP API、MYCOEIROINKモデルのメタデータ管理、リクエスト処理を担当し、音声合成は隣接する[coeiroink_core](https://github.com/cyrne1-7208/coeiroink_core)へ委譲します。GUIと歌唱機能は対象外です。

## 対象環境

EngineとCoreを同じ親ディレクトリへ配置し、利用するバックエンドを1つ選択します。

| OS | バックエンド | uv extra | 起動時の指定 | Python |
| --- | --- | --- | --- | --- |
| Linux x64 | CPU | `cpu` | `--device cpu` | 3.12 |
| Linux x64 | CUDA | `cuda` | `--device cuda` | 3.12 |
| Linux x64 | OpenCL | `opencl` | `--device opencl` | 3.12 |
| Windows x64 | CPU | `cpu` | `--device cpu` | 3.12 |
| Windows x64 | CUDA | `cuda` | `--device cuda` | 3.12 |
| Windows x64 | DirectML | `directml` | `--device directml` | 3.12 |

## セットアップ

LinuxまたはWindowsのCPU環境では、Engineディレクトリから次を実行します。

```bash
uv sync --locked --extra cpu
```

CUDAまたはOpenCLでは`cpu`を`cuda`または`opencl`へ置き換えてください。Windows DirectMLでは次を実行します。

```powershell
uv sync --python 3.12 --locked --extra directml
```

Linux CPUではセットアップスクリプトも利用できます。

```bash
bash build_util/setup_mycoeiroink_linux_cpu.bash .venv ../coeiroink_core
```

MYCOEIROINKのZIPを展開し、モデルフォルダを`/path/to/speaker_info`直下へ配置します。ZIPファイル自体ではなく、展開後のディレクトリを指定してください。

```bash
.venv/bin/python run.py \
  --host 127.0.0.1 \
  --speaker_info_dir /path/to/speaker_info \
  --device cpu
```

Windowsでは`.venv\Scripts\python.exe`を使用します。既定の待受ポートは`50032`です。モデルは必要になった時点でロードされ、別モデルへ切り替える際は現在のモデルを解放して置き換えます。

Dockerで起動する場合は、CoreとEngineを含む親ディレクトリをビルドコンテキストにしてください。

```bash
docker build --build-arg COEIROINK_BACKEND=cpu \
  -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
docker run --rm -p 127.0.0.1:50032:50032 \
  -v /path/to/speaker_info:/opt/coeiroink/speaker_info:ro \
  coeiroink-engine:cpu
```

Linux CUDAとOpenCLも同じDockerfileから`COEIROINK_BACKEND=cuda`または`opencl`を指定してビルドできます。CUDAは`--gpus all`、OpenCLはホストのICDとデバイスをコンテナへ渡し、必要に応じて`COEIROINK_DEVICE`を指定してください。Windows CPU・CUDA・DirectMLはWindows用のリリースアーカイブで提供し、Linux用Dockerイメージとは分けて扱います。

## API

起動確認と話者一覧は次で取得できます。

```bash
curl http://127.0.0.1:50032/
curl http://127.0.0.1:50032/v1/engine_info
curl http://127.0.0.1:50032/v1/speakers
```

COEIROINK v2形式のAPIは`/v1/predict`、`/v1/predict_with_duration`、`/v1/process`、`/v1/synthesis`などを提供します。OpenAPIドキュメントは`http://127.0.0.1:50032/docs`、定義JSONは`/openapi.json`で確認できます。

VOICEVOX互換の通常音声APIは`/voicevox`配下にあります。クエリ作成と合成の例は次のとおりです。

```bash
curl -s -X POST 'http://127.0.0.1:50032/voicevox/audio_query?speaker=1' --get --data-urlencode 'text=こんにちは' > query.json
curl -s -H 'Content-Type: application/json' -X POST --data-binary @query.json 'http://127.0.0.1:50032/voicevox/synthesis?speaker=1' > audio.wav
```

通常音声APIの主なルートは`/voicevox/audio_query`、`/voicevox/accent_phrases`、`/voicevox/synthesis`、`/voicevox/cancellable_synthesis`、`/voicevox/multi_synthesis`です。COEIROINKが提供しない機能は理由付きの`501 Not Implemented`を返します。利用可能な全ルートは起動後の`/docs`で確認してください。

## モデルと辞書

モデルのメタデータには話者UUID、スタイルID、アイコン、ライセンス、サンプルを含めてください。Engineはモデルフォルダ名ではなくメタデータの識別子を使います。

ユーザー辞書は`/voicevox/user_dict`と`/voicevox/user_dict_word`から管理できます。Open JTalkの既定辞書は`default.csv`です。ユーザー辞書はOSのユーザーデータ領域へ保存されます。

## テストと開発

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
uv run --locked --extra cpu --group dev ruff check .
uv run --locked --extra cpu --group dev ruff format --check .
```

## ライセンス

LGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。依存ライセンスの情報は`engine_manifest_assets/dependency_licenses.json`に収録されています。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)、[shirowanisan/voicevox_engine](https://github.com/shirowanisan/voicevox_engine)、[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[FastAPI](https://github.com/fastapi/fastapi)、[ESPnet](https://github.com/espnet/espnet)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
