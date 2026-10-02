# COEIROINK Engine (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Engineです。HTTP API、MYCOEIROINKモデルのメタデータ管理、リクエスト処理を担当します。音声合成は、隣接する[coeiroink_core](https://github.com/cyrne1-7208/coeiroink_core)が行います。GUIと歌唱機能は対象外です。

## 対象環境

EngineとCoreを同じ親ディレクトリに配置し、利用するバックエンドを1つ選択します。

| OS | バックエンド | uv extra | 起動時の指定 | Python |
| --- | --- | --- | --- | --- |
| Linux x64 | CPU | `cpu` | `--device cpu` | 3.12 |
| Linux x64 | CUDA | `cuda` | `--device cuda` | 3.12 |
| Linux x64 | OpenCL | `opencl` | `--device opencl` | 3.12 |
| Windows x64 | CPU | `cpu` | `--device cpu` | 3.12 |
| Windows x64 | CUDA | `cuda` | `--device cuda` | 3.12 |
| Windows x64 | DirectML | `directml` | `--device directml` | 3.12 |

## 事前に必要なもの

このリポジトリのソースから`uv sync`でセットアップする場合は、次のものが必要です。

- [uv](https://docs.astral.sh/uv/)
- [Git](https://git-scm.com/)
- C/C++のビルド環境（LinuxではGCC/G++、WindowsではMSVC Build Tools）
- インターネット接続（初回セットアップではPyPI、GitHub、PyTorchのパッケージ配布先に接続します）
- CUDA版：CUDA 12.8に対応するNVIDIAドライバ
- OpenCL版：GPUベンダーのOpenCLドライバ、OpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダー
- DirectML版：Windows 10 バージョン1709以降、DirectX 12対応GPU、最新のGPUドライバ
- Dockerで起動する場合：Docker Engine

Pythonは3.12を使用します。インストールされていない場合は、`uv`がセットアップ時に取得します。

## セットアップ

`--extra`には、使用するバックエンドを指定します。複数のバックエンドを同時に指定することはできません。

LinuxまたはWindowsのCPU環境では、Engineディレクトリから次を実行します。

```bash
uv sync --locked --extra cpu
```

CUDAまたはOpenCLを利用する場合は、`cpu`を`cuda`または`opencl`に置き換えてください。Windows DirectMLでは次を実行します。

```powershell
uv sync --python 3.12 --locked --extra directml
```

Linux CPUではセットアップスクリプトも利用できます。

```bash
bash build_util/setup_mycoeiroink_linux_cpu.bash .venv ../coeiroink_core
```

MYCOEIROINKのZIPを展開し、モデルのディレクトリを`/path/to/speaker_info`直下へ配置します。ZIPファイル自体は指定できません。

```bash
.venv/bin/python run.py \
  --host 127.0.0.1 \
  --speaker_info_dir /path/to/speaker_info \
  --device cpu
```

Windowsでは`.venv\Scripts\python.exe`を使用します。既定の待受ポートは`50032`です。

モデルは必要になったときに読み込まれます。既定では、最後に使った1モデルを保持します。

- `--max-loaded-models 3`のように数値を指定すると、最近使ったモデルから順にその数まで保持します。
- `--max-loaded-models`または`--max-loaded-models all`を指定すると、起動時に全モデルを読み込みます。

空きメモリが足りない場合は、設定にかかわらず、最後に使ってから最も時間が経ったモデルを解放します。そのため、すべてのモデルがメモリに収まらない環境では、`all`を指定してもリクエスト時にモデルの再読み込みが発生します。

モデルの読み込みには、既定でgenerator-onlyローダーを使用します。VITSの推論に必要な重みだけを読み込むため、合成結果を変えずにメモリ使用量を抑えられます。

実験的な音声補正は、`--experimental voice-smoothing`を指定すると有効になります。既定では無効です。必要なライブラリは各バックエンド用のuv extraに含まれます。

```bash
.venv/bin/python run.py --speaker_info_dir /path/to/speaker_info --device cpu --experimental voice-smoothing
```

この機能は、Coreで母音の音色や周期の細かな揺れを抑えます。`/v1/synthesis`とVOICEVOX互換の合成で有効になります。`/v1/predict`、`/v1/predict_with_duration`、`/v1/process`が返す波形には適用されません。CPUとDirectMLでは補正をCPU上のParselmouthとSciPyで行い、CUDAとOpenCLでは選択したGPU上で行います。`--experimental soxr --experimental voice-smoothing`のように、ほかの実験的機能と併用できます。CPUとDirectMLで利用する[Parselmouth](https://github.com/YannickJadoul/Parselmouth)にはGPL-3.0-or-laterが適用されます。

Dockerで起動する場合は、CoreとEngineを含む親ディレクトリをビルドコンテキストにしてください。

```bash
docker build --build-arg COEIROINK_BACKEND=cpu \
  -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
docker run --rm -p 127.0.0.1:50032:50032 \
  -v /path/to/speaker_info:/opt/coeiroink/speaker_info:ro \
  coeiroink-engine:cpu
```

Linux CUDA版とOpenCL版も、同じDockerfileからビルドできます。`COEIROINK_BACKEND`には`cuda`または`opencl`を指定してください。CUDAでは`--gpus all`を指定します。OpenCLではホストのICDとデバイスをコンテナへ渡し、必要に応じて`COEIROINK_DEVICE`を指定します。WindowsのCPU・CUDA・DirectML版は、Dockerイメージではなくリリースアーカイブで提供します。

## API

起動確認と話者一覧は次で取得できます。

```bash
curl http://127.0.0.1:50032/
curl http://127.0.0.1:50032/v1/engine_info
curl http://127.0.0.1:50032/v1/speakers
```

COEIROINK v2形式のAPIは`/v1/predict`、`/v1/predict_with_duration`、`/v1/process`、`/v1/synthesis`などを提供します。OpenAPIドキュメントは`http://127.0.0.1:50032/docs`、定義JSONは`http://127.0.0.1:50032/openapi.json`で確認できます。

VOICEVOX互換の通常音声APIは`/voicevox`配下にあります。クエリ作成と合成の例は次のとおりです。

```bash
curl -s -X POST 'http://127.0.0.1:50032/voicevox/audio_query?speaker=1' --get --data-urlencode 'text=こんにちは' > query.json
curl -s -H 'Content-Type: application/json' -X POST --data-binary @query.json 'http://127.0.0.1:50032/voicevox/synthesis?speaker=1' > audio.wav
```

通常音声APIの主なルートは`/voicevox/audio_query`、`/voicevox/accent_phrases`、`/voicevox/synthesis`、`/voicevox/cancellable_synthesis`、`/voicevox/multi_synthesis`です。COEIROINKが提供しない機能は理由付きの`501 Not Implemented`を返します。利用可能な全ルートは起動後の`/docs`で確認してください。

## モデルと辞書

モデルのメタデータには話者UUID、スタイルID、アイコン、ライセンス、サンプルを含めてください。Engineはモデルのディレクトリ名ではなく、メタデータの識別子を使います。

ユーザー辞書は`/voicevox/user_dict`と`/voicevox/user_dict_word`から管理できます。Open JTalkの既定辞書は`default.csv`です。ユーザー辞書はOSのユーザーデータ領域へ保存されます。

## テストと開発

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
uv run --locked --extra cpu --group dev ruff check .
uv run --locked --extra cpu --group dev ruff format --check .
```

## ライセンス

本リポジトリのソースコードは、個別にライセンスが示されているものを除き、LGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。GPLv3の参照本文と、同梱するライブラリのライセンス原文は`licenses/`に収録しています。配布パッケージに含まれるライブラリの一覧は、バックエンドごとに`engine_manifest_assets/dependency_licenses.json`へ出力されます。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)、[shirowanisan/voicevox_engine](https://github.com/shirowanisan/voicevox_engine)、[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[FastAPI](https://github.com/fastapi/fastapi)、[ESPnet](https://github.com/espnet/espnet)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
