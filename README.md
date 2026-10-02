# COEIROINK Engine (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Engineです。HTTP API、MYCOEIROINKモデルのメタデータ管理、リクエスト処理を担当し、音声合成は隣接する`coeiroink_core`へ委譲します。GUIと歌唱機能は対象外です。

## 対象環境

現在の公式検証対象はLinux x64 CPU、Python 3.12です。EngineとCoreを同じ親ディレクトリへ配置してください。

## セットアップ

Python 3.12とC/C++コンパイラを用意し、Engineディレクトリから実行します。

```bash
bash build_util/setup_mycoeiroink_linux_cpu.bash .venv ../coeiroink_core
```

Dockerで起動する場合は、CoreとEngineを含む親ディレクトリをビルドコンテキストにしてください。イメージのビルドにはCPU版PyTorchとESPnetの取得が必要です。

```bash
docker build -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
docker run --rm -p 50032:50032 -v /path/to/speaker_info:/speaker_info:ro coeiroink-engine:cpu
```

コンテナ内のモデルは`/speaker_info`へ読み取り専用でマウントします。

MYCOEIROINKのZIPを展開し、モデルフォルダを`/path/to/speaker_info`直下へ配置します。ZIPファイル自体ではなく、展開後のディレクトリを指定してください。

```bash
.venv/bin/python run.py --host 127.0.0.1 --speaker_info_dir /path/to/speaker_info --cpu_num_threads 1
```

既定の待受ポートは`50032`です。モデルは必要になった時点でロードされ、ロード済みモデルは保持されます。明示的なモデル数上限は設けません。

## API

起動確認と話者一覧は次で取得できます。

```bash
curl http://127.0.0.1:50032/
curl http://127.0.0.1:50032/v1/engine_info
curl http://127.0.0.1:50032/v1/speakers
```

COEIROINK v2形式のAPIは`/v1/predict`、`/v1/predict_with_duration`、`/v1/process`、`/v1/synthesis`などを提供します。OpenAPIドキュメントは`http://127.0.0.1:50032/docs`、定義JSONは`/openapi.json`で確認できます。

VOICEVOX互換の通常音声APIは、互換用名前空間として`/voicevox`配下にあります。クエリ作成と合成の例は次のとおりです。

```bash
curl -s -X POST 'http://127.0.0.1:50032/voicevox/audio_query?speaker=1' --get --data-urlencode 'text=こんにちは' > query.json
curl -s -H 'Content-Type: application/json' -X POST --data-binary @query.json 'http://127.0.0.1:50032/voicevox/synthesis?speaker=1' > audio.wav
```

通常音声APIの主なルートは`/voicevox/audio_query`、`/voicevox/accent_phrases`、`/voicevox/synthesis`、`/voicevox/cancellable_synthesis`、`/voicevox/multi_synthesis`です。旧形式のルートは提供しません。

VOICEVOXに存在する歌唱、音声ライブラリ管理、モーラ単位の調整、音声モーフィングなど、COEIROINKが提供しない機能はOpenAPIへ通常掲載せず、呼び出された場合は理由付きの`501 Not Implemented`を返します。利用可能な全ルートは起動後の`/docs`で確認してください。

## モデルと辞書

モデルのメタデータには話者UUID、スタイルID、アイコン、ライセンス、サンプルを含めてください。Engineはモデルフォルダ名ではなくメタデータの識別子を使います。

ユーザー辞書は`/voicevox/user_dict`と`/voicevox/user_dict_word`から管理できます。Open JTalkの既定辞書は`default.csv`です。ユーザー辞書はOSのユーザーデータ領域へ保存されます。

## テストと開発

Coreを隣接ディレクトリへ配置し、次を実行します。

```bash
PYTHONPATH=.:../coeiroink_core/src .venv/bin/python -m pytest -q
```

モデルファイルや外部配布物の内部コードはリポジトリへ含めません。

静的検査には開発依存関係を導入したうえで、次を使用できます。

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/ruff check run.py voicevox_engine test --select E4,E7,E9
```

## ライセンス

LGPL v3です。詳細は[LICENSE](./LICENSE)を参照してください。依存ライセンスの情報は`engine_manifest_assets/dependency_licenses.json`に収録されています。

## 謝辞

本プロジェクトは、[shirowanisan/voicevox_engine](https://github.com/shirowanisan/voicevox_engine)と[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[FastAPI](https://github.com/fastapi/fastapi)、[ESPnet](https://github.com/espnet/espnet)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
