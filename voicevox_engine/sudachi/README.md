# Sudachi full

Sudachiのfull辞書を使った実験的な形態素解析機能です。COEIROINKのHTTP API、Open JTalkによる変換、音声合成には使用しません。

## 有効化と基本実行

通常のEngineセットアップでは導入されないため、利用時のみEngineのディレクトリからPython 3.14環境へインストールします。実行時は`--experimental sudachi`の指定が必須です。

```bash
uv sync --locked --extra sudachi
uv run --locked --extra sudachi python -m voicevox_engine.sudachi --experimental sudachi --json "東京都へ行く"

# Engine既定辞書と保存済みユーザー辞書を併用する例
uv run --locked --extra sudachi python -m voicevox_engine.sudachi \
  --experimental sudachi --open-jtalk-dictionaries --json "COEIROINKを使う"
```

- **環境要件**：SudachiPyはRust実装を含むビルド済みwheelを使うため、Rustコンパイラは不要です。対応wheelがない環境では利用できません。
- **辞書データ**：`SudachiDict`のfull辞書パッケージを読み込みます。辞書本体はリポジトリに含まれません。

## 主なCLIオプション

| オプション | 説明 |
| --- | --- |
| `--experimental sudachi` | 実験的Sudachi機能の有効化。実行時は必須 |
| `--mode` | 分割単位を`A`、`B`、`C`から選択。既定値は`C` |
| `--json` | 解析結果をJSON形式で出力 |
| `--open-jtalk-dictionaries` | Engineの`default.csv`と保存済み`user_dict.json`を追加で読込 |
| `--open-jtalk-csv PATH` | 追加のOpen JTalk辞書CSVを指定。複数回指定可 |
| `--open-jtalk-user-json PATH` | Engine形式のユーザー辞書JSONを指定 |

## Open JTalk辞書変換の仕様

- **入力形式と変換時期**：変換は初期化時に一度だけ行い、解析ごとに再変換しません。`.dic`からは変換できないため、UTF-8形式のCSVを指定してください。
- **単語コスト**：`-32767`から`32767`の範囲外は丸めて警告します。値が低い単語ほど優先され、同語かつ同コストならユーザー辞書を優先します。
- **品詞と文脈ID**：固有名詞、普通名詞、動詞、形容詞、接尾辞をUniDic 2.1.2に基づくSudachi定義へ変換します。未対応の品詞は普通名詞として警告し、アクセント型とアクセント結合規則はOpen JTalk側に残します。

## 注意

- `normalized_form`は正規化後の文字列で入力と異なる場合があるため、連結して読みや音素変換へ渡さないでください。
- 入力文字列上の位置を参照するときは`begin`と`end`を使います。
- `--open-jtalk-dictionaries`指定時、`user_dict.json`はOSのユーザーデータ領域から探索され、存在しない場合は既定辞書`default.csv`のみ使います。
- 利用時は上流実装と辞書のライセンスおよび配布条件に従ってください。
