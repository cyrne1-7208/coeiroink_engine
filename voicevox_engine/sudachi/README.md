# Sudachi full

Sudachiのfull辞書を使った形態素解析を、実験的な機能として提供します。この機能は、COEIROINKのHTTP API、Open JTalkによる変換、音声合成には使用されません。

## 有効化

通常のEngineセットアップではSudachiをインストールしません。利用する場合だけ、Python 3.12環境で次を実行します。

```bash
uv sync --locked --extra sudachi
uv run --locked --extra sudachi python -m voicevox_engine.sudachi --experimental sudachi --json "東京都へ行く"
```

実行時にも`--experimental sudachi`が必要です。SudachiPyはRust実装を含むビルド済みwheelを利用するため、通常のインストールにRustコンパイラは不要です。対応するwheelがない環境では、この機能を利用できません。

辞書は`SudachiDict`パッケージのfull辞書を読み込みます。辞書データはリポジトリに含まれません。

## Open JTalk辞書

Engineの既定辞書`default.csv`と、OSのユーザーデータ領域に保存された`user_dict.json`を追加して解析できます。`user_dict.json`が存在しない場合は、既定辞書だけを使用します。

```bash
uv run --locked --extra sudachi python -m voicevox_engine.sudachi \
  --experimental sudachi --open-jtalk-dictionaries --json "COEIROINKを使う"
```

ほかのOpen JTalk辞書を使う場合は、辞書のCSVを`--open-jtalk-csv PATH`で追加します。Engine形式のユーザー辞書は、`--open-jtalk-user-json PATH`で指定します。CSVは複数指定できます。

既定辞書と、`--open-jtalk-csv`で追加した環境辞書、ユーザー辞書は、起動時に一度だけSudachi形式へ変換します。形態素解析のたびに変換し直すことはありません。

Open JTalkの単語コストは、Sudachiで扱える範囲では変更しません。範囲外の値は`-32767`から`32767`の範囲に収め、警告を出します。Open JTalkと同じく、値が低い単語を優先します。同じ単語が同じコストで登録されている場合は、後から登録したユーザー辞書を優先します。

Open JTalkとSudachiでは、接続行列と品詞体系が異なります。そのため、VOICEVOX互換辞書で使う固有名詞、普通名詞、動詞、形容詞、接尾辞を、UniDic 2.1.2を基にしたSudachiの品詞と文脈IDへ変換します。対応していない品詞は普通名詞として扱い、警告を出します。アクセント型とアクセント結合規則はOpen JTalk側に残します。

コンパイル済みのOpen JTalk辞書（`.dic`）からは、変換に必要な語彙情報を復元できません。元になったUTF-8形式のCSVを指定してください。

## 注意

`normalized_form`はSudachiが正規化した文字列です。活用語では入力と異なる場合があるため、この値を連結してCOEIROINKの読みや音素変換へ渡さないでください。`begin`と`end`は、入力文字列上の位置を表します。

分割単位は`--mode A`、`--mode B`、`--mode C`で選択できます。

上流実装と辞書のライセンスおよび配布条件に従って利用してください。
