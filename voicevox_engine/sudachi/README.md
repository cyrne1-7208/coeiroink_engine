# Sudachi full

このフォルダは、Sudachiのfull辞書を個別に検証するための任意機能です。既存のCOEIROINKのHTTP API、Open JTalk変換、音声合成処理には接続していません。

## 有効化

通常のEngine構築ではSudachiをインストールしません。必要な時だけ、Python 3.12環境で次を実行します。

```bash
uv sync --locked --extra sudachi
uv run --locked --extra sudachi python -m voicevox_engine.sudachi --experimental sudachi --json "東京都へ行く"
```

実行時にも`--experimental sudachi`が必要です。SudachiPyはRust実装を含む検証済みwheelを利用するため、通常のビルド時にRustコンパイラを導入しません。対応wheelがない環境でのソースビルドは、この機能の要件に含めません。

辞書は`SudachiDict`のfull辞書を依存パッケージから読み込みます。辞書データは大きいため、リポジトリ内に複製しません。

## Open JTalk辞書

Engineの既定辞書`default.csv`と、存在する場合はOSのユーザーデータ領域に保存された`user_dict.json`を追加して解析できます。

```bash
uv run --locked --extra sudachi python -m voicevox_engine.sudachi \
  --experimental sudachi --open-jtalk-dictionaries --json "COEIROINKを使う"
```

任意のOpen JTalk辞書ソースを指定する場合は、環境辞書CSVを`--open-jtalk-csv PATH`で追加し、Engine形式のユーザー辞書JSONを`--open-jtalk-user-json PATH`で指定します。CSVは複数回指定できます。

環境辞書とユーザー辞書は起動時に一度だけSudachi形式へ変換され、形態素解析ごとの再コンパイルは行いません。Open JTalkの単語コストはSudachiで表現可能な範囲では変更せず、範囲外だけを`-32767`から`32767`へ補正して警告を出します。Open JTalkと同じく低いコストを優先し、同じコストの同一語では後から登録するユーザー辞書を優先します。Open JTalkとSudachiでは接続行列と品詞体系が異なるため、VOICEVOX互換辞書で使用する固有名詞、普通名詞、動詞、形容詞、接尾辞をUniDic 2.1.2由来の代表的なSudachi品詞と文脈IDへ対応付けます。未知の品詞は読み込みを拒否せず、警告した上で普通名詞として利用します。アクセント型とアクセント結合規則はOpen JTalk側に残します。

コンパイル済みのOpen JTalk `.dic`は元の語彙情報へ安全に戻せないため直接読み込みません。変換にはその辞書のUTF-8ソースCSVを指定してください。

## 注意

`normalized_form`はSudachiの正規化結果です。活用語では入力表記と異なるため、値を連結してCOEIROINKの読みや音素変換へ渡す用途には使いません。`begin`と`end`は入力文字列上の境界です。

分割粒度は`--mode A`、`--mode B`、`--mode C`で選択できます。Sudachiの結果を音声合成へ反映する場合は、既存のCOEIROINK解析との品質比較と別途の設計が必要です。

上流実装と辞書のライセンスおよび配布条件に従って利用してください。
