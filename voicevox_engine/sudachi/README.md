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

## 注意

`normalized_form`はSudachiの正規化結果です。活用語では入力表記と異なるため、値を連結してCOEIROINKの読みや音素変換へ渡す用途には使いません。`begin`と`end`は入力文字列上の境界です。

分割粒度は`--mode A`、`--mode B`、`--mode C`で選択できます。Sudachiの結果を音声合成へ反映する場合は、既存のCOEIROINK解析との品質比較と別途の設計が必要です。

上流実装と辞書のライセンスおよび配布条件に従って利用してください。
