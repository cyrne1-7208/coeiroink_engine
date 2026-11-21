# CIの起動確認で使う、ランダムな重みの小さなMYCOEIROINK話者を作るスクリプト
#
# 実モデルを使わずに、モデルの読み込みから合成までの経路を確かめる。合成される音声は雑音になる。

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from espnet2.tasks.tts import TTSTask

SPEAKER_UUID = "00000000-0000-4000-8000-00000000c1c1"
STYLE_ID = 424242
# ESPnetとCoreがモデルの構築と読み込みで参照する項目
CONFIG = {
    # 音素はすべて<unk>として扱われるが、読み込みと合成の経路を確かめるには足りる。
    "token_list": ["<blank>", "<unk>", "<sos/eos>"],
    "token_type": "phn",
    "bpemodel": None,
    "non_linguistic_symbols": None,
    "cleaner": "jaconv",
    "g2p": "pyopenjtalk_prosody",
    "use_preprocessor": True,
    "odim": None,
    "feats_extract": "linear_spectrogram",
    "feats_extract_conf": {"n_fft": 2048, "hop_length": 512, "win_length": None},
    "normalize": None,
    "pitch_extract": None,
    "pitch_normalize": None,
    "energy_extract": None,
    "energy_normalize": None,
    "model_conf": {},
    "tts": "vits",
    "tts_conf": {
        "sampling_rate": 44100,
        # 既定のVITSから層とチャンネルを減らし、アップサンプリング倍率の積をhop_lengthの512に合わせる。
        "generator_params": {
            "hidden_channels": 16,
            "text_encoder_blocks": 1,
            "text_encoder_ffn_expand": 1,
            "decoder_channels": 32,
            "decoder_upsample_scales": [8, 8, 2, 2, 2],
            "decoder_upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "posterior_encoder_layers": 2,
            "flow_flows": 1,
            "flow_layers": 1,
            "stochastic_duration_predictor_flows": 1,
            "stochastic_duration_predictor_dds_conv_layers": 1,
        },
    },
}


# speaker_info_dirの下に、テスト用の話者をMYCOEIROINKの形式で書き出す。
def write_speaker(speaker_info_dir: Path) -> None:
    speaker_dir = speaker_info_dir / SPEAKER_UUID
    model_dir = speaker_dir / "model" / str(STYLE_ID)
    model_dir.mkdir(parents=True)
    # 実行ごとに同じ重みになるよう乱数を固定する。
    torch.manual_seed(0)
    model = TTSTask.build_model(argparse.Namespace(**CONFIG))
    # 推論に使う生成器の重みだけを保存し、学習用の識別器は含めない。
    generator = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith("tts.generator.")
    }
    torch.save(generator, model_dir / "model.pth")
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8"
    )
    metas = {
        "speakerName": "CI",
        "speakerUuid": SPEAKER_UUID,
        "styles": [{"styleName": "CI", "styleId": STYLE_ID}],
    }
    (speaker_dir / "metas.json").write_text(
        json.dumps(metas, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker_info_dir", type=Path)
    args = parser.parse_args()
    write_speaker(args.speaker_info_dir)
    # シェルスクリプトが合成に使うスタイルIDを受け取れるように出力する。
    print(STYLE_ID)


if __name__ == "__main__":
    main()
