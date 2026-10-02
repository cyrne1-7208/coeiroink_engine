"""旧Core用の音素IDとラベル入出力を、継承による同一テストの再実行なしに検証する。"""

import numpy as np
import pytest

from voicevox_engine.acoustic_feature_extractor import (
    BasePhoneme,
    JvsPhoneme,
    OjtPhoneme,
)


def test_label_parsing_rounds_timestamps():
    phoneme = BasePhoneme.parse("32.67543\t33.48933 e")
    assert phoneme == BasePhoneme("e", 32.68, 33.49)
    assert phoneme.duration == pytest.approx(0.81)


@pytest.mark.parametrize(
    "phoneme_class, ids, width",
    [
        (JvsPhoneme, [0, 19, 4, 3, 0], 39),
        (OjtPhoneme, [0, 23, 7, 6, 0], 45),
    ],
)
def test_silence_conversion_ids_and_onehot(phoneme_class, ids, width):
    phonemes = phoneme_class.convert(
        [
            phoneme_class(name, index, index + 1)
            for index, name in enumerate(["sil", "k", "a", "U", "sil"])
        ]
    )
    assert [p.phoneme for p in phonemes] == ["pau", "k", "a", "U", "pau"]
    assert [p.phoneme_id for p in phonemes] == ids
    encoded = np.stack([p.onehot for p in phonemes])
    assert encoded.shape == (5, width)
    np.testing.assert_array_equal(encoded.argmax(axis=1), ids)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(5))


@pytest.mark.parametrize("phoneme_class", [JvsPhoneme, OjtPhoneme])
def test_label_file_round_trip_and_validation(tmp_path, phoneme_class):
    path = tmp_path / "phonemes.lab"
    phonemes = [phoneme_class("pau", 0, 0.1), phoneme_class("a", 0.1, 0.3)]
    phoneme_class.save_lab_list(phonemes, path)
    assert path.read_text(encoding="utf-8") == "0.00\t0.10\tpau\n0.10\t0.30\ta"
    assert phoneme_class.load_lab_list(path) == phonemes
    path.write_text("0 1 unknown", encoding="utf-8")
    with pytest.raises(AssertionError, match="not defined"):
        phoneme_class.load_lab_list(path)
