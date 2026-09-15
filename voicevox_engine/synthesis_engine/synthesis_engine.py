import threading
from itertools import chain, pairwise

import numpy
from scipy.signal import resample

from ..acoustic_feature_extractor import OjtPhoneme
from ..model import AccentPhrase, AudioQuery, Mora
from .core_wrapper import CoreWrapper, OldCoreError
from .synthesis_engine_base import SynthesisEngineBase

unvoiced_mora_phoneme_list = ["A", "I", "U", "E", "O", "cl", "pau"]
mora_phoneme_list = ["a", "i", "u", "e", "o", "N", *unvoiced_mora_phoneme_list]


def to_flatten_moras(accent_phrases: list[AccentPhrase]) -> list[Mora]:
    """アクセント句のモーラと休止モーラを、一つのリストにまとめる。"""
    return list(
        chain.from_iterable(
            [
                *accent_phrase.moras,
                *(
                    [accent_phrase.pause_mora]
                    if accent_phrase.pause_mora is not None
                    else []
                ),
            ]
            for accent_phrase in accent_phrases
        )
    )


def to_phoneme_data_list(phoneme_str_list: list[str]):
    """音素の文字列をOjtPhonemeへ変換する。"""
    phoneme_data_list = [
        OjtPhoneme(phoneme=p, start=i, end=i + 1)
        for i, p in enumerate(phoneme_str_list)
    ]
    return OjtPhoneme.convert(phoneme_data_list)


def split_mora(phoneme_list: list[OjtPhoneme]):
    """音素列をモーラ単位の子音・母音と、元の音素列における母音位置へ分割する。

    子音を持たないモーラに対応する`consonant_phoneme_list`の要素は`None`になる。

    Parameters
    ----------
    phoneme_list : list[OjtPhoneme]
        音素のリスト

    Returns
    -------
    consonant_phoneme_list : list[OjtPhoneme | None]
        モーラごとの子音
    vowel_phoneme_list : list[OjtPhoneme]
        母音の音素列
    vowel_indexes : list[int]
        入力音素列における母音の位置
    """
    vowel_indexes = [
        i for i, p in enumerate(phoneme_list) if p.phoneme in mora_phoneme_list
    ]
    vowel_phoneme_list = [phoneme_list[i] for i in vowel_indexes]
    # 隣接する母音位置の差が1なら母音単独、2なら直前の音素を子音として持つモーラになる。
    consonant_phoneme_list: list[OjtPhoneme | None] = [
        None,
        *(
            None if post - prev == 1 else phoneme_list[post - 1]
            for prev, post in pairwise(vowel_indexes)
        ),
    ]
    return consonant_phoneme_list, vowel_phoneme_list, vowel_indexes


def pre_process(
    accent_phrases: list[AccentPhrase],
) -> tuple[list[Mora], list[OjtPhoneme]]:
    """アクセント句から、モーラと前後の休止を含む音素列を作る。"""
    flatten_moras = to_flatten_moras(accent_phrases)

    phoneme_each_mora = [
        [
            *([mora.consonant] if mora.consonant is not None else []),
            mora.vowel,
        ]
        for mora in flatten_moras
    ]
    phoneme_str_list = list(chain.from_iterable(phoneme_each_mora))
    phoneme_str_list = ["pau", *phoneme_str_list, "pau"]

    phoneme_data_list = to_phoneme_data_list(phoneme_str_list)

    return flatten_moras, phoneme_data_list


class SynthesisEngine(SynthesisEngineBase):
    """旧VOICEVOX Coreと開発用モックの互換性を維持する合成エンジン。"""

    def __init__(
        self,
        core: CoreWrapper,
    ):
        """Coreのメタデータと対応デバイスを読み込み、推論用のロックを作る。"""
        super().__init__()
        self.core = core
        self._speakers = self.core.metas()
        self.mutex = threading.Lock()
        try:
            self._supported_devices = self.core.supported_devices()
        except OldCoreError:
            self._supported_devices = None
        self.default_sampling_rate = 24000

    @property
    def speakers(self) -> str:
        return self._speakers

    @property
    def supported_devices(self) -> str | None:
        return self._supported_devices

    def initialize_speaker_synthesis(self, speaker_id: int, skip_reinit: bool):
        # 旧Coreはモデルを明示的に読み込むAPIを持たないため、対応の有無を先に確認する。
        if not getattr(self.core, "exist_load_model", True):
            return
        with self.mutex:
            # 再初期化が必要な場合、またはモデルが未読み込みの場合だけロードする。
            if (
                not skip_reinit
                or not getattr(self.core, "exist_is_model_loaded", True)
                or not self.core.is_model_loaded(speaker_id)
            ):
                self.core.load_model(speaker_id)

    def is_initialized_speaker_synthesis(self, speaker_id: int) -> bool:
        if not getattr(self.core, "exist_is_model_loaded", True):
            # 旧Coreは読み込み状態を取得できず、必要なモデルは合成時に読み込まれる。
            return True
        return self.core.is_model_loaded(speaker_id)

    def replace_phoneme_length(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        """Coreで音素長を推論し、入力アクセント句の各モーラを直接更新する。

        Parameters
        ----------
        accent_phrases : list[AccentPhrase]
            直接更新するアクセント句のリスト
        speaker_id : int
            話者スタイルID

        Returns
        -------
        list[AccentPhrase]
            入力と同じアクセント句リスト
        """
        self.initialize_speaker_synthesis(speaker_id, skip_reinit=True)
        flatten_moras, phoneme_data_list = pre_process(accent_phrases)
        _, _, vowel_indexes_data = split_mora(phoneme_data_list)

        phoneme_list_s = numpy.array(
            [p.phoneme_id for p in phoneme_data_list], dtype=numpy.int64
        )
        with self.mutex:
            phoneme_length = self.core.yukarin_s_forward(
                length=len(phoneme_list_s),
                phoneme_list=phoneme_list_s,
                speaker_id=numpy.array(speaker_id, dtype=numpy.int64).reshape(-1),
            )

        # flatten_morasの要素は元のaccent_phrasesと同じMoraを参照している。
        for i, mora in enumerate(flatten_moras):
            mora.consonant_length = (
                phoneme_length[vowel_indexes_data[i + 1] - 1]
                if mora.consonant is not None
                else None
            )
            mora.vowel_length = phoneme_length[vowel_indexes_data[i + 1]]

        return accent_phrases

    def replace_mora_pitch(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        """Coreでモーラ音高を推論し、入力アクセント句の各モーラを直接更新する。

        Parameters
        ----------
        accent_phrases : list[AccentPhrase]
            直接更新するアクセント句のリスト
        speaker_id : int
            話者スタイルID

        Returns
        -------
        list[AccentPhrase]
            入力と同じアクセント句リスト
        """
        self.initialize_speaker_synthesis(speaker_id, skip_reinit=True)
        # numpy.concatenateは空のリストを受け付けない。
        if len(accent_phrases) == 0:
            return []

        flatten_moras, phoneme_data_list = pre_process(accent_phrases)

        def _create_one_hot(accent_phrase: AccentPhrase, position: int):
            """指定モーラの位置だけを1にし、休止を持つ句には末尾の0を追加する。"""
            one_hot = numpy.zeros(len(accent_phrase.moras), dtype=numpy.int64)
            one_hot[position] = 1
            return (
                numpy.r_[one_hot, 0]
                if accent_phrase.pause_mora is not None
                else one_hot
            )

        start_accent_list = numpy.concatenate(
            [
                # accentは1始まり。先頭アクセントとそれ以外で上昇位置が異なる。
                _create_one_hot(accent_phrase, 0 if accent_phrase.accent == 1 else 1)
                for accent_phrase in accent_phrases
            ]
        )

        end_accent_list = numpy.concatenate(
            [
                # accentは1始まりのため、配列の位置へ変換する。
                _create_one_hot(accent_phrase, accent_phrase.accent - 1)
                for accent_phrase in accent_phrases
            ]
        )

        start_accent_phrase_list = numpy.concatenate(
            [_create_one_hot(accent_phrase, 0) for accent_phrase in accent_phrases]
        )

        end_accent_phrase_list = numpy.concatenate(
            [_create_one_hot(accent_phrase, -1) for accent_phrase in accent_phrases]
        )

        # 音素列の前後に追加したpauの位置を0で埋める。
        start_accent_list = numpy.r_[0, start_accent_list, 0]
        end_accent_list = numpy.r_[0, end_accent_list, 0]
        start_accent_phrase_list = numpy.r_[0, start_accent_phrase_list, 0]
        end_accent_phrase_list = numpy.r_[0, end_accent_phrase_list, 0]

        start_accent_list = numpy.array(start_accent_list, dtype=numpy.int64)
        end_accent_list = numpy.array(end_accent_list, dtype=numpy.int64)
        start_accent_phrase_list = numpy.array(
            start_accent_phrase_list, dtype=numpy.int64
        )
        end_accent_phrase_list = numpy.array(end_accent_phrase_list, dtype=numpy.int64)

        (
            consonant_phoneme_data_list,
            vowel_phoneme_data_list,
            _,
        ) = split_mora(phoneme_data_list)

        vowel_phoneme_list = numpy.array(
            [p.phoneme_id for p in vowel_phoneme_data_list], dtype=numpy.int64
        )
        consonant_phoneme_list = numpy.array(
            [
                p.phoneme_id if p is not None else -1
                for p in consonant_phoneme_data_list
            ],
            dtype=numpy.int64,
        )

        with self.mutex:
            f0_list = self.core.yukarin_sa_forward(
                length=vowel_phoneme_list.shape[0],
                vowel_phoneme_list=vowel_phoneme_list[numpy.newaxis],
                consonant_phoneme_list=consonant_phoneme_list[numpy.newaxis],
                start_accent_list=start_accent_list[numpy.newaxis],
                end_accent_list=end_accent_list[numpy.newaxis],
                start_accent_phrase_list=start_accent_phrase_list[numpy.newaxis],
                end_accent_phrase_list=end_accent_phrase_list[numpy.newaxis],
                speaker_id=numpy.array(speaker_id, dtype=numpy.int64).reshape(-1),
            )[0]

        # 無声母音と休止にはF0を設定しない。
        for i, p in enumerate(vowel_phoneme_data_list):
            if p.phoneme in unvoiced_mora_phoneme_list:
                f0_list[i] = 0

        # flatten_morasの要素は元のaccent_phrasesと同じMoraを参照している。
        for i, mora in enumerate(flatten_moras):
            mora.pitch = f0_list[i + 1]

        return accent_phrases

    def _synthesis_impl(self, query: AudioQuery, speaker_id: int):
        """AudioQueryをCoreの入力へ変換し、音声波形を生成する。"""
        self.initialize_speaker_synthesis(speaker_id, skip_reinit=True)
        flatten_moras, phoneme_data_list = pre_process(query.accent_phrases)

        phoneme_list_s = numpy.array(
            [p.phoneme_id for p in phoneme_data_list], dtype=numpy.int64
        )

        # 音素長には、前後の無音時間も含める。
        phoneme_length_list = (
            [query.prePhonemeLength]
            + [
                length
                for mora in flatten_moras
                for length in (
                    [mora.consonant_length] if mora.consonant is not None else []
                )
                + [mora.vowel_length]
            ]
            + [query.postPhonemeLength]
        )
        phoneme_length = numpy.array(phoneme_length_list, dtype=numpy.float32)

        # 話速はすべての音素長へ一様に適用する。
        phoneme_length /= query.speedScale

        f0_list = [0] + [mora.pitch for mora in flatten_moras] + [0]
        f0 = numpy.array(f0_list, dtype=numpy.float32)
        # pitchScaleはオクターブ単位で指定される。
        f0 *= 2**query.pitchScale

        voiced = f0 > 0
        # 無声音だけのクエリでは平均を計算せず、F0=0を保つ。
        if numpy.any(voiced):
            mean_f0 = f0[voiced].mean()
            f0[voiced] = (f0[voiced] - mean_f0) * query.intonationScale + mean_f0

        _, _, vowel_indexes_data = split_mora(phoneme_data_list)
        vowel_indexes = numpy.array(vowel_indexes_data)

        # 秒単位の音素長を、Coreが扱うフレーム数へ変換する。
        rate = 24000 / 256
        phoneme_bin_num = numpy.round(phoneme_length * rate).astype(numpy.int32)

        phoneme = numpy.repeat(phoneme_list_s, phoneme_bin_num)
        f0 = numpy.repeat(
            f0,
            [a.sum() for a in numpy.split(phoneme_bin_num, vowel_indexes[:-1] + 1)],
        )

        # Coreへ渡す音素IDをone-hot表現へ変換する。
        array = numpy.zeros((len(phoneme), OjtPhoneme.num_phoneme), dtype=numpy.float32)
        array[numpy.arange(len(phoneme)), phoneme] = 1
        phoneme = array

        with self.mutex:
            wave = self.core.decode_forward(
                length=phoneme.shape[0],
                phoneme_size=phoneme.shape[1],
                f0=f0[:, numpy.newaxis],
                phoneme=phoneme,
                speaker_id=numpy.array(speaker_id, dtype=numpy.int64).reshape(-1),
            )

        wave *= query.volumeScale

        # Coreの出力は24kHzのため、指定されたサンプリングレートへ変換する。
        if query.outputSamplingRate != self.default_sampling_rate:
            wave = resample(
                wave,
                query.outputSamplingRate * len(wave) // self.default_sampling_rate,
            )

        if query.outputStereo:
            wave = numpy.array([wave, wave]).T

        return wave
