import copy
from abc import ABCMeta, abstractmethod

import numpy as np

from ..model import AccentPhrase, AudioQuery, Mora
from ..mora_list import openjtalk_mora2text
from ..text_analysis import (
    analyze_text,
    full_context_label_moras_to_moras,
    mora_to_text,
)

__all__ = ["SynthesisEngineBase", "full_context_label_moras_to_moras", "mora_to_text"]


def adjust_interrogative_accent_phrases(
    accent_phrases: list[AccentPhrase],
) -> list[AccentPhrase]:
    """疑問文に指定されたアクセント句の末尾へ、音高を上げた疑問形発音用モーラを追加する。"""
    return [
        AccentPhrase(
            moras=adjust_interrogative_moras(accent_phrase),
            accent=accent_phrase.accent,
            pause_mora=accent_phrase.pause_mora,
            is_interrogative=accent_phrase.is_interrogative,
        )
        for accent_phrase in accent_phrases
    ]


def adjust_interrogative_moras(accent_phrase: AccentPhrase) -> list[Mora]:
    moras = copy.deepcopy(accent_phrase.moras)
    if accent_phrase.is_interrogative and not (len(moras) == 0 or moras[-1].pitch == 0):
        interrogative_mora = make_interrogative_mora(moras[-1])
        moras.append(interrogative_mora)
        return moras
    return moras


def make_interrogative_mora(last_mora: Mora) -> Mora:
    fix_vowel_length = 0.15
    adjust_pitch = 0.3
    max_pitch = 6.5
    vowel_key = last_mora.vowel
    if vowel_key not in openjtalk_mora2text:
        vowel_key = vowel_key.lower()
    if vowel_key not in openjtalk_mora2text:
        raise ValueError(f"unsupported interrogative mora vowel: {last_mora.vowel!r}")
    return Mora(
        text=openjtalk_mora2text[vowel_key],
        consonant=None,
        consonant_length=None,
        vowel=last_mora.vowel,
        vowel_length=fix_vowel_length,
        pitch=min(last_mora.pitch + adjust_pitch, max_pitch),
    )


class SynthesisEngineBase(metaclass=ABCMeta):
    # FIXME: JSON文字列ではなくモデルを返すようにする。
    @property
    @abstractmethod
    def speakers(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_devices(self) -> str | None:
        raise NotImplementedError

    def initialize_speaker_synthesis(self, speaker_id: int, skip_reinit: bool) -> None:
        """
        指定した話者での音声合成を初期化する。何度も実行可能。
        未実装の場合は何もしない
        Parameters
        ----------
        speaker_id : int
            話者ID
        skip_reinit : bool
            Trueの場合、既に初期化済みの話者の再初期化をスキップする
        """
        return

    def is_initialized_speaker_synthesis(self, speaker_id: int) -> bool:
        """
        指定した話者での音声合成が初期化されているかどうかを返す
        Parameters
        ----------
        speaker_id : int
            話者ID
        Returns
        -------
        bool
            初期化されているかどうか
        """
        return True

    @abstractmethod
    def replace_phoneme_length(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        """
        accent_phrasesの母音・子音の長さを設定する
        Parameters
        ----------
        accent_phrases : List[AccentPhrase]
            アクセント句モデルのリスト
        speaker_id : int
            話者ID
        Returns
        -------
        accent_phrases : List[AccentPhrase]
            母音・子音の長さが設定されたアクセント句モデルのリスト
        """
        raise NotImplementedError()

    @abstractmethod
    def replace_mora_pitch(
        self, accent_phrases: list[AccentPhrase], speaker_id: int
    ) -> list[AccentPhrase]:
        """
        accent_phrasesの音高(ピッチ)を設定する
        Parameters
        ----------
        accent_phrases : List[AccentPhrase]
            アクセント句モデルのリスト
        speaker_id : int
            話者ID
        Returns
        -------
        accent_phrases : List[AccentPhrase]
            音高(ピッチ)が設定されたアクセント句モデルのリスト
        """
        raise NotImplementedError()

    def replace_mora_data(
        self,
        accent_phrases: list[AccentPhrase],
        speaker_id: int,
    ) -> list[AccentPhrase]:
        """音素長、モーラ音高の順に推論結果をアクセント句へ反映する。"""

        return self.replace_mora_pitch(
            accent_phrases=self.replace_phoneme_length(
                accent_phrases=accent_phrases,
                speaker_id=speaker_id,
            ),
            speaker_id=speaker_id,
        )

    def create_accent_phrases(
        self,
        text: str,
        speaker_id: int,
        enable_katakana_english: bool = False,
    ) -> list[AccentPhrase]:
        """Open JTalkのフルコンテキストラベルをアクセント句へ変換し、継続長と音高を補完する。"""

        return self.replace_mora_data(
            accent_phrases=analyze_text(
                text,
                enable_katakana_english=enable_katakana_english,
            ),
            speaker_id=speaker_id,
        )

    def synthesis(
        self,
        query: AudioQuery,
        speaker_id: int,
        enable_interrogative_upspeak: bool = False,
    ) -> np.ndarray:
        """
        音声合成クエリ内で疑問文に指定されたMoraを変形した後、継承先の`_synthesis_impl`を使って音声合成を行う
        Parameters
        ----------
        query : AudioQuery
            音声合成クエリ
        speaker_id : int
            話者ID
        enable_interrogative_upspeak : bool
            疑問形のテキストの語尾を自動調整する機能を有効にするか
        Returns
        -------
        wave : numpy.ndarray
            音声合成結果
        """
        # モーフィング時などに同一参照のqueryで複数回呼ばれる可能性があるので、元の引数のqueryに破壊的変更を行わない
        query = copy.deepcopy(query)
        if enable_interrogative_upspeak:
            query.accent_phrases = adjust_interrogative_accent_phrases(
                query.accent_phrases
            )
        return self._synthesis_impl(query, speaker_id)

    @abstractmethod
    def _synthesis_impl(self, query: AudioQuery, speaker_id: int) -> np.ndarray:
        """
        音声合成クエリから音声合成に必要な情報を構成し、実際に音声合成を行う
        Parameters
        ----------
        query : AudioQuery
            音声合成クエリ
        speaker_id : int
            話者ID
        Returns
        -------
        wave : numpy.ndarray
            音声合成結果
        """
        raise NotImplementedError()
