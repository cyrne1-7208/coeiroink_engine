from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain

import pyopenjtalk

from .katakana_english import text_to_full_context_labels
from .user_dict import mutex_openjtalk_dict


@dataclass
class Phoneme:
    """
    音素の元となるcontextを保持する音素クラス
    音素には母音や子音のほかに無音を表すsilentとpauseも含まれる。

    Attributes
    ----------
    contexts : dict[str, str]
        音素の元
    """

    contexts: dict[str, str]

    @classmethod
    def from_label(cls, label: str) -> Phoneme:
        """
        pyopenjtalk.extract_fullcontextで得られる音素の元であるラベルから、Phonemeクラスを作成する。

        Parameters
        ----------
        label : str
            pyopenjtalk.extract_fullcontextで得られるラベル

        Returns
        -------
        phoneme : Phoneme
            作成したPhonemeクラス
        """
        # フルコンテキストラベルの仕様は、http://hts.sp.nitech.ac.jp/?Download のHTS-2.3 Japanese tar.bz2に含まれるdata/lab_format.pdfに記載されている。
        contexts = re.search(
            r"^(?P<p1>.+?)\^(?P<p2>.+?)\-(?P<p3>.+?)\+(?P<p4>.+?)\=(?P<p5>.+?)"
            r"/A\:(?P<a1>.+?)\+(?P<a2>.+?)\+(?P<a3>.+?)"
            r"/B\:(?P<b1>.+?)\-(?P<b2>.+?)\_(?P<b3>.+?)"
            r"/C\:(?P<c1>.+?)\_(?P<c2>.+?)\+(?P<c3>.+?)"
            r"/D\:(?P<d1>.+?)\+(?P<d2>.+?)\_(?P<d3>.+?)"
            r"/E\:(?P<e1>.+?)\_(?P<e2>.+?)\!(?P<e3>.+?)\_(?P<e4>.+?)\-(?P<e5>.+?)"
            r"/F\:(?P<f1>.+?)\_(?P<f2>.+?)\#(?P<f3>.+?)\_(?P<f4>.+?)\@(?P<f5>.+?)\_(?P<f6>.+?)\|(?P<f7>.+?)\_(?P<f8>.+?)"
            r"/G\:(?P<g1>.+?)\_(?P<g2>.+?)\%(?P<g3>.+?)\_(?P<g4>.+?)\_(?P<g5>.+?)"
            r"/H\:(?P<h1>.+?)\_(?P<h2>.+?)"
            r"/I\:(?P<i1>.+?)\-(?P<i2>.+?)\@(?P<i3>.+?)\+(?P<i4>.+?)\&(?P<i5>.+?)\-(?P<i6>.+?)\|(?P<i7>.+?)\+(?P<i8>.+?)"
            r"/J\:(?P<j1>.+?)\_(?P<j2>.+?)"
            r"/K\:(?P<k1>.+?)\+(?P<k2>.+?)\-(?P<k3>.+?)$",
            label,
        ).groupdict()
        return cls(contexts=contexts)

    @property
    def label(self) -> str:
        """
        pyopenjtalk.extract_fullcontextで得られるものと等しいラベルを返す。

        Returns
        -------
        label : str
            contextsから組み立てたラベル
        """
        return (
            "{p1}^{p2}-{p3}+{p4}={p5}"
            "/A:{a1}+{a2}+{a3}"
            "/B:{b1}-{b2}_{b3}"
            "/C:{c1}_{c2}+{c3}"
            "/D:{d1}+{d2}_{d3}"
            "/E:{e1}_{e2}!{e3}_{e4}-{e5}"
            "/F:{f1}_{f2}#{f3}_{f4}@{f5}_{f6}|{f7}_{f8}"
            "/G:{g1}_{g2}%{g3}_{g4}_{g5}"
            "/H:{h1}_{h2}"
            "/I:{i1}-{i2}@{i3}+{i4}&{i5}-{i6}|{i7}+{i8}"
            "/J:{j1}_{j2}"
            "/K:{k1}+{k2}-{k3}"
        ).format(**self.contexts)

    @property
    def phoneme(self) -> str:
        """
        音素クラスの中で発声に必要な要素を返す。

        Returns
        -------
        phoneme : str
            発声に必要な要素
        """
        return self.contexts["p3"]

    def is_pause(self) -> bool:
        """
        音素が無音を表すsilentかpauseであるかを返す。

        Returns
        -------
        is_pause : bool
            音素が無音であればTrue、そうでなければFalse
        """
        return self.contexts["f1"] == "xx"

    def __repr__(self) -> str:
        return f"<Phoneme phoneme='{self.phoneme}'>"


@dataclass
class Mora:
    """
    モーラクラス
    モーラは、母音や促音「っ」、撥音「ん」などの1音素か、母音と子音を組み合わせた2音素で成り立つ。

    Attributes
    ----------
    consonant : Phoneme | None
        子音
    vowel : Phoneme
        母音
    """

    consonant: Phoneme | None
    vowel: Phoneme

    def set_context(self, key: str, value: str) -> None:
        """
        Moraクラス内に含まれるPhonemeのcontextのうち、指定されたキーの値を変更する。
        consonantが存在する場合は、vowelと同じようにcontextを変更する。

        Parameters
        ----------
        key : str
            変更したいcontextのキー
        value : str
            変更したいcontextの値
        """
        self.vowel.contexts[key] = value
        if self.consonant is not None:
            self.consonant.contexts[key] = value

    @property
    def phonemes(self) -> list[Phoneme]:
        """
        音素群を返す。

        Returns
        -------
        phonemes : list[Phoneme]
            母音しかない場合は母音のみ、子音もある場合は子音、母音の順番に並べたPhonemeのリスト
        """
        if self.consonant is not None:
            return [self.consonant, self.vowel]
        return [self.vowel]

    @property
    def labels(self) -> list[str]:
        """
        ラベル群を返す。

        Returns
        -------
        labels : list[str]
            Moraに含まれるすべてのラベル
        """
        return [p.label for p in self.phonemes]


@dataclass
class AccentPhrase:
    """
    アクセント句クラス
    同じアクセントのMoraを複数保持する。

    Attributes
    ----------
    moras : list[Mora]
        モーラのリスト
    accent : int
        アクセント
    is_interrogative : bool
        疑問形のアクセント句であるか
    """

    moras: list[Mora]
    accent: int
    is_interrogative: bool

    @classmethod
    def from_phonemes(cls, phonemes: list[Phoneme]) -> AccentPhrase:
        """
        PhonemeのリストからAccentPhraseクラスを作成する。

        Parameters
        ----------
        phonemes : list[Phoneme]
            phonemeのリスト

        Returns
        -------
        accent_phrase : AccentPhrase
            作成したAccentPhraseクラス
        """
        moras: list[Mora] = []

        mora_phonemes: list[Phoneme] = []
        for phoneme, next_phoneme in zip(phonemes, [*phonemes[1:], None], strict=True):
            # Hihosiba/voicevox_engine#57の回避策
            # Open JTalkによるアクセント句内のモーラへの附番は49番目までである。
            # 49番目のモーラについては、続く音素のモーラ番号を単一モーラの特定に使えない。
            if int(phoneme.contexts["a2"]) == 49:
                break

            mora_phonemes.append(phoneme)

            if (
                next_phoneme is None
                or phoneme.contexts["a2"] != next_phoneme.contexts["a2"]
            ):
                if len(mora_phonemes) == 1:
                    consonant, vowel = None, mora_phonemes[0]
                elif len(mora_phonemes) == 2:
                    consonant, vowel = mora_phonemes[0], mora_phonemes[1]
                else:
                    raise ValueError(mora_phonemes)
                mora = Mora(consonant=consonant, vowel=vowel)
                moras.append(mora)
                mora_phonemes = []

        accent = int(moras[0].vowel.contexts["f2"])
        # Hihosiba/voicevox_engine#55の回避策
        # アクセント位置とするキーf2の値が、アクセント句内のモーラ数を超える場合がある。
        accent = min(accent, len(moras))
        is_interrogative = moras[-1].vowel.contexts["f3"] == "1"
        return cls(moras=moras, accent=accent, is_interrogative=is_interrogative)

    def set_context(self, key: str, value: str) -> None:
        """
        AccentPhraseに間接的に含まれる全てのPhonemeのcontextのうち、指定されたキーの値を変更する。

        Parameters
        ----------
        key : str
            変更したいcontextのキー
        value : str
            変更したいcontextの値
        """
        for mora in self.moras:
            mora.set_context(key, value)

    @property
    def phonemes(self) -> list[Phoneme]:
        """
        音素群を返す。

        Returns
        -------
        phonemes : list[Phoneme]
            AccentPhraseに間接的に含まれる全てのPhoneme
        """
        return list(chain.from_iterable(m.phonemes for m in self.moras))

    @property
    def labels(self) -> list[str]:
        """
        ラベル群を返す。

        Returns
        -------
        labels : list[str]
            AccentPhraseに間接的に含まれる全てのラベル
        """
        return [p.label for p in self.phonemes]

    def merge(self, accent_phrase: AccentPhrase) -> AccentPhrase:
        """
        AccentPhraseを結合する。
        このクラスが保持するmorasの後ろに、引数として渡されたAccentPhraseのmorasを結合する。

        Parameters
        ----------
        accent_phrase : AccentPhrase
            結合対象のAccentPhrase

        Returns
        -------
        accent_phrase : AccentPhrase
            結合後のAccentPhrase
        """
        return AccentPhrase(
            moras=self.moras + accent_phrase.moras,
            accent=self.accent,
            is_interrogative=accent_phrase.is_interrogative,
        )


@dataclass
class BreathGroup:
    """
    発声の区切りクラス
    ポーズ間の区間である呼気段落に含まれる複数のアクセント句を保持する。

    Attributes
    ----------
    accent_phrases : list[AccentPhrase]
        アクセント句のリスト
    """

    accent_phrases: list[AccentPhrase]

    @classmethod
    def from_phonemes(cls, phonemes: list[Phoneme]) -> BreathGroup:
        """
        PhonemeのリストからBreathGroupクラスを作成する。

        Parameters
        ----------
        phonemes : list[Phoneme]
            phonemeのリスト

        Returns
        -------
        breath_group : BreathGroup
            作成したBreathGroupクラス
        """
        accent_phrases: list[AccentPhrase] = []
        accent_phonemes: list[Phoneme] = []
        for phoneme, next_phoneme in zip(phonemes, [*phonemes[1:], None], strict=True):
            accent_phonemes.append(phoneme)

            if (
                next_phoneme is None
                or phoneme.contexts["i3"] != next_phoneme.contexts["i3"]
                or phoneme.contexts["f5"] != next_phoneme.contexts["f5"]
            ):
                accent_phrase = AccentPhrase.from_phonemes(accent_phonemes)
                accent_phrases.append(accent_phrase)
                accent_phonemes = []

        return cls(accent_phrases=accent_phrases)

    def set_context(self, key: str, value: str) -> None:
        """
        BreathGroupに間接的に含まれる全てのPhonemeのcontextのうち、指定されたキーの値を変更する。

        Parameters
        ----------
        key : str
            変更したいcontextのキー
        value : str
            変更したいcontextの値
        """
        for accent_phrase in self.accent_phrases:
            accent_phrase.set_context(key, value)

    @property
    def phonemes(self) -> list[Phoneme]:
        """
        音素群を返す。

        Returns
        -------
        phonemes : list[Phoneme]
            BreathGroupに間接的に含まれる全てのPhoneme
        """
        return list(
            chain.from_iterable(
                accent_phrase.phonemes for accent_phrase in self.accent_phrases
            )
        )

    @property
    def labels(self) -> list[str]:
        """
        ラベル群を返す。

        Returns
        -------
        labels : list[str]
            BreathGroupに間接的に含まれる全てのラベル
        """
        return [p.label for p in self.phonemes]


@dataclass
class Utterance:
    """
    発声クラス
    発声の区切りと無音を複数保持する。

    Attributes
    ----------
    breath_groups : list[BreathGroup]
        発声の区切りのリスト
    pauses : list[Phoneme]
        無音のリスト
    """

    breath_groups: list[BreathGroup]
    pauses: list[Phoneme]

    @classmethod
    def from_phonemes(cls, phonemes: list[Phoneme]) -> Utterance:
        """
        Phonemeの完全なリストからUtteranceクラスを作成する。

        Parameters
        ----------
        phonemes : list[Phoneme]
            phonemeのリスト

        Returns
        -------
        utterance : Utterance
            作成したUtteranceクラス
        """
        pauses: list[Phoneme] = []

        breath_groups: list[BreathGroup] = []
        group_phonemes: list[Phoneme] = []
        for phoneme in phonemes:
            if not phoneme.is_pause():
                group_phonemes.append(phoneme)

            else:
                pauses.append(phoneme)

                if len(group_phonemes) > 0:
                    breath_group = BreathGroup.from_phonemes(group_phonemes)
                    breath_groups.append(breath_group)
                    group_phonemes = []

        return cls(breath_groups=breath_groups, pauses=pauses)

    def set_context(self, key: str, value: str) -> None:
        """
        Utteranceに間接的に含まれる全てのPhonemeのcontextのうち、指定されたキーの値を変更する。

        Parameters
        ----------
        key : str
            変更したいcontextのキー
        value : str
            変更したいcontextの値
        """
        for breath_group in self.breath_groups:
            breath_group.set_context(key, value)

    @property
    def phonemes(self) -> list[Phoneme]:
        """
        音素群を返す。

        Returns
        -------
        phonemes : list[Phoneme]
            Utteranceクラスに直接的・間接的に含まれる全てのPhoneme
        """
        accent_phrases = list(
            chain.from_iterable(
                breath_group.accent_phrases for breath_group in self.breath_groups
            )
        )
        for prev, cent, post in zip(
            [None, *accent_phrases[:-1]],
            accent_phrases,
            [*accent_phrases[1:], None],
            strict=True,
        ):
            mora_num = len(cent.moras)
            accent = cent.accent

            if prev is not None:
                prev.set_context("g1", str(mora_num))
                prev.set_context("g2", str(accent))

            if post is not None:
                post.set_context("e1", str(mora_num))
                post.set_context("e2", str(accent))

            cent.set_context("f1", str(mora_num))
            cent.set_context("f2", str(accent))
            for i_mora, mora in enumerate(cent.moras):
                mora.set_context("a1", str(i_mora - accent + 1))
                mora.set_context("a2", str(i_mora + 1))
                mora.set_context("a3", str(mora_num - i_mora))

        for prev, cent, post in zip(
            [None, *self.breath_groups[:-1]],
            self.breath_groups,
            [*self.breath_groups[1:], None],
            strict=True,
        ):
            accent_phrase_num = len(cent.accent_phrases)

            if prev is not None:
                prev.set_context("j1", str(accent_phrase_num))

            if post is not None:
                post.set_context("h1", str(accent_phrase_num))

            cent.set_context("i1", str(accent_phrase_num))
            cent.set_context(
                "i5", str(accent_phrases.index(cent.accent_phrases[0]) + 1)
            )
            cent.set_context(
                "i6",
                str(len(accent_phrases) - accent_phrases.index(cent.accent_phrases[0])),
            )

        self.set_context(
            "k2",
            str(
                sum(
                    [
                        len(breath_group.accent_phrases)
                        for breath_group in self.breath_groups
                    ]
                )
            ),
        )

        phonemes: list[Phoneme] = []
        for i in range(len(self.pauses)):
            if self.pauses[i] is not None:
                phonemes += [self.pauses[i]]

            if i < len(self.pauses) - 1:
                phonemes += self.breath_groups[i].phonemes

        return phonemes

    @property
    def labels(self) -> list[str]:
        """
        ラベル群を返す。

        Returns
        -------
        labels : list[str]
            Utteranceクラスに直接的・間接的に含まれる全てのラベル
        """
        return [p.label for p in self.phonemes]


def extract_full_context_label(
    text: str, enable_katakana_english: bool = False
) -> Utterance:
    """
    テキストをOpen JTalkで解析し、Utteranceクラスを作成する。

    Parameters
    ----------
    text : str
        解析するテキスト
    enable_katakana_english : bool
        未知の英単語にカタカナ読みを補うか
        Falseの場合はCOEIROINKの従来の解析結果を保つ。

    Returns
    -------
    utterance : Utterance
        解析結果から作成したUtteranceクラス
    """
    # 辞書の再構築・切替とOpen JTalk解析を同時に走らせない。
    with mutex_openjtalk_dict:
        labels = (
            text_to_full_context_labels(text)
            if enable_katakana_english
            else pyopenjtalk.extract_fullcontext(text)
        )
    phonemes = [Phoneme.from_label(label=label) for label in labels]
    return Utterance.from_phonemes(phonemes)
