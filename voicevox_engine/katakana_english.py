"""pyopenjtalkが未知語とした英単語へカタカナ読みを補う。"""

import re

import kanalizer
import pyopenjtalk

from .user_dict import mutex_openjtalk_dict

_FULLWIDTH_ASCII = str.maketrans(
    "".join(chr(0xFF01 + index) for index in range(94)),
    "".join(chr(0x21 + index) for index in range(94)),
)
_ALPHABET_KANA = {
    "A": "エー",
    "B": "ビー",
    "C": "シー",
    "D": "ディー",
    "E": "イー",
    "F": "エフ",
    "G": "ジー",
    "H": "エイチ",
    "I": "アイ",
    "J": "ジェー",
    "K": "ケー",
    "L": "エル",
    "M": "エム",
    "N": "エヌ",
    "O": "オー",
    "P": "ピー",
    "Q": "キュー",
    "R": "アール",
    "S": "エス",
    "T": "ティー",
    "U": "ユー",
    "V": "ブイ",
    "W": "ダブリュー",
    "X": "エックス",
    "Y": "ワイ",
    "Z": "ズィー",
}
_MORA_PATTERN = re.compile(
    "(?:[イ][ェ]|[ヴ][ャュョ]|[ウクグトド][ゥ]|[テデ][ィェャュョ]|[クグ][ヮ]|"
    "[キシチニヒミリギジヂビピ][ェャュョ]|[キニヒミリギビピ][ィ]|"
    "[クツフヴグ][ァ]|[ウクスツフヴグズ][ィ]|[ウクツフヴグ][ェォ]|[ァ-ヴー])"
)


def _halfwidth_alphabet(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    converted = value.translate(_FULLWIDTH_ASCII)
    return converted if re.fullmatch("[A-Za-z]+", converted) else None


def _alphabet_reading(text: str) -> str:
    result = ""
    for word in re.findall("[A-Za-z][a-z]*", text):
        if len(word) > 1 and word != word.upper():
            result += kanalizer.convert(word.lower())
        else:
            result += "".join(_ALPHABET_KANA[letter.upper()] for letter in word)
    return result


def _unknown_english_feature(feature: dict) -> dict:
    alphabet = _halfwidth_alphabet(feature.get("string"))
    if (
        alphabet is None
        or feature.get("pos") != "フィラー"
        or feature.get("chain_rule") != "*"
    ):
        return feature
    reading = _alphabet_reading(alphabet)
    return {
        "string": feature["string"],
        "pos": "名詞",
        "pos_group1": "固有名詞",
        "pos_group2": "一般",
        "pos_group3": "*",
        "ctype": "*",
        "cform": "*",
        "orig": feature["string"],
        "read": reading,
        "pron": reading,
        "acc": 1,
        "mora_size": len(_MORA_PATTERN.findall(reading)),
        "chain_rule": "*",
        "chain_flag": -1,
    }


def _is_alphabet_feature(feature: dict) -> bool:
    return _halfwidth_alphabet(feature.get("string")) is not None


def text_to_full_context_labels(text: str) -> list[str]:
    """未知英単語を補正してから既存pyopenjtalkのラベルを生成する。"""

    if not text.strip():
        return []
    # run_frontendとmake_labelの間で辞書を切り替えず、一つの解析として扱う。
    with mutex_openjtalk_dict:
        features = [
            _unknown_english_feature(item) for item in pyopenjtalk.run_frontend(text)
        ]
        # Open JTalkが英単語間の全角空白を読点として解析した場合だけ除去し、不要な休止を防ぐ。
        features = [
            feature
            for index, feature in enumerate(features)
            if not (
                feature.get("string") == "　"
                and feature.get("pron") == "、"
                and 0 < index < len(features) - 1
                and _is_alphabet_feature(features[index - 1])
                and _is_alphabet_feature(features[index + 1])
            )
        ]
        return pyopenjtalk.make_label(features)
