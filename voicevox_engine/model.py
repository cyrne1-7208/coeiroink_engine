from __future__ import annotations

from enum import Enum
from re import findall, fullmatch
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .metas.metas import Speaker, SpeakerInfo


class Mora(BaseModel):
    """
    モーラ（子音＋母音）ごとの情報
    """

    text: str = Field(title="文字")
    consonant: str | None = Field(default=None, title="子音の音素")
    consonant_length: float | None = Field(default=None, title="子音の音長")
    vowel: str = Field(title="母音の音素")
    vowel_length: float = Field(title="母音の音長")
    pitch: float = Field(
        title="音高"
    )  # デフォルト値をつけるとts側のOpenAPIで生成されたコードの型がOptionalになる

    def __hash__(self):
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class AccentPhrase(BaseModel):
    """
    アクセント句ごとの情報
    """

    moras: list[Mora] = Field(title="モーラのリスト")
    accent: int = Field(title="アクセント箇所")
    pause_mora: Mora | None = Field(default=None, title="後ろに無音を付けるかどうか")
    is_interrogative: bool = Field(default=False, title="疑問系かどうか")

    def __hash__(self):
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class AudioQuery(BaseModel):
    """
    音声合成用のクエリ
    """

    accent_phrases: list[AccentPhrase] = Field(title="アクセント句のリスト")
    speedScale: float = Field(title="全体の話速")
    pitchScale: float = Field(title="全体の音高")
    intonationScale: float = Field(title="全体の抑揚")
    volumeScale: float = Field(title="全体の音量")
    prePhonemeLength: float = Field(title="音声の前の無音時間")
    postPhonemeLength: float = Field(title="音声の後の無音時間")
    pauseLength: float | None = Field(
        default=None,
        title="句読点などの無音時間。nullのときは無視される",
    )
    pauseLengthScale: float = Field(default=1, title="句読点などの無音時間（倍率）")
    outputSamplingRate: int = Field(title="音声データの出力サンプリングレート")
    outputStereo: bool = Field(title="音声データをステレオ出力するか否か")
    kana: str | None = Field(
        default=None,
        title="[読み取り専用]AquesTalkライクな読み仮名。音声合成クエリとしては無視される",
    )

    def __hash__(self):
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class ParseKanaErrorCode(Enum):
    UNKNOWN_TEXT = "判別できない読み仮名があります: {text}"
    ACCENT_TOP = "句頭にアクセントは置けません: {text}"
    ACCENT_TWICE = "1つのアクセント句に二つ以上のアクセントは置けません: {text}"
    ACCENT_NOTFOUND = "アクセントを指定していないアクセント句があります: {text}"
    EMPTY_PHRASE = "{position}番目のアクセント句が空白です"
    INTERROGATION_MARK_NOT_AT_END = "アクセント句末以外に「？」は置けません: {text}"
    INFINITE_LOOP = "処理時に無限ループになってしまいました...バグ報告をお願いします。"


class ParseKanaError(Exception):
    def __init__(self, errcode: ParseKanaErrorCode, **kwargs):
        self.errcode = errcode
        self.errname = errcode.name
        self.kwargs: dict[str, str] = kwargs
        err_fmt: str = errcode.value
        self.text = err_fmt.format(**kwargs)


class ParseKanaBadRequest(BaseModel):
    text: str = Field(title="エラーメッセージ")
    error_name: str = Field(
        title="エラー名",
        description="|name|description|\n|---|---|\n"
        + "\n".join(
            [f"| {err.name} | {err.value} |" for err in list(ParseKanaErrorCode)]
        ),
    )
    error_args: dict[str, str] = Field(title="エラーを起こした箇所")

    def __init__(self, err: ParseKanaError):
        super().__init__(text=err.text, error_name=err.errname, error_args=err.kwargs)


class MorphableTargetInfo(BaseModel):
    is_morphable: bool = Field(title="指定した話者に対してモーフィングの可否")
    # FIXME: add reason property
    # reason: Optional[str] = Field(title="is_morphableがfalseである場合、その理由")


class ResourceFormat(str, Enum):
    BASE64 = "base64"
    URL = "url"


class SpeakerNotFoundError(LookupError):
    def __init__(self, speaker: int, *args: object, **kywrds: object) -> None:
        self.speaker = speaker
        super().__init__(f"speaker {speaker} is not found.", *args, **kywrds)


class DownloadableModel(BaseModel):
    """
    ダウンロード可能な音声ライブラリの情報（最新情報をwebで取得することを考慮して、ローカルの情報はない）
    """

    download_path: str = Field(title="音声ライブラリのダウンロードURL")
    volume: str = Field(title="音声ライブラリのバイト数")
    speaker: Speaker = Field(title="話者情報")
    speaker_info: SpeakerInfo = Field(title="話者の追加情報")


class DownloadInfo(BaseModel):
    """
    ダウンロード話者情報（ローカルの情報が付与されたもの）
    """

    downloadable_model: DownloadableModel = Field(title="ダンロードモデル情報")
    current_version: str = Field(title="現在のバージョン")
    character_exists: bool = Field(
        title="このキャラクターのライブラリがローカルに存在するか"
    )
    latest_model_exists: bool = Field(title="最新モデルがローカルに存在するか")


USER_DICT_MIN_PRIORITY = 0
USER_DICT_MAX_PRIORITY = 10


class UserDictWord(BaseModel):
    """
    辞書のコンパイルに使われる情報
    """

    surface: str = Field(title="表層形")
    priority: Annotated[
        int,
        Field(ge=USER_DICT_MIN_PRIORITY, le=USER_DICT_MAX_PRIORITY, title="優先度"),
    ]
    context_id: int = Field(title="文脈ID", default=1348)
    part_of_speech: str = Field(title="品詞")
    part_of_speech_detail_1: str = Field(title="品詞細分類1")
    part_of_speech_detail_2: str = Field(title="品詞細分類2")
    part_of_speech_detail_3: str = Field(title="品詞細分類3")
    inflectional_type: str = Field(title="活用型")
    inflectional_form: str = Field(title="活用形")
    stem: str = Field(title="原形")
    yomi: str = Field(title="読み")
    pronunciation: str = Field(title="発音")
    accent_type: int = Field(title="アクセント型")
    mora_count: int | None = Field(default=None, title="モーラ数")
    accent_associative_rule: str = Field(title="アクセント結合規則")

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    @field_validator("surface")
    def convert_to_zenkaku(cls, surface):
        return surface.translate(
            str.maketrans(
                "".join(chr(0x21 + i) for i in range(94)),
                "".join(chr(0xFF01 + i) for i in range(94)),
            )
        )

    @field_validator("pronunciation", mode="before")
    def check_is_katakana(cls, pronunciation):
        if not fullmatch(r"[ァ-ヴー]+", pronunciation):
            raise ValueError("発音は有効なカタカナでなくてはいけません。")
        sutegana = ["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ", "ッ"]
        for i in range(len(pronunciation)):
            # 「キャット」のように捨て仮名が連続し得るため、「ッ」は後続も含めて判定する。
            if (
                pronunciation[i] in sutegana
                and i < len(pronunciation) - 1
                and (
                    pronunciation[i + 1] in sutegana[:-1]
                    or (
                        pronunciation[i] == sutegana[-1]
                        and pronunciation[i + 1] == sutegana[-1]
                    )
                )
            ):
                raise ValueError("無効な発音です。(捨て仮名の連続)")
            if (
                pronunciation[i] == "ヮ"
                and i != 0
                and pronunciation[i - 1] not in ["ク", "グ"]
            ):
                raise ValueError("無効な発音です。(「くゎ」「ぐゎ」以外の「ゎ」の使用)")
        return pronunciation

    @field_validator("mora_count", mode="before")
    def check_mora_count_and_accent_type(cls, mora_count, info: ValidationInfo):
        """発音表記からモーラ数を補完し、アクセント位置がその範囲内か検証する。"""

        values = info.data
        if "pronunciation" not in values or "accent_type" not in values:
            # 適切な場所でエラーを出すようにする
            return mora_count

        if mora_count is None:
            # 辞書のモーラ数をkana_parserの最長一致表と揃え、キィやクォなどの新しい外来音表記も1モーラとして数える。
            rule_others = (
                "[イ][ェ]|[ヴ][ャュョ]|[ウクグトド][ゥ]|[テデ][ィェャュョ]|[クグ][ヮ]"
            )
            rule_line_i = "[キシチニヒミリギジヂビピ][ェャュョ]|[キニヒミリギビピ][ィ]"
            rule_line_u = "[クツフヴグ][ァ]|[ウクスツフヴグズ][ィ]|[ウクツフヴグ][ェォ]"
            rule_one_mora = "[ァ-ヴー]"
            mora_count = len(
                findall(
                    f"(?:{rule_others}|{rule_line_i}|{rule_line_u}|{rule_one_mora})",
                    values["pronunciation"],
                )
            )

        if not 0 <= values["accent_type"] <= mora_count:
            raise ValueError(
                "誤ったアクセント型です({})。 expect: 0 <= accent_type <= {}".format(
                    values["accent_type"], mora_count
                )
            )
        return mora_count


class PartOfSpeechDetail(BaseModel):
    """
    品詞ごとの情報
    """

    part_of_speech: str = Field(title="品詞")
    part_of_speech_detail_1: str = Field(title="品詞細分類1")
    part_of_speech_detail_2: str = Field(title="品詞細分類2")
    part_of_speech_detail_3: str = Field(title="品詞細分類3")
    # context_idは辞書の左・右文脈IDのこと
    # https://github.com/VOICEVOX/open_jtalk/blob/427cfd761b78efb6094bea3c5bb8c968f0d711ab/src/mecab-naist-jdic/_left-id.def
    context_id: int = Field(title="文脈ID")
    cost_candidates: list[int] = Field(title="コストのパーセンタイル")
    accent_associative_rules: list[str] = Field(title="アクセント結合規則の一覧")


class WordTypes(str, Enum):
    """
    fastapiでword_type引数を検証する時に使用するクラス
    """

    PROPER_NOUN = "PROPER_NOUN"
    COMMON_NOUN = "COMMON_NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    SUFFIX = "SUFFIX"


class SupportedDevicesInfo(BaseModel):
    """
    対応しているデバイスの情報
    """

    cpu: bool = Field(title="CPUに対応しているか")
    cuda: bool = Field(title="CUDA(Nvidia GPU)に対応しているか")
    dml: bool = Field(title="DirectML(Nvidia GPU/Radeon GPU等)に対応しているか")


class SupportedFeaturesInfo(BaseModel):
    """
    エンジンの機能の情報
    """

    support_adjusting_mora: bool = Field(title="モーラが調整可能かどうか")
    support_adjusting_speed_scale: bool = Field(title="話速が調整可能かどうか")
    support_adjusting_pitch_scale: bool = Field(title="音高が調整可能かどうか")
    support_adjusting_intonation_scale: bool = Field(title="抑揚が調整可能かどうか")
    support_adjusting_volume_scale: bool = Field(title="音量が調整可能かどうか")
    support_adjusting_silence_scale: bool = Field(
        title="前後の無音時間が調節可能かどうか"
    )
    support_interrogative_upspeak: bool = Field(
        title="疑似疑問文に対応しているかどうか"
    )
    support_switching_device: bool = Field(title="CPU/GPUの切り替えが可能かどうか")
