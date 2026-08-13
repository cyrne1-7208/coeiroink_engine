"""公開COEIROINK v2話者フォルダ向けのメタデータ処理です。
このモジュールの処理対象は``speaker_info``ディレクトリですが、ディレクトリ名は話者の識別子ではなく``metas.json``の値を識別子として使います。
これは``rintos_ver1.0``のようにディレクトリ名とspeaker UUIDが異なるMYCOEIROINKパッケージで重要です。
HTTP処理とダウンロード処理は含めず、v2ルートが使う話者メタデータとアセットのファイル操作だけを提供します。
"""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from voicevox_engine.metas.MetasStore import MetasStore

from .models import (
    SpeakerInfo,
    SpeakerMeta,
    SpeakerMetaForTextBox,
    SpeakerMetaPathVariant,
    SpeakerMetaStyle,
    SpeakerPolicy,
    StyleInfo,
    StylePathVariant,
)


class MetadataError(Exception):
    """不正な話者メタデータや不足した話者アセットの基底例外です。"""


class SpeakerNotFoundError(MetadataError, LookupError):
    """話者UUIDが話者ディレクトリに存在しない場合に発生します。"""

    def __init__(self, speaker_uuid: str) -> None:
        self.speaker_uuid = speaker_uuid
        super().__init__(f"speakerUuid {speaker_uuid!r} was not found")


class StyleNotFoundError(MetadataError, LookupError):
    """話者に指定スタイルが存在しない場合に発生します。"""

    def __init__(self, speaker_uuid: str, style_id: int) -> None:
        self.speaker_uuid = speaker_uuid
        self.style_id = style_id
        super().__init__(
            f"styleId {style_id!r} was not found for speakerUuid {speaker_uuid!r}"
        )


class AmbiguousStyleError(MetadataError, LookupError):
    """一意な話者UUIDなしで複数話者に存在するスタイルIDを使った場合に発生します。"""

    def __init__(self, style_id: int, speaker_uuids: List[str]) -> None:
        self.style_id = style_id
        self.speaker_uuids = tuple(speaker_uuids)
        joined = ", ".join(repr(uuid) for uuid in speaker_uuids)
        super().__init__(
            f"styleId {style_id!r} is ambiguous; specify speakerUuid "
            f"(matches: {joined})"
        )


class MetadataAssetNotFoundError(MetadataError, FileNotFoundError):
    """必要な肖像・アイコン・規約・ライセンス・サンプルがない場合に発生します。"""

    def __init__(self, path: Path, description: str) -> None:
        self.path = path
        self.description = description
        super().__init__(f"{description} was not found: {path}")


@dataclass(frozen=True)
class _SpeakerRecord:
    metadata: SpeakerMeta
    folder: Path


def _style_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"style_id must be an int, got {value!r}")
    return value


def _sample_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"sample index must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"sample index must be non-negative, got {value!r}")
    return value


def _required_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise MetadataAssetNotFoundError(path, description)
    return path


def _read_bytes(path: Path, description: str) -> bytes:
    path = _required_file(path, description)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MetadataError(f"could not read {description} {path}: {exc}") from exc


def _read_text(path: Path, description: str) -> str:
    try:
        return _read_bytes(path, description).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetadataError(f"{description} is not valid UTF-8: {path}") from exc


def _base64_file(path: Path, description: str) -> str:
    return base64.b64encode(_read_bytes(path, description)).decode("ascii")


def _parse_folder_metadata(raw_metadata: object, meta_path: Path) -> SpeakerMeta:
    """アセットを追加する前に応答モデルでフォルダメタデータを検証します。
    ``metas.json``はモデル情報なのでv2応答に必要なBase64項目を持ちません。
    検証時だけ空のプレースホルダーを使い、応答作成時に実際のアセットファイルを読み込みます。
    """

    if not isinstance(raw_metadata, dict):
        raise MetadataError(f"invalid speaker metadata: {meta_path}: expected an object")
    wire_metadata = dict(raw_metadata)
    wire_metadata.setdefault("base64Portrait", "")
    wire_styles = []
    raw_styles = wire_metadata.get("styles")
    if not isinstance(raw_styles, list):
        raise MetadataError(f"invalid speaker metadata: {meta_path}: styles must be a list")
    for raw_style in raw_styles:
        if not isinstance(raw_style, dict):
            raise MetadataError(
                f"invalid speaker metadata: {meta_path}: each style must be an object"
            )
        wire_style = dict(raw_style)
        wire_style.setdefault("base64Icon", "")
        wire_styles.append(wire_style)
    wire_metadata["styles"] = wire_styles
    try:
        return SpeakerMeta.model_validate(wire_metadata)
    except ValidationError as exc:
        raise MetadataError(f"invalid speaker metadata: {meta_path}: {exc}") from exc


class SpeakerMetadataStore:
    """公開モデルディレクトリからv2話者メタデータとアセットを解決します。
    UUIDとフォルダの対応は``MetasStore``を正とし、各フォルダの``metas.json``をv2通信モデルへ変換してアセットをエンコードします。
    """

    def __init__(
        self, speaker_info_dir: Path, metas_store: Optional[MetasStore] = None
    ) -> None:
        root = Path(speaker_info_dir).expanduser().resolve()
        if not root.is_dir():
            raise MetadataError(f"speaker info directory was not found: {root}")

        try:
            self._metas_store = metas_store or MetasStore(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise MetadataError(f"could not load speaker metadata from {root}: {exc}") from exc

        self._speaker_info_dir = root
        # UUID・メタデータ・フォルダの対応は起動時に検証して固定します。
        self._records: Dict[str, _SpeakerRecord] = {}
        for speaker_uuid in sorted(self._metas_store.loaded_metas):
            try:
                folder = Path(self._metas_store.speaker_path(speaker_uuid)).resolve()
            except KeyError as exc:
                raise MetadataError(
                    f"MetasStore has no folder for speakerUuid {speaker_uuid!r}"
                ) from exc

            meta_path = folder / "metas.json"
            try:
                raw_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = _parse_folder_metadata(raw_metadata, meta_path)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                MetadataError,
            ) as exc:
                raise MetadataError(f"invalid speaker metadata: {meta_path}: {exc}") from exc

            if metadata.speaker_uuid != speaker_uuid:
                raise MetadataError(
                    f"speakerUuid mismatch between MetasStore and {meta_path}: "
                    f"{speaker_uuid!r} != {metadata.speaker_uuid!r}"
                )

            style_ids = [style.style_id for style in metadata.styles]
            if len(style_ids) != len(set(style_ids)):
                raise MetadataError(
                    f"duplicate styleId in speakerUuid {speaker_uuid!r}: {style_ids}"
                )

            ordered_styles = sorted(
                metadata.styles, key=lambda style: (style.style_id, style.style_name)
            )
            self._records[speaker_uuid] = _SpeakerRecord(
                metadata=metadata.model_copy(update={"styles": ordered_styles}),
                folder=folder,
            )

    @property
    def speaker_info_dir(self) -> Path:
        return self._speaker_info_dir

    @property
    def speaker_uuids(self) -> Tuple[str, ...]:
        """すべての話者UUIDを決定的な順序で返します。"""

        return tuple(sorted(self._records))

    def _record(self, speaker_uuid: str) -> _SpeakerRecord:
        try:
            return self._records[speaker_uuid]
        except KeyError as exc:
            raise SpeakerNotFoundError(speaker_uuid) from exc

    def speaker_path(self, speaker_uuid: str) -> Path:
        """``speaker_uuid``に対応する物理フォルダを返します。"""

        return self._record(speaker_uuid).folder

    def lookup_speaker_folder(self, speaker_uuid: str) -> Path:
        """ルートアダプターが使う``speaker_path``の別名です。"""

        return self.speaker_path(speaker_uuid)

    def _style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        style_id = _style_id(style_id)
        record = self._record(speaker_uuid)
        for style in record.metadata.styles:
            if style.style_id == style_id:
                return style
        raise StyleNotFoundError(speaker_uuid, style_id)

    def get_style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        """一意なUUIDとスタイルIDの組み合わせでスタイルを検索します。"""

        return self._style(speaker_uuid, style_id)

    def speaker_meta_for_style(
        self, style_id: int, speaker_uuid: Optional[str] = None
    ) -> SpeakerMetaForTextBox:
        """UUIDとスタイルIDで選択された話者・スタイルの組を返します。"""

        speaker_uuid, style = self.find_style(style_id, speaker_uuid)
        metadata = self._record(speaker_uuid).metadata
        return SpeakerMetaForTextBox(
            speakerUuid=speaker_uuid,
            styleId=style.style_id,
            speakerName=metadata.speaker_name,
            styleName=style.style_name,
        )

    def style_id_to_speaker_meta(
        self, style_id: int, speaker_uuid: Optional[str] = None
    ) -> SpeakerMetaForTextBox:
        """v2のスタイル単独検索規則でテキストボックス用メタデータを返します。
        Coreのモデル読込は話者UUIDなしの曖昧なスタイルIDを拒否しますが、表示専用のこのメソッドでは最後に一致した値を使います。
        互換動作をモデル読込へ広げないため、例外的な規則をこのメソッド内に限定します。
        """

        if speaker_uuid is not None:
            return self.speaker_meta_for_style(style_id, speaker_uuid)

        style_id = _style_id(style_id)
        selected: Optional[Tuple[str, SpeakerMetaStyle]] = None
        for candidate_uuid in self.speaker_uuids:
            for style in self._records[candidate_uuid].metadata.styles:
                if style.style_id == style_id:
                    selected = (candidate_uuid, style)
        if selected is None:
            raise StyleNotFoundError("<unspecified>", style_id)

        selected_uuid, selected_style = selected
        metadata = self._record(selected_uuid).metadata
        return SpeakerMetaForTextBox(
            speakerUuid=selected_uuid,
            styleId=selected_style.style_id,
            speakerName=metadata.speaker_name,
            styleName=selected_style.style_name,
        )

    def find_style(
        self, style_id: int, speaker_uuid: Optional[str] = None
    ) -> Tuple[str, SpeakerMetaStyle]:
        """複数話者に一致するスタイルIDを拒否してスタイルを検索します。"""

        style_id = _style_id(style_id)
        if speaker_uuid is not None:
            return speaker_uuid, self._style(speaker_uuid, style_id)

        matches = [
            (uuid, style)
            for uuid in self.speaker_uuids
            for style in self._records[uuid].metadata.styles
            if style.style_id == style_id
        ]
        if not matches:
            raise StyleNotFoundError("<unspecified>", style_id)
        if len(matches) > 1:
            raise AmbiguousStyleError(style_id, [uuid for uuid, _ in matches])
        return matches[0]

    def _speaker_meta(self, record: _SpeakerRecord) -> SpeakerMeta:
        styles = []
        for style in record.metadata.styles:
            icon_path = record.folder / "icons" / f"{style.style_id}.png"
            style_portrait_path = record.folder / "portraits" / f"{style.style_id}.png"
            styles.append(
                SpeakerMetaStyle(
                    styleName=style.style_name,
                    styleId=style.style_id,
                    base64Icon=_base64_file(icon_path, "style icon"),
                    base64Portrait=(
                        _base64_file(style_portrait_path, "style portrait")
                        if style_portrait_path.is_file()
                        else None
                    ),
                )
            )
        return SpeakerMeta(
            speakerName=record.metadata.speaker_name,
            speakerUuid=record.metadata.speaker_uuid,
            styles=styles,
            version=record.metadata.version,
            base64Portrait=_base64_file(record.folder / "portrait.png", "speaker portrait"),
        )

    def speaker_meta(self, speaker_uuid: str) -> SpeakerMeta:
        # Base64化した肖像・アイコンは話者数に比例してメモリを消費するため、プロセス全体のキャッシュには保持しません。
        # 合成経路ではないこのAPIで毎回アセットを読むことで、再起動せずに差し替えも反映できます。
        return self._speaker_meta(self._record(speaker_uuid))

    def list_speakers(self) -> List[SpeakerMeta]:
        """Base64化した肖像とスタイルアイコンを含むv2話者メタデータを返します。"""

        return [self.speaker_meta(uuid) for uuid in self.speaker_uuids]

    def speakers(self) -> List[SpeakerMeta]:
        """``list_speakers``の簡易エイリアスです。"""

        return self.list_speakers()

    def _speaker_meta_path_variant(
        self, record: _SpeakerRecord
    ) -> SpeakerMetaPathVariant:
        styles = []
        for style in record.metadata.styles:
            icon_path = _required_file(
                record.folder / "icons" / f"{style.style_id}.png", "style icon"
            )
            style_portrait = record.folder / "portraits" / f"{style.style_id}.png"
            styles.append(
                StylePathVariant(
                    styleName=style.style_name,
                    styleId=style.style_id,
                    pathIcon=str(icon_path),
                    pathPortrait=(
                        str(style_portrait.resolve())
                        if style_portrait.is_file()
                        else None
                    ),
                )
            )
        portrait = _required_file(record.folder / "portrait.png", "speaker portrait")
        return SpeakerMetaPathVariant(
            speakerName=record.metadata.speaker_name,
            speakerUuid=record.metadata.speaker_uuid,
            styles=styles,
            version=record.metadata.version,
            pathPortrait=str(portrait),
        )

    def speaker_meta_path_variant(self, speaker_uuid: str) -> SpeakerMetaPathVariant:
        return self._speaker_meta_path_variant(self._record(speaker_uuid))

    def list_speakers_path_variant(self) -> List[SpeakerMetaPathVariant]:
        """ローカル画像アセットのパスを含むv2話者メタデータを返します。"""

        return [
            self._speaker_meta_path_variant(self._records[uuid])
            for uuid in self.speaker_uuids
        ]

    def speakers_path_variant(self) -> List[SpeakerMetaPathVariant]:
        """``list_speakers_path_variant``の簡易エイリアスです。"""

        return self.list_speakers_path_variant()

    def voice_sample_paths(self, speaker_uuid: str, style_id: int) -> List[Path]:
        """スタイルの音声サンプルを数値サフィックス順で返します。"""

        style_id = _style_id(style_id)
        self._style(speaker_uuid, style_id)
        sample_dir = self.speaker_path(speaker_uuid) / "voice_samples"
        expression = re.compile(rf"{re.escape(str(style_id))}_(\d+)\.wav$")
        matches = []
        if sample_dir.is_dir():
            for path in sample_dir.iterdir():
                match = expression.fullmatch(path.name)
                if match and path.is_file():
                    matches.append((int(match.group(1)), path.name, path.resolve()))
        matches.sort(key=lambda item: (item[0], item[1]))
        if not matches:
            raise MetadataAssetNotFoundError(
                sample_dir, f"voice samples for styleId {style_id}"
            )
        return [path for _, _, path in matches]

    def sample_voice_path(
        self, speaker_uuid: str, style_id: int, index: int = 0
    ) -> Path:
        """数値ファイル順の0始まりインデックスで音声サンプルを返します。"""

        index = _sample_index(index)
        paths = self.voice_sample_paths(speaker_uuid, style_id)
        if index >= len(paths):
            raise MetadataAssetNotFoundError(
                paths[0].parent,
                f"voice sample index {index} for styleId {_style_id(style_id)}",
            )
        return paths[index]

    def read_sample_voice(
        self, speaker_uuid: str, style_id: int, index: int = 0
    ) -> bytes:
        path = self.sample_voice_path(speaker_uuid, style_id, index)
        return _read_bytes(path, "voice sample")

    def sample_voice_base64(
        self, speaker_uuid: str, style_id: int, index: int = 0
    ) -> str:
        path = self.sample_voice_path(speaker_uuid, style_id, index)
        return _base64_file(path, "voice sample")

    def read_policy(self, speaker_uuid: str) -> str:
        """必須の``policy.md``をUTF-8テキストとして読み込みます。"""

        return _read_text(
            self.speaker_path(speaker_uuid) / "policy.md", "speaker policy"
        )

    def license_paths(self, speaker_uuid: str) -> List[Path]:
        """すべてのLICENSEファイルを安定したファイル名順で返します。"""

        folder = self.speaker_path(speaker_uuid)
        paths = [
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.name.lower().startswith("license")
        ]
        return sorted(paths, key=lambda path: (path.name.lower(), path.name))

    def read_license(self, speaker_uuid: str) -> Optional[str]:
        """標準ライセンスを読み込み、なければ最初のLICENSEファイルへフォールバックします。
        パッケージに作者固有の``LICENSE_*.txt``があっても``LICENSE.txt``を優先します。
        すべてのライセンスを個別に表示する呼び出し元は``license_paths``を使えます。
        """

        paths = self.license_paths(speaker_uuid)
        if not paths:
            return None
        preferred = next(
            (path for path in paths if path.name.lower() == "license.txt"), paths[0]
        )
        return _read_text(preferred, "speaker license")

    def speaker_policy(self, speaker_uuid: str) -> SpeakerPolicy:
        """1話者分の規約と任意のライセンス本文を返します。"""

        policy_path = self.speaker_path(speaker_uuid) / "policy.md"
        policy = (
            _read_text(policy_path, "speaker policy")
            if policy_path.is_file()
            else None
        )
        return SpeakerPolicy(policy=policy, license=self.read_license(speaker_uuid))

    def read_policy_license(self, speaker_uuid: str) -> SpeakerPolicy:
        """``speaker_policy``の簡易エイリアスです。"""

        return self.speaker_policy(speaker_uuid)

    def speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        """旧形式の追加話者情報オブジェクトを構築します。"""

        record = self._record(speaker_uuid)
        style_infos = []
        for style in record.metadata.styles:
            style_portrait_path = record.folder / "portraits" / f"{style.style_id}.png"
            style_infos.append(
                StyleInfo(
                    id=style.style_id,
                    icon=_base64_file(
                        record.folder / "icons" / f"{style.style_id}.png", "style icon"
                    ),
                    portrait=(
                        _base64_file(style_portrait_path, "style portrait")
                        if style_portrait_path.is_file()
                        else None
                    ),
                    voice_samples=[
                        _base64_file(path, "voice sample")
                        for path in self.voice_sample_paths(speaker_uuid, style.style_id)
                    ],
                )
            )
        return SpeakerInfo(
            policy=self.read_policy(speaker_uuid),
            portrait=_base64_file(record.folder / "portrait.png", "speaker portrait"),
            style_infos=style_infos,
        )


MetadataStore = SpeakerMetadataStore


__all__ = [
    "AmbiguousStyleError",
    "MetadataAssetNotFoundError",
    "MetadataError",
    "MetadataStore",
    "SpeakerMetadataStore",
    "SpeakerNotFoundError",
    "StyleNotFoundError",
]
