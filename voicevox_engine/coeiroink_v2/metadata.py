# 公開COEIROINK v2の話者フォルダのメタデータを扱う補助処理
#
# このモジュールの補助関数は意図的に`speaker_info`ディレクトリ単位で動作する。
# ディレクトリ名は話者の識別子ではなく、識別子は`metas.json`から得る。
# これはMYCOEIROINKパッケージのディレクトリ名が話者UUIDと異なる場合に重要になる。
#
# このモジュールにはHTTPやダウンロードのコードを含めない。
# v2のルートが話者のメタデータと素材ファイルに使うファイルシステム操作を提供する。

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from voicevox_engine.metas.metas_store import MetasStore

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


# 話者のメタデータの不正や素材ファイルの欠落を表すエラーの基底クラス
class MetadataError(Exception):
    pass


# 話者ディレクトリに話者UUIDが存在しない場合のエラー
class SpeakerNotFoundError(MetadataError, LookupError):
    def __init__(self, speaker_uuid: str) -> None:
        self.speaker_uuid = speaker_uuid
        super().__init__(f"speakerUuid {speaker_uuid!r} was not found")


# 話者にスタイルが存在しない場合のエラー
class StyleNotFoundError(MetadataError, LookupError):
    def __init__(self, speaker_uuid: str, style_id: int) -> None:
        self.speaker_uuid = speaker_uuid
        self.style_id = style_id
        super().__init__(
            f"styleId {style_id!r} was not found for speakerUuid {speaker_uuid!r}"
        )


# 話者UUIDで一意に定まらないままスタイルIDが使われた場合のエラー
class AmbiguousStyleError(MetadataError, LookupError):
    def __init__(self, style_id: int, speaker_uuids: list[str]) -> None:
        self.style_id = style_id
        self.speaker_uuids = tuple(speaker_uuids)
        joined = ", ".join(repr(uuid) for uuid in speaker_uuids)
        super().__init__(
            f"styleId {style_id!r} is ambiguous; specify speakerUuid "
            f"(matches: {joined})"
        )


# 必要な立ち絵、アイコン、利用規約、ライセンス、サンプル音声がない場合のエラー
class MetadataAssetNotFoundError(MetadataError, FileNotFoundError):
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
    except OSError as error:
        raise MetadataError(f"could not read {description} {path}: {error}") from error


def _read_text(path: Path, description: str) -> str:
    try:
        return _read_bytes(path, description).decode("utf-8")
    except UnicodeDecodeError as error:
        raise MetadataError(f"{description} is not valid UTF-8: {path}") from error


def _base64_file(path: Path, description: str) -> str:
    return base64.b64encode(_read_bytes(path, description)).decode("ascii")


# 素材ファイルの値を加える前に、フォルダのメタデータを応答モデルで検証する。
#
# `metas.json`はモデルを記述するものなので、v2の応答モデルが必要とするbase64のフィールドを含まない。
# 空の仮の値は検証にだけ使い、実際の値は応答を作るときに素材ファイルから読み込む。
def _parse_folder_metadata(raw_metadata: object, meta_path: Path) -> SpeakerMeta:
    if not isinstance(raw_metadata, dict):
        raise MetadataError(
            f"invalid speaker metadata: {meta_path}: expected an object"
        )
    wire_metadata = dict(raw_metadata)
    # 古いモデルの数値・nullもCoreと同じ文字列表記で受け付ける。
    wire_metadata["version"] = str(wire_metadata.get("version", "0.0.1"))
    wire_metadata.setdefault("base64Portrait", "")
    wire_styles = []
    raw_styles = wire_metadata.get("styles")
    if not isinstance(raw_styles, list):
        raise MetadataError(
            f"invalid speaker metadata: {meta_path}: styles must be a list"
        )
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
    except ValidationError as error:
        raise MetadataError(
            f"invalid speaker metadata: {meta_path}: {error}"
        ) from error


# 公開モデルのディレクトリからv2の話者メタデータと素材ファイルを解決するクラス
#
# UUIDからフォルダへの正式な索引として`MetasStore`を使う。
# そのうえで各フォルダの`metas.json`からv2のメタデータを読み取り、v2の通信用モデルと素材ファイルのエンコードで結果を返せるようにする。
class SpeakerMetadataStore:
    def __init__(
        self, speaker_info_dir: Path, metas_store: MetasStore | None = None
    ) -> None:
        root = Path(speaker_info_dir).expanduser().resolve()
        if not root.is_dir():
            raise MetadataError(f"speaker info directory was not found: {root}")

        try:
            self._metas_store = metas_store or MetasStore(root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise MetadataError(
                f"could not load speaker metadata from {root}: {error}"
            ) from error

        self._speaker_info_dir = root
        self._records: dict[str, _SpeakerRecord] = {}
        for speaker_uuid in sorted(self._metas_store.loaded_metas):
            try:
                folder = Path(self._metas_store.speaker_path(speaker_uuid)).resolve()
            except KeyError as error:
                raise MetadataError(
                    f"MetasStore has no folder for speakerUuid {speaker_uuid!r}"
                ) from error

            meta_path = folder / "metas.json"
            try:
                raw_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = _parse_folder_metadata(raw_metadata, meta_path)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                MetadataError,
            ) as error:
                raise MetadataError(
                    f"invalid speaker metadata: {meta_path}: {error}"
                ) from error

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

    # 全話者のUUIDを決まった順序で返す。
    @property
    def speaker_uuids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def _record(self, speaker_uuid: str) -> _SpeakerRecord:
        try:
            return self._records[speaker_uuid]
        except KeyError as error:
            raise SpeakerNotFoundError(speaker_uuid) from error

    # `speaker_uuid`の実際のフォルダを返す。
    def speaker_path(self, speaker_uuid: str) -> Path:
        return self._record(speaker_uuid).folder

    def _style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        style_id = _style_id(style_id)
        record = self._record(speaker_uuid)
        for style in record.metadata.styles:
            if style.style_id == style_id:
                return style
        raise StyleNotFoundError(speaker_uuid, style_id)

    # 一意に定まるUUIDとスタイルIDの組でスタイルを1つ探す。
    def get_style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        return self._style(speaker_uuid, style_id)

    # UUIDとスタイルIDで選んだ話者とスタイルの組を返す。
    def speaker_meta_for_style(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> SpeakerMetaForTextBox:
        speaker_uuid, style = self.find_style(style_id, speaker_uuid)
        metadata = self._record(speaker_uuid).metadata
        return SpeakerMetaForTextBox(
            speakerUuid=speaker_uuid,
            styleId=style.style_id,
            speakerName=metadata.speaker_name,
            styleName=style.style_name,
        )

    # v2のスタイルIDだけで引く規則でテキストボックス用のメタデータを返す。
    #
    # Coreのモデル読み込みは、話者UUIDが指定されない限り曖昧なスタイルIDを拒否する。
    # 一方、公式のメタデータのエンドポイントは最後に読み込んだ一致を採用するので、この互換動作は表示専用のこのメソッドの中だけに留める。
    def style_id_to_speaker_meta(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> SpeakerMetaForTextBox:
        if speaker_uuid is not None:
            return self.speaker_meta_for_style(style_id, speaker_uuid)

        style_id = _style_id(style_id)
        selected: tuple[str, SpeakerMetaStyle] | None = None
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

    # スタイルを探し、複数の話者に一致するスタイルIDは拒否する。
    def find_style(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> tuple[str, SpeakerMetaStyle]:
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
            base64Portrait=_base64_file(
                record.folder / "portrait.png", "speaker portrait"
            ),
        )

    def speaker_meta(self, speaker_uuid: str) -> SpeakerMeta:
        # base64画像は話者数に比例してメモリを占有し、合成のホットパスでもないためキャッシュせず、差替えも再起動なしで反映する。
        return self._speaker_meta(self._record(speaker_uuid))

    # base64の立ち絵とスタイルのアイコンを含むv2の話者メタデータを返す。
    def list_speakers(self) -> list[SpeakerMeta]:
        return [self.speaker_meta(uuid) for uuid in self.speaker_uuids]

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

    # ローカルの画像ファイルへのパスを含むv2の話者メタデータを返す。
    def list_speakers_path_variant(self) -> list[SpeakerMetaPathVariant]:
        return [
            self._speaker_meta_path_variant(self._records[uuid])
            for uuid in self.speaker_uuids
        ]

    # スタイルの全サンプル音声をファイル名末尾の数値の順に並べて返す。
    def voice_sample_paths(self, speaker_uuid: str, style_id: int) -> list[Path]:
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

    # ファイル名の数値の順で0から数えた位置のサンプル音声を返す。
    def sample_voice_path(
        self, speaker_uuid: str, style_id: int, index: int = 0
    ) -> Path:
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

    # 必須の`policy.md`をUTF-8のテキストとして読み込む。
    def read_policy(self, speaker_uuid: str) -> str:
        return _read_text(
            self.speaker_path(speaker_uuid) / "policy.md", "speaker policy"
        )

    # 全LICENSEファイルをファイル名による一定の順序で返す。
    def license_paths(self, speaker_uuid: str) -> list[Path]:
        folder = self.speaker_path(speaker_uuid)
        paths = [
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.name.lower().startswith("license")
        ]
        return sorted(
            paths,
            key=lambda path: (
                path.name.lower() != "license.txt",
                path.name.lower(),
                path.name,
            ),
        )

    # 話者のライセンスをすべて読み込む。ファイルが1つの場合の応答は従来どおりの形を保つ。
    def read_license(self, speaker_uuid: str) -> str | None:
        paths = self.license_paths(speaker_uuid)
        if not paths:
            return None
        if len(paths) == 1:
            return _read_text(paths[0], "speaker license")
        return "\n\n".join(
            f"## {path.name}\n\n{_read_text(path, 'speaker license')}" for path in paths
        )

    # 1話者の利用規約と、あればライセンスのテキストを返す。
    def speaker_policy(self, speaker_uuid: str) -> SpeakerPolicy:
        policy_path = self.speaker_path(speaker_uuid) / "policy.md"
        policy = (
            _read_text(policy_path, "speaker policy") if policy_path.is_file() else None
        )
        return SpeakerPolicy(policy=policy, license=self.read_license(speaker_uuid))

    # 従来形式の追加の話者情報オブジェクトを作る。
    def speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        record = self._record(speaker_uuid)
        style_infos = []
        for style in record.metadata.styles:
            style_infos.append(
                StyleInfo(
                    id=style.style_id,
                    icon=_base64_file(
                        record.folder / "icons" / f"{style.style_id}.png", "style icon"
                    ),
                    voice_samples=[
                        _base64_file(path, "voice sample")
                        for path in self.voice_sample_paths(
                            speaker_uuid, style.style_id
                        )
                    ],
                )
            )
        return SpeakerInfo(
            policy=self.read_policy(speaker_uuid),
            portrait=_base64_file(record.folder / "portrait.png", "speaker portrait"),
            style_infos=style_infos,
        )


__all__ = [
    "AmbiguousStyleError",
    "MetadataAssetNotFoundError",
    "MetadataError",
    "SpeakerMetadataStore",
    "SpeakerNotFoundError",
    "StyleNotFoundError",
]
