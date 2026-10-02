"""Metadata helpers for the public COEIROINK v2 speaker folders.

The helpers in this module deliberately operate on a ``speaker_info``
directory.  A directory name is not a speaker identity: the identity comes
from ``metas.json``.  This matters for MYCOEIROINK packages such as
``rintos_ver1.0`` whose directory name is different from its speaker UUID.

This module contains no HTTP or download code.  It provides the file-system
operations that v2 routes can use for speaker metadata and assets.
"""

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


class MetadataError(Exception):
    """Base class for invalid speaker metadata or missing speaker assets."""


class SpeakerNotFoundError(MetadataError, LookupError):
    """Raised when a speaker UUID is not present in the speaker directory."""

    def __init__(self, speaker_uuid: str) -> None:
        self.speaker_uuid = speaker_uuid
        super().__init__(f"speakerUuid {speaker_uuid!r} was not found")


class StyleNotFoundError(MetadataError, LookupError):
    """Raised when a style is not present for a speaker."""

    def __init__(self, speaker_uuid: str, style_id: int) -> None:
        self.speaker_uuid = speaker_uuid
        self.style_id = style_id
        super().__init__(
            f"styleId {style_id!r} was not found for speakerUuid {speaker_uuid!r}"
        )


class AmbiguousStyleError(MetadataError, LookupError):
    """Raised when a style ID is used without a unique speaker UUID."""

    def __init__(self, style_id: int, speaker_uuids: list[str]) -> None:
        self.style_id = style_id
        self.speaker_uuids = tuple(speaker_uuids)
        joined = ", ".join(repr(uuid) for uuid in speaker_uuids)
        super().__init__(
            f"styleId {style_id!r} is ambiguous; specify speakerUuid "
            f"(matches: {joined})"
        )


class MetadataAssetNotFoundError(MetadataError, FileNotFoundError):
    """Raised when a required portrait, icon, policy, license, or sample is absent."""

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
    """Validate folder metadata with the response model before adding assets.

    ``metas.json`` describes a model and therefore does not contain the
    base64 fields required by the v2 response model.  Empty placeholders are
    used only for validation; actual values are read from the asset files
    when a response is built.
    """

    if not isinstance(raw_metadata, dict):
        raise MetadataError(
            f"invalid speaker metadata: {meta_path}: expected an object"
        )
    wire_metadata = dict(raw_metadata)
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
    except ValidationError as exc:
        raise MetadataError(f"invalid speaker metadata: {meta_path}: {exc}") from exc


class SpeakerMetadataStore:
    """Resolve v2 speaker metadata and assets from a public model directory.

    ``MetasStore`` is used as the authoritative UUID-to-folder index.  The
    v2 metadata is then parsed from each folder's ``metas.json`` so that the
    result can be returned with the v2 wire models and asset encodings.
    """

    def __init__(
        self, speaker_info_dir: Path, metas_store: MetasStore | None = None
    ) -> None:
        root = Path(speaker_info_dir).expanduser().resolve()
        if not root.is_dir():
            raise MetadataError(f"speaker info directory was not found: {root}")

        try:
            self._metas_store = metas_store or MetasStore(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise MetadataError(
                f"could not load speaker metadata from {root}: {exc}"
            ) from exc

        self._speaker_info_dir = root
        self._records: dict[str, _SpeakerRecord] = {}
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
                raise MetadataError(
                    f"invalid speaker metadata: {meta_path}: {exc}"
                ) from exc

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
    def speaker_uuids(self) -> tuple[str, ...]:
        """Return all speaker UUIDs in deterministic order."""

        return tuple(sorted(self._records))

    def _record(self, speaker_uuid: str) -> _SpeakerRecord:
        try:
            return self._records[speaker_uuid]
        except KeyError as exc:
            raise SpeakerNotFoundError(speaker_uuid) from exc

    def speaker_path(self, speaker_uuid: str) -> Path:
        """Return the physical folder for ``speaker_uuid``."""

        return self._record(speaker_uuid).folder

    def lookup_speaker_folder(self, speaker_uuid: str) -> Path:
        """Alias for :meth:`speaker_path` used by route adapters."""

        return self.speaker_path(speaker_uuid)

    def _style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        style_id = _style_id(style_id)
        record = self._record(speaker_uuid)
        for style in record.metadata.styles:
            if style.style_id == style_id:
                return style
        raise StyleNotFoundError(speaker_uuid, style_id)

    def get_style(self, speaker_uuid: str, style_id: int) -> SpeakerMetaStyle:
        """Look up one style using the unambiguous UUID plus style ID key."""

        return self._style(speaker_uuid, style_id)

    def speaker_meta_for_style(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> SpeakerMetaForTextBox:
        """Return the speaker/style pair selected by UUID and style ID."""

        speaker_uuid, style = self.find_style(style_id, speaker_uuid)
        metadata = self._record(speaker_uuid).metadata
        return SpeakerMetaForTextBox(
            speakerUuid=speaker_uuid,
            styleId=style.style_id,
            speakerName=metadata.speaker_name,
            styleName=style.style_name,
        )

    def style_id_to_speaker_meta(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> SpeakerMetaForTextBox:
        """Return text-box metadata using the v2 style-only lookup rule.

        Core model loading rejects an ambiguous style ID unless a speaker UUID
        is supplied.  The official metadata endpoint instead keeps the last
        loaded match, so that compatibility behavior stays local to this
        display-only method.
        """

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

    def find_style(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> tuple[str, SpeakerMetaStyle]:
        """Find a style, rejecting a style ID that matches several speakers."""

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
        # base64画像は話者数に比例してメモリを占有し合成のホットパスでもないためキャッシュせず、差替えも再起動なしで反映する。
        return self._speaker_meta(self._record(speaker_uuid))

    def list_speakers(self) -> list[SpeakerMeta]:
        """Return v2 speaker metadata with base64 portrait and style icons."""

        return [self.speaker_meta(uuid) for uuid in self.speaker_uuids]

    def speakers(self) -> list[SpeakerMeta]:
        """Convenience alias for :meth:`list_speakers`."""

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

    def list_speakers_path_variant(self) -> list[SpeakerMetaPathVariant]:
        """Return v2 speaker metadata with paths to local image assets."""

        return [
            self._speaker_meta_path_variant(self._records[uuid])
            for uuid in self.speaker_uuids
        ]

    def speakers_path_variant(self) -> list[SpeakerMetaPathVariant]:
        """Convenience alias for :meth:`list_speakers_path_variant`."""

        return self.list_speakers_path_variant()

    def voice_sample_paths(self, speaker_uuid: str, style_id: int) -> list[Path]:
        """Return all voice samples for a style, ordered by numeric suffix."""

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
        """Return a voice sample by zero-based position in numeric file order."""

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
        """Read the required ``policy.md`` file as UTF-8 text."""

        return _read_text(
            self.speaker_path(speaker_uuid) / "policy.md", "speaker policy"
        )

    def license_paths(self, speaker_uuid: str) -> list[Path]:
        """Return all LICENSE files in stable filename order."""

        folder = self.speaker_path(speaker_uuid)
        paths = [
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.name.lower().startswith("license")
        ]
        return sorted(paths, key=lambda path: (path.name.lower(), path.name))

    def read_license(self, speaker_uuid: str) -> str | None:
        """Read the canonical license, falling back to the first LICENSE file.

        ``LICENSE.txt`` is preferred when a package also contains an
        author-specific ``LICENSE_*.txt``.  ``license_paths`` remains
        available to callers that need to present every license separately.
        """

        paths = self.license_paths(speaker_uuid)
        if not paths:
            return None
        preferred = next(
            (path for path in paths if path.name.lower() == "license.txt"), paths[0]
        )
        return _read_text(preferred, "speaker license")

    def speaker_policy(self, speaker_uuid: str) -> SpeakerPolicy:
        """Return policy and optional license text for one speaker."""

        policy_path = self.speaker_path(speaker_uuid) / "policy.md"
        policy = (
            _read_text(policy_path, "speaker policy") if policy_path.is_file() else None
        )
        return SpeakerPolicy(policy=policy, license=self.read_license(speaker_uuid))

    def read_policy_license(self, speaker_uuid: str) -> SpeakerPolicy:
        """Convenience alias for :meth:`speaker_policy`."""

        return self.speaker_policy(speaker_uuid)

    def speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        """Build the legacy-style additional speaker information object."""

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
