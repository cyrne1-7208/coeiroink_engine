"""VOICEVOX互換APIで返す公開リソースを不透明なIDへ対応付ける。"""

from base64 import b64encode
from hashlib import sha256
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import FileResponse

from ..model import ResourceFormat


class ResourceNotFoundError(LookupError):
    """未登録または公開対象外のリソースが指定された。"""


class ResourceManager:
    """speaker_info配下で明示的に登録されたファイルだけを公開する。"""

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir.expanduser().resolve()
        self._resources: dict[str, Path] = {}

    def register(self, resource_path: Path) -> str:
        """公開対象ファイルを登録し、内容に対応するIDを返す。"""

        try:
            resolved = resource_path.expanduser().resolve(strict=True)
            resolved.relative_to(self._root_dir)
        except (FileNotFoundError, ValueError) as error:
            raise ResourceNotFoundError(str(resource_path)) from error
        if not resolved.is_file():
            raise ResourceNotFoundError(str(resource_path))
        resource_hash = sha256(resolved.read_bytes()).hexdigest()
        self._resources[resource_hash] = resolved
        return resource_hash

    def path(self, resource_hash: str) -> Path:
        """登録済みIDに対応するファイルを返す。"""

        try:
            return self._resources[resource_hash]
        except KeyError as error:
            raise ResourceNotFoundError(resource_hash) from error


def resource_value(
    manager: ResourceManager,
    request: Request,
    resource_format: ResourceFormat,
    path: Path,
) -> str:
    """指定形式に応じてbase64本体または同一サーバー上のURLを返す。"""

    if resource_format == ResourceFormat.BASE64:
        return b64encode(path.read_bytes()).decode("utf-8")
    resource_hash = manager.register(path)
    return str(request.url_for("voicevox_resource", resource_hash=resource_hash))


def add_resource_route(router: APIRouter, manager: ResourceManager) -> None:
    """登録済みリソースだけを取得できる非公開ルートを追加する。"""

    @router.get(
        "/_resources/{resource_hash}",
        include_in_schema=False,
        name="voicevox_resource",
    )
    def voicevox_resource(resource_hash: str) -> FileResponse:
        try:
            resource_path = manager.path(resource_hash)
        except ResourceNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="リソースが見つかりません"
            ) from error
        return FileResponse(resource_path)


__all__ = [
    "ResourceManager",
    "ResourceNotFoundError",
    "add_resource_route",
    "resource_value",
]
