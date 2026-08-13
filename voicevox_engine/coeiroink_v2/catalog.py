"""公開COEIROINKカタログAPI向けの小さなクライアントです。
v2のEngineエンドポイントは公開COEIROINKサイトのカタログ値を使います。
サイトのパスはハイフン、ローカルEngineのパスはアンダースコアを使います。
``/v1/download_info`` -> ``/api/v1/download-info``
``/v1/downloadable_speakers`` -> ``/api/v1/downloadable-speakers``
``/v1/update_info`` -> ``/api/v1/update-info``
このクライアントはJSONメタデータだけを読み取り、ダウンロードURLの追跡やディスクへの保存は行いません。
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import requests
from pydantic import BaseModel, ValidationError as PydanticValidationError

from .models import DownloadableModel, DownloadableSpeaker, UpdateInfo


Timeout = Union[float, Tuple[float, float]]
ModelT = TypeVar("ModelT", bound=BaseModel)


class CatalogClientError(RuntimeError):
    """カタログの通信・サイズ・スキーマ失敗の基底クラスです。"""


class CatalogNetworkError(CatalogClientError):
    """カタログ要求を完了できない場合に発生します。"""


class CatalogHTTPError(CatalogClientError):
    """カタログが成功以外のHTTPステータスを返した場合に発生します。"""

    def __init__(self, status_code: int, url: str, preview: str = "") -> None:
        self.status_code = status_code
        self.url = url
        self.preview = preview
        message = f"catalog request failed with HTTP {status_code}: {url}"
        if preview:
            message += f" ({preview})"
        super().__init__(message)


class CatalogResponseError(CatalogClientError):
    """成功応答がサイズ制限内のJSONデータでない場合に発生します。"""


class CatalogResponseTooLarge(CatalogResponseError):
    """設定した上限を超える応答を保持する前に発生します。"""

    def __init__(self, url: str, size: int, limit: int) -> None:
        self.url = url
        self.size = size
        self.limit = limit
        super().__init__(
            f"catalog response exceeds {limit} bytes: {url} ({size} bytes)"
        )


class CatalogSchemaError(CatalogResponseError):
    """カタログJSONがv2応答モデルに一致しない場合に発生します。"""


class OfficialSiteCatalogClient:
    """coeiroink.comから現在のカタログデータを取得して検証します。
    ``session``を差し替えられるため、呼び出し元は設定済みセッションを渡せ、テストでは通信しない簡易スタブを使えます。
    ``max_response_bytes``はContent-Lengthの有無にかかわらず応答サイズを制限します。
    """

    DEFAULT_BASE_URL = "https://coeiroink.com/api/v1"
    DEFAULT_TIMEOUT: Timeout = (5.0, 30.0)
    DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
    ERROR_PREVIEW_BYTES = 512
    CHUNK_SIZE = 64 * 1024

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout: Timeout = DEFAULT_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(max_response_bytes, int) or isinstance(
            max_response_bytes, bool
        ) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    def get_download_info(self) -> List[DownloadableModel]:
        """サイトの``/api/v1/download-info``項目を返します。"""

        return self._get_model_list("download-info", DownloadableModel)

    def get_downloadable_speakers(self) -> List[DownloadableSpeaker]:
        """サイトの``/api/v1/downloadable-speakers``項目を返します。"""

        return self._get_model_list("downloadable-speakers", DownloadableSpeaker)

    def get_update_info(self) -> List[UpdateInfo]:
        """サイトの``/api/v1/update-info``項目を返します。"""

        return self._get_model_list("update-info", UpdateInfo)

    # 省略名をルートアダプターから使えるようにし、呼び出し箇所ではget_*形式も明示的に残します。
    download_info = get_download_info
    downloadable_speakers = get_downloadable_speakers
    update_info = get_update_info

    def _get_model_list(
        self, endpoint: str, model_type: Type[ModelT]
    ) -> List[ModelT]:
        payload = self._get_json(endpoint)
        if not isinstance(payload, list):
            raise CatalogSchemaError(
                f"catalog endpoint {endpoint!r} returned {type(payload).__name__}; "
                "expected a JSON array"
            )

        parsed: List[ModelT] = []
        for index, item in enumerate(payload):
            try:
                # HTTP境界でPydantic 2のAPIを使って検証します。
                parsed.append(model_type.model_validate(item))
            except (PydanticValidationError, TypeError, ValueError) as exc:
                raise CatalogSchemaError(
                    f"invalid catalog item at index {index} from {endpoint!r}: {exc}"
                ) from exc
        return parsed

    def _get_json(self, endpoint: str) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(
                url, timeout=self.timeout, stream=True
            )
        except requests.RequestException as exc:
            raise CatalogNetworkError(f"catalog request failed: {url}") from exc

        try:
            status_code = getattr(response, "status_code", None)
            if not isinstance(status_code, int):
                raise CatalogResponseError(
                    f"catalog response has no valid HTTP status: {url}"
                )
            if status_code < 200 or status_code >= 300:
                raise CatalogHTTPError(
                    status_code, url, self._read_error_preview(response)
                )

            headers = getattr(response, "headers", {}) or {}
            content_type = str(headers.get("Content-Type", "")).lower()
            if content_type and "json" not in content_type:
                raise CatalogResponseError(
                    f"catalog response is not JSON ({content_type}): {url}"
                )

            body = self._read_bounded(response, url, headers)
            try:
                return json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CatalogResponseError(
                    f"catalog response is not valid UTF-8 JSON: {url}"
                ) from exc
        except CatalogClientError:
            raise
        except requests.RequestException as exc:
            raise CatalogNetworkError(
                f"catalog response could not be read: {url}"
            ) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _read_bounded(
        self, response: Any, url: str, headers: Dict[str, Any]
    ) -> bytes:
        content_length = headers.get(
            "Content-Length", headers.get("content-length")
        )
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError) as exc:
                raise CatalogResponseError(
                    f"catalog response has invalid Content-Length: {url}"
                ) from exc
            if declared_size < 0:
                raise CatalogResponseError(
                    f"catalog response has negative Content-Length: {url}"
                )
            if declared_size > self.max_response_bytes:
                raise CatalogResponseTooLarge(
                    url, declared_size, self.max_response_bytes
                )

        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            raise CatalogResponseError(
                f"catalog response does not support bounded streaming: {url}"
            )

        chunks: List[bytes] = []
        total = 0
        try:
            for chunk in iterator(chunk_size=self.CHUNK_SIZE):
                if not chunk:
                    continue
                if not isinstance(chunk, (bytes, bytearray)):
                    raise CatalogResponseError(
                        f"catalog response yielded non-byte data: {url}"
                    )
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise CatalogResponseTooLarge(
                        url, total, self.max_response_bytes
                    )
                chunks.append(bytes(chunk))
        except CatalogClientError:
            raise
        except requests.RequestException as exc:
            raise CatalogNetworkError(
                f"catalog response could not be read: {url}"
            ) from exc
        return b"".join(chunks)

    def _read_error_preview(self, response: Any) -> str:
        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            return ""
        preview = bytearray()
        try:
            for chunk in iterator(chunk_size=self.ERROR_PREVIEW_BYTES):
                if not chunk:
                    continue
                if isinstance(chunk, (bytes, bytearray)):
                    remaining = self.ERROR_PREVIEW_BYTES - len(preview)
                    preview.extend(bytes(chunk)[:remaining])
                break
        except requests.RequestException:
            return ""
        return bytes(preview).decode("utf-8", errors="replace").replace("\n", " ")


CatalogClient = OfficialSiteCatalogClient


__all__ = [
    "CatalogClient",
    "CatalogClientError",
    "CatalogHTTPError",
    "CatalogNetworkError",
    "CatalogResponseError",
    "CatalogResponseTooLarge",
    "CatalogSchemaError",
    "OfficialSiteCatalogClient",
]
