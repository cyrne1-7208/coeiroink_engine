"""Small client for the public COEIROINK catalog API.

The v2 engine endpoints use the catalog values exposed by the public
COEIROINK website.  The website paths use hyphens while the local engine
paths use underscores:

``/v1/download_info`` -> ``/api/v1/download-info``
``/v1/downloadable_speakers`` -> ``/api/v1/downloadable-speakers``
``/v1/update_info`` -> ``/api/v1/update-info``

This client only reads JSON metadata.  It never follows a catalog download
URL itself and never writes a response to disk.
"""

import json
from typing import Any, TypeVar

import requests
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .models import DownloadableModel, DownloadableSpeaker, UpdateInfo

Timeout = float | tuple[float, float]
ModelT = TypeVar("ModelT", bound=BaseModel)


class CatalogClientError(RuntimeError):
    """Base class for catalog transport, size, and schema failures."""


class CatalogNetworkError(CatalogClientError):
    """Raised when the catalog request cannot be completed."""


class CatalogHTTPError(CatalogClientError):
    """Raised when the catalog returns a non-success HTTP status."""

    def __init__(self, status_code: int, url: str, preview: str = "") -> None:
        self.status_code = status_code
        self.url = url
        self.preview = preview
        message = f"catalog request failed with HTTP {status_code}: {url}"
        if preview:
            message += f" ({preview})"
        super().__init__(message)


class CatalogResponseError(CatalogClientError):
    """Raised when a successful response is not bounded JSON data."""


class CatalogResponseTooLarge(CatalogResponseError):
    """Raised before retaining a response larger than the configured limit."""

    def __init__(self, url: str, size: int, limit: int) -> None:
        self.url = url
        self.size = size
        self.limit = limit
        super().__init__(
            f"catalog response exceeds {limit} bytes: {url} ({size} bytes)"
        )


class CatalogSchemaError(CatalogResponseError):
    """Raised when catalog JSON does not match a v2 response model."""


class OfficialSiteCatalogClient:
    """Fetch and validate current catalog data from coeiroink.com.

    ``session`` is injectable so callers can provide a configured requests
    session and tests can use a small fake without making network calls.
    ``max_response_bytes`` bounds both known and unknown response sizes.  The
    current website catalog contains embedded image data, so the default is
    deliberately large enough for the current JSON while remaining finite.
    """

    DEFAULT_BASE_URL = "https://coeiroink.com/api/v1"
    DEFAULT_TIMEOUT: Timeout = (5.0, 30.0)
    DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
    ERROR_PREVIEW_BYTES = 512
    CHUNK_SIZE = 64 * 1024

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: Timeout = DEFAULT_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    def get_download_info(self) -> list[DownloadableModel]:
        """Return website ``/api/v1/download-info`` entries."""

        return self._get_model_list("download-info", DownloadableModel)

    def get_downloadable_speakers(self) -> list[DownloadableSpeaker]:
        """Return website ``/api/v1/downloadable-speakers`` entries."""

        return self._get_model_list("downloadable-speakers", DownloadableSpeaker)

    def get_update_info(self) -> list[UpdateInfo]:
        """Return website ``/api/v1/update-info`` entries."""

        return self._get_model_list("update-info", UpdateInfo)

    def _get_model_list(self, endpoint: str, model_type: type[ModelT]) -> list[ModelT]:
        payload = self._get_json(endpoint)
        if not isinstance(payload, list):
            raise CatalogSchemaError(
                f"catalog endpoint {endpoint!r} returned {type(payload).__name__}; "
                "expected a JSON array"
            )

        parsed: list[ModelT] = []
        for index, item in enumerate(payload):
            try:
                # 外部JSONとの境界でPydantic 2による型・制約検証を完了させる。
                parsed.append(model_type.model_validate(item))
            except (PydanticValidationError, TypeError, ValueError) as exc:
                raise CatalogSchemaError(
                    f"invalid catalog item at index {index} from {endpoint!r}: {exc}"
                ) from exc
        return parsed

    def _get_json(self, endpoint: str) -> Any:
        """HTTP状態・Content-Type・サイズ・UTF-8 JSONを順に検証してカタログ本文を返す。"""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
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

    def _read_bounded(self, response: Any, url: str, headers: dict[str, Any]) -> bytes:
        """Content-Lengthの有無にかかわらず、ストリームを設定上限以内で読み取る。"""

        content_length = headers.get("Content-Length", headers.get("content-length"))
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

        chunks: list[bytes] = []
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
                    raise CatalogResponseTooLarge(url, total, self.max_response_bytes)
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
