# 公開されているCOEIROINKカタログAPIの小さなクライアント
#
# v2のEngineのエンドポイントは公開COEIROINKサイトが提供するカタログの値を使う。
# サイトのパスはハイフン区切り、ローカルのEngineのパスはアンダースコア区切りで、次のように対応する。
#
# `/v1/download_info` -> `/api/v1/download-info`
# `/v1/downloadable_speakers` -> `/api/v1/downloadable-speakers`
# `/v1/update_info` -> `/api/v1/update-info`
#
# このクライアントはJSONのメタデータを読むだけで、カタログのダウンロードURLを自らたどることも、応答をディスクに書き込むこともない。

import json
from typing import Any, TypeVar

import requests
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .models import DownloadableModel, DownloadableSpeaker, UpdateInfo

Timeout = float | tuple[float, float]
ModelT = TypeVar("ModelT", bound=BaseModel)


# カタログの通信・サイズ・スキーマに関するエラーの基底クラス
class CatalogClientError(RuntimeError):
    pass


# カタログへのリクエストを完了できない場合のエラー
class CatalogNetworkError(CatalogClientError):
    pass


# カタログが成功以外のHTTPステータスを返した場合のエラー
class CatalogHTTPError(CatalogClientError):
    def __init__(self, status_code: int, url: str, preview: str = "") -> None:
        self.status_code = status_code
        self.url = url
        self.preview = preview
        message = f"catalog request failed with HTTP {status_code}: {url}"
        if preview:
            message += f" ({preview})"
        super().__init__(message)


# 成功した応答がサイズ上限内のJSONデータでない場合のエラー
class CatalogResponseError(CatalogClientError):
    pass


# 設定した上限より大きい応答を保持する前に送出するエラー
class CatalogResponseTooLarge(CatalogResponseError):
    def __init__(self, url: str, size: int, limit: int) -> None:
        self.url = url
        self.size = size
        self.limit = limit
        super().__init__(
            f"catalog response exceeds {limit} bytes: {url} ({size} bytes)"
        )


# カタログのJSONがv2の応答モデルと一致しない場合のエラー
class CatalogSchemaError(CatalogResponseError):
    pass


# coeiroink.comから現在のカタログデータを取得して検証するクライアント
#
# `session`は外部から渡せるので、呼び出し側は設定済みのrequestsのセッションを使え、テストはネットワークに接続しない小さな偽物を使える。
# `max_response_bytes`はサイズが分かっている応答と分からない応答の両方の上限になる。
# 現在のサイトのカタログは画像データを埋め込んでいるため、既定値は現在のJSONが収まる程度に大きく、かつ有限の値にしている。
class OfficialSiteCatalogClient:
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

    # サイトの`/api/v1/download-info`の項目を返す。
    def get_download_info(self) -> list[DownloadableModel]:
        return self._get_model_list("download-info", DownloadableModel)

    # サイトの`/api/v1/downloadable-speakers`の項目を返す。
    def get_downloadable_speakers(self) -> list[DownloadableSpeaker]:
        return self._get_model_list("downloadable-speakers", DownloadableSpeaker)

    # サイトの`/api/v1/update-info`の項目を返す。
    def get_update_info(self) -> list[UpdateInfo]:
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
            except PydanticValidationError as error:
                raise CatalogSchemaError(
                    f"invalid catalog item at index {index} from {endpoint!r}: {error}"
                ) from error
        return parsed

    # HTTP状態・Content-Type・サイズ・UTF-8 JSONを順に検証してカタログ本文を返す。
    def _get_json(self, endpoint: str) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
        except requests.RequestException as error:
            raise CatalogNetworkError(f"catalog request failed: {url}") from error

        try:
            status_code = response.status_code
            if status_code < 200 or status_code >= 300:
                raise CatalogHTTPError(
                    status_code, url, self._read_error_preview(response)
                )

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "json" not in content_type:
                raise CatalogResponseError(
                    f"catalog response is not JSON ({content_type}): {url}"
                )

            body = self._read_bounded(response, url)
            try:
                return json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CatalogResponseError(
                    f"catalog response is not valid UTF-8 JSON: {url}"
                ) from error
        except requests.RequestException as error:
            raise CatalogNetworkError(
                f"catalog response could not be read: {url}"
            ) from error
        finally:
            response.close()

    # Content-Lengthの有無にかかわらず、ストリームを設定上限以内で読み取る。
    def _read_bounded(self, response: requests.Response, url: str) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise CatalogResponseError(
                    f"catalog response has invalid Content-Length: {url}"
                ) from error
            if declared_size < 0:
                raise CatalogResponseError(
                    f"catalog response has negative Content-Length: {url}"
                )
            if declared_size > self.max_response_bytes:
                raise CatalogResponseTooLarge(
                    url, declared_size, self.max_response_bytes
                )

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_response_bytes:
                raise CatalogResponseTooLarge(url, total, self.max_response_bytes)
            chunks.append(chunk)
        return b"".join(chunks)

    def _read_error_preview(self, response: requests.Response) -> str:
        try:
            for chunk in response.iter_content(chunk_size=self.ERROR_PREVIEW_BYTES):
                if chunk:
                    return (
                        chunk[: self.ERROR_PREVIEW_BYTES]
                        .decode("utf-8", errors="replace")
                        .replace("\n", " ")
                    )
        except requests.RequestException:
            # 補足の本文を読めなくても、受信済みのHTTPステータスを主原因として報告する。
            return ""
        return ""


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
