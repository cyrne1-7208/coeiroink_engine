import json
from typing import Any, Dict, Iterable, List, Optional

import pytest
import requests

from voicevox_engine.coeiroink_v2.catalog import (
    CatalogClient,
    CatalogHTTPError,
    CatalogNetworkError,
    CatalogResponseError,
    CatalogResponseTooLarge,
    CatalogSchemaError,
)


_BASE_URL = "https://catalog.example.test/api/v1"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        status_code: int = 200,
        content_type: str = "application/json; charset=utf-8",
        content_length: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        else:
            self.headers["Content-Length"] = str(len(body))
        self._body = body
        self.closed = False
        self.iterated = False

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        self.iterated = True
        for offset in range(0, len(self._body), max(1, chunk_size)):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self, responses: Any = None, error: Optional[Exception] = None
    ) -> None:
        self.responses = responses
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        if callable(self.responses):
            return self.responses(url)
        if isinstance(self.responses, dict):
            return self.responses[url]
        return self.responses


def _download_info_payload() -> List[Dict[str, Any]]:
    return [
        {
            "download_path": "https://downloads.example.test/model.zip",
            "volume": "1.04 GB",
            "speaker": {
                "name": "テスト話者",
                "speaker_uuid": "00000000-0000-0000-0000-000000000001",
                "styles": [{"name": "標準", "id": 140}],
                "version": "1.0.0",
            },
            "speaker_info": {
                "policy": "",
                "portrait": "",
                "style_infos": [{"id": 140, "icon": "", "voice_samples": []}],
            },
        }
    ]


def _downloadable_speakers_payload() -> List[Dict[str, Any]]:
    return [
        {
            "speakerName": "テスト話者",
            "speakerUuid": "00000000-0000-0000-0000-000000000001",
            "subSpeakerUuids": [],
            "styles": [
                {
                    "styleName": "標準",
                    "styleId": 140,
                    "version": "1.0.0",
                    "iconBase64": "",
                    "voiceSampleBase64s": [],
                    "downloadUrl": "https://downloads.example.test/style.zip",
                }
            ],
            "version": "1.0.0",
            "portraitBase64": "",
            "metaDownloadUrl": "https://downloads.example.test/meta.zip",
            "prefix": "test",
        }
    ]


def _update_info_payload() -> List[Dict[str, Any]]:
    return [
        {
            "version": "v.2.13.0",
            "date": "2026-03-23",
            "contents": ["サーバーAPIの更新"],
        }
    ]


@pytest.mark.parametrize(
    "method_name, endpoint, payload, expected_field",
    [
        ("get_download_info", "download-info", _download_info_payload(), "speaker"),
        (
            "get_downloadable_speakers",
            "downloadable-speakers",
            _downloadable_speakers_payload(),
            "speaker_uuid",
        ),
        ("get_update_info", "update-info", _update_info_payload(), "version"),
    ],
)
def test_catalog_methods_use_public_endpoints_and_validate_models(
    method_name: str,
    endpoint: str,
    payload: List[Dict[str, Any]],
    expected_field: str,
) -> None:
    response = FakeResponse(json.dumps(payload).encode("utf-8"))
    session = FakeSession(response)
    client = CatalogClient(
        base_url=_BASE_URL, session=session, timeout=(1.0, 2.0), max_response_bytes=4096
    )

    result = getattr(client, method_name)()

    assert len(result) == 1
    assert hasattr(result[0], expected_field)
    assert session.calls == [
        {
            "url": f"{_BASE_URL}/{endpoint}",
            "timeout": (1.0, 2.0),
            "stream": True,
        }
    ]
    assert response.closed


def test_short_method_aliases_share_the_same_validated_client() -> None:
    payload = _update_info_payload()
    session = FakeSession(FakeResponse(json.dumps(payload).encode("utf-8")))
    client = CatalogClient(session=session, max_response_bytes=4096)

    result = client.update_info()

    assert result[0].version == "v.2.13.0"


def test_http_errors_are_bounded_and_include_only_a_small_preview() -> None:
    response = FakeResponse(
        b"service unavailable\n" + b"x" * 10000,
        status_code=503,
        content_type="text/plain",
        content_length="10020",
    )
    session = FakeSession(response)
    client = CatalogClient(session=session, max_response_bytes=8)

    with pytest.raises(CatalogHTTPError) as raised:
        client.get_update_info()

    assert raised.value.status_code == 503
    assert "service unavailable" in str(raised.value)
    assert len(raised.value.preview.encode("utf-8")) <= 512
    assert response.closed


def test_declared_oversize_is_rejected_before_streaming() -> None:
    response = FakeResponse(b"[]", content_length="129")
    session = FakeSession(response)
    client = CatalogClient(session=session, max_response_bytes=128)

    with pytest.raises(CatalogResponseTooLarge):
        client.get_update_info()

    assert not response.iterated
    assert response.closed


def test_undeclared_or_inaccurate_oversize_is_rejected_while_streaming() -> None:
    response = FakeResponse(b"123456789", content_length="1")
    session = FakeSession(response)
    client = CatalogClient(session=session, max_response_bytes=8)

    with pytest.raises(CatalogResponseTooLarge):
        client.get_update_info()

    assert response.closed


def test_non_json_and_malformed_json_are_rejected() -> None:
    html_response = FakeResponse(b"<html></html>", content_type="text/html")
    with pytest.raises(CatalogResponseError):
        CatalogClient(session=FakeSession(html_response)).get_update_info()

    invalid_response = FakeResponse(b"not-json")
    with pytest.raises(CatalogResponseError):
        CatalogClient(session=FakeSession(invalid_response)).get_update_info()


def test_schema_validation_rejects_wrong_catalog_items() -> None:
    response = FakeResponse(json.dumps([{"version": "missing fields"}]).encode("utf-8"))

    with pytest.raises(CatalogSchemaError):
        CatalogClient(
            session=FakeSession(response), max_response_bytes=4096
        ).get_downloadable_speakers()


def test_network_failures_are_wrapped() -> None:
    session = FakeSession(error=requests.Timeout("timed out"))

    with pytest.raises(CatalogNetworkError):
        CatalogClient(session=session).get_update_info()


def test_constructor_rejects_an_unbounded_configuration() -> None:
    with pytest.raises(ValueError):
        CatalogClient(max_response_bytes=0)
