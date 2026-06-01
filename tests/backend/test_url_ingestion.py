from io import BytesIO

from PIL import Image
import pytest

from app.backend.media_download import DownloadError, download_image_url
from app.backend.url_security import UrlValidationError, validate_public_http_url


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(220, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_validate_public_http_url_rejects_private_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [
            (None, None, None, None, ("127.0.0.1", port or 443))
        ],
    )

    with pytest.raises(UrlValidationError, match="public media URL"):
        validate_public_http_url("https://example.test/plate.jpg")


def test_validate_public_http_url_allows_public_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [
            (None, None, None, None, ("93.184.216.34", port or 443))
        ],
    )

    assert (
        validate_public_http_url("https://example.com/plate.jpg")
        == "https://example.com/plate.jpg"
    )


def test_validate_public_http_url_rejects_non_http_scheme() -> None:
    with pytest.raises(UrlValidationError, match="http or https"):
        validate_public_http_url("file:///tmp/plate.jpg")


def test_download_image_url_rejects_non_image_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        headers = {"Content-Type": "text/html", "Content-Length": "12"}
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b"<html></html>"

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr(
        "app.backend.media_download.urlopen",
        lambda request, timeout: Response(),
    )

    with pytest.raises(DownloadError, match="image"):
        download_image_url("https://example.com/page")


def test_download_image_url_reads_bounded_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_jpeg_bytes()

    class Response:
        headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(payload))}
        status = 200

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if self.offset >= len(payload):
                return b""
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr(
        "app.backend.media_download.urlopen",
        lambda request, timeout: Response(),
    )

    assert download_image_url("https://example.com/plate.jpg") == payload
