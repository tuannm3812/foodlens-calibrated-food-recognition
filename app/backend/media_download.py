"""Bounded media downloads for URL-based prediction."""

from __future__ import annotations

from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .url_security import UrlValidationError, validate_public_http_url


MAX_IMAGE_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 12
MAX_REDIRECTS = 3
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class DownloadError(ValueError):
    """Raised when remote media cannot be downloaded safely."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _open_url(request: Request, timeout: int) -> object:
    opener = build_opener(_NoRedirectHandler)
    return opener.open(request, timeout=timeout)


def _read_bounded(response: object, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise DownloadError("The image URL is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def _content_type(response: object) -> str:
    raw_content_type = response.headers.get("Content-Type", "")
    return raw_content_type.split(";")[0].strip().lower()


def _content_length(response: object) -> Optional[int]:
    raw_content_length = response.headers.get("Content-Length")
    if not raw_content_length:
        return None
    try:
        return int(raw_content_length)
    except ValueError:
        return None


def download_image_url(url: str) -> bytes:
    """Download a public image URL with redirect, type, timeout, and size limits."""
    current_url = validate_public_http_url(url)

    for _redirect_count in range(MAX_REDIRECTS + 1):
        request = Request(
            current_url,
            headers={"User-Agent": "FoodLens/0.1 URL ingestion"},
            method="GET",
        )
        try:
            with _open_url(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                content_type = _content_type(response)
                if content_type and not content_type.startswith("image/"):
                    raise DownloadError("The URL did not return an image.")

                content_length = _content_length(response)
                if content_length and content_length > MAX_IMAGE_BYTES:
                    raise DownloadError("The image URL is too large.")

                return _read_bounded(response, MAX_IMAGE_BYTES)
        except HTTPError as exc:
            if exc.code in REDIRECT_STATUSES and exc.headers.get("Location"):
                current_url = validate_public_http_url(
                    urljoin(current_url, exc.headers["Location"])
                )
                continue
            raise DownloadError(f"The image URL returned HTTP {exc.code}.") from exc
        except UrlValidationError:
            raise
        except (OSError, URLError) as exc:
            raise DownloadError("The image URL could not be downloaded.") from exc

    raise DownloadError("The image URL redirected too many times.")
