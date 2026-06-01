"""Safety checks for backend URL ingestion."""

from __future__ import annotations

from ipaddress import ip_address
import socket
from urllib.parse import urlparse


class UrlValidationError(ValueError):
    """Raised when a user-provided URL is not safe to fetch."""


def _is_public_address(address: str) -> bool:
    parsed = ip_address(address)
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def validate_public_http_url(url: str) -> str:
    """Return a validated URL or raise a user-readable validation error."""
    trimmed_url = url.strip()
    parsed = urlparse(trimmed_url)
    if parsed.scheme not in {"http", "https"}:
        raise UrlValidationError("Enter a valid http or https media URL.")

    if not parsed.hostname:
        raise UrlValidationError("Enter a valid public media URL.")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UrlValidationError("Enter a public media URL, not a local address.")

    try:
        address_infos = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise UrlValidationError("The media URL host could not be resolved.") from exc

    addresses = {info[4][0] for info in address_infos}
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise UrlValidationError(
            "Enter a public media URL, not a private or local address."
        )

    return trimmed_url
