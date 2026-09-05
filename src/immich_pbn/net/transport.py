"""HTTP transport for talking to an Immich server.

Thin on purpose. It owns exactly three things the rest of the code should not
have to think about: the API key header, retry/backoff, and the seam where a
:class:`~immich_pbn.net.resolver.HostResolver` gets to redirect a connection
without breaking TLS.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ServerProfile
from .resolver import HostResolver, build_resolver

log = logging.getLogger(__name__)

USER_AGENT = "immich-pbn/0.1 (+https://github.com/CmdrJorgs/immich-paint-by-number-generator)"

#: Retried on the assumption that a home server may be asleep, busy
#: transcoding, or behind a proxy that briefly 502s.
RETRY_STATUS = (429, 500, 502, 503, 504)


class TransportError(RuntimeError):
    """A request failed in a way the caller cannot fix by retrying."""


class ResolvedHostAdapter(HTTPAdapter):
    """Dial a resolver-supplied address while keeping the hostname's identity.

    The URL's host is swapped for the address, but the ``Host`` header and the
    TLS server name stay as written, so virtual hosts still route and
    certificates still validate against the name you configured rather than
    against a bare IP.
    """

    def __init__(self, resolver: HostResolver, **kwargs: Any):
        self._resolver = resolver
        super().__init__(**kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        parsed = urlparse(request.url)
        hostname = parsed.hostname
        if hostname:
            addresses = self._resolver.resolve(hostname)
            if addresses:
                address = addresses[0]
                netloc = f"[{address}]" if ":" in address else address
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                request.url = urlunparse(parsed._replace(netloc=netloc))
                request.headers["Host"] = parsed.netloc
                if parsed.scheme == "https":
                    # Keep SNI and certificate validation pinned to the name.
                    self.poolmanager.connection_pool_kw["server_hostname"] = hostname
                    self.poolmanager.connection_pool_kw["assert_hostname"] = hostname
                log.debug("resolver sent %s to %s", hostname, address)
        return super().send(request, **kwargs)


def build_session(profile: ServerProfile) -> requests.Session:
    """Create a :class:`requests.Session` wired up for one server profile."""
    resolver = build_resolver(profile.resolver.kind, **profile.resolver.options)

    retry = Retry(
        total=profile.retries,
        connect=profile.retries,
        read=profile.retries,
        status=profile.retries,
        backoff_factor=0.5,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset({"GET", "HEAD", "POST"}),
        raise_on_status=False,
    )
    adapter = ResolvedHostAdapter(resolver, max_retries=retry, pool_maxsize=8)

    session = requests.Session()
    # Mounted on the profile's own origin so pool settings (SNI in particular)
    # never leak to another host.
    session.mount(profile.base_url + "/", adapter)
    session.mount("http://", ResolvedHostAdapter(resolver, max_retries=retry))
    session.mount("https://", ResolvedHostAdapter(resolver, max_retries=retry))

    session.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
    if profile.api_key:
        session.headers["x-api-key"] = profile.api_key
    session.headers.update(profile.headers)
    session.verify = profile.verify_tls
    session.trust_env = profile.trust_env
    return session


class Transport:
    """Request helpers bound to one profile. Everything above this is JSON."""

    def __init__(self, profile: ServerProfile, session: requests.Session | None = None):
        self.profile = profile
        self.session = session if session is not None else build_session(profile)

    @property
    def timeout(self) -> tuple[float, float]:
        connect = self.profile.connect_timeout or min(self.profile.timeout, 10.0)
        return (connect, self.profile.timeout)

    def url(self, path: str) -> str:
        return f"{self.profile.api_root}/{path.lstrip('/')}"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,
        stream: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> requests.Response:
        url = self.url(path)
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json,
                stream=stream,
                headers=dict(headers or {}),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TransportError(f"{method} {url} failed: {exc}") from exc
        return response

    def get_json(self, path: str, **kwargs: Any) -> Any:
        return _json_or_raise(self.request("GET", path, **kwargs))

    def post_json(self, path: str, payload: Any, **kwargs: Any) -> Any:
        return _json_or_raise(self.request("POST", path, json=payload, **kwargs))

    def stream_bytes(self, path: str, *, chunk: int = 1 << 16, **kwargs: Any) -> Iterator[bytes]:
        response = self.request("GET", path, stream=True, **kwargs)
        _raise_for_status(response)
        try:
            yield from response.iter_content(chunk_size=chunk)
        finally:
            response.close()

    def get_bytes(self, path: str, **kwargs: Any) -> bytes:
        return b"".join(self.stream_bytes(path, **kwargs))

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HTTPStatusError(TransportError):
    def __init__(self, status: int, url: str, body: str):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")


def _raise_for_status(response: requests.Response) -> None:
    if response.status_code >= 400:
        body = ""
        try:
            body = response.text
        except Exception:  # pragma: no cover - body already consumed/streamed
            pass
        hint = ""
        if response.status_code in (401, 403):
            hint = (
                " (check the API key: Immich issues these under "
                "Account Settings -> API Keys)"
            )
        raise HTTPStatusError(response.status_code, response.url, body + hint)


def _json_or_raise(response: requests.Response) -> Any:
    _raise_for_status(response)
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError as exc:
        raise TransportError(
            f"expected JSON from {response.url} but got "
            f"{response.headers.get('Content-Type', 'unknown content type')}"
        ) from exc
