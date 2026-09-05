"""A small, defensive Immich API client.

Scoped to what a photo picker needs: list albums, list people, find a random
image matching some filters, and download it.

Two pieces of deliberate paranoia, both earned by how much Immich's API has
moved between releases:

* ``/search/random`` is preferred but only exists on newer servers, and its
  response has been both a bare list and ``{"assets": {"items": [...]}}``.
  Both shapes are accepted, and a server that lacks the endpoint falls back to
  paging ``/search/metadata`` and choosing locally.
* The flat filter fields (``isFavorite``, ``personIds``, ``albumIds``) are
  marked deprecated as of the 3.2 API in favour of a nested ``filter`` object.
  Deprecated is not removed, so the flat form stays the default and the nested
  form is available via ``filter_style="nested"`` for whenever it is not.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any, Literal

from ..config import ServerProfile
from ..net.transport import HTTPStatusError, Transport, TransportError
from .models import Album, Asset, Person

log = logging.getLogger(__name__)

FilterStyle = Literal["flat", "nested", "auto"]

#: ``/search/random`` caps out here per the API schema.
MAX_RANDOM_SIZE = 1000
#: How many pages of /search/metadata to walk before giving up on the fallback.
MAX_FALLBACK_PAGES = 20


class ImmichError(RuntimeError):
    pass


class NoMatchingAssets(ImmichError):
    """The filters are valid but nothing on the server matches them."""


class ImmichClient:
    def __init__(
        self,
        profile: ServerProfile,
        transport: Transport | None = None,
        *,
        filter_style: FilterStyle = "auto",
    ):
        self.profile = profile
        self.transport = transport if transport is not None else Transport(profile)
        self.filter_style: FilterStyle = filter_style
        self._supports_random: bool | None = None

    # ---------------------------------------------------------------- server

    def ping(self) -> bool:
        try:
            data = self.transport.get_json("/server/ping")
        except TransportError:
            return False
        return bool(data and data.get("res") == "pong")

    def server_version(self) -> str:
        for path in ("/server/about", "/server/version"):
            try:
                data = self.transport.get_json(path)
            except TransportError:
                continue
            if not isinstance(data, dict):
                continue
            if version := data.get("version"):
                return str(version)
            if all(k in data for k in ("major", "minor", "patch")):
                return f"v{data['major']}.{data['minor']}.{data['patch']}"
        return "unknown"

    @property
    def random_endpoint_used(self) -> str | None:
        """Which search endpoint the last pick went through, or None if unknown."""
        if self._supports_random is None:
            return None
        return "/search/random" if self._supports_random else "/search/metadata (fallback)"

    # ---------------------------------------------------------------- lookups

    def albums(self) -> list[Album]:
        data = self.transport.get_json("/albums")
        if not isinstance(data, list):
            raise ImmichError(f"unexpected /albums response: {type(data).__name__}")
        return [Album.from_api(item) for item in data]

    def people(self, *, include_hidden: bool = False) -> list[Person]:
        found: list[Person] = []
        page = 1
        while True:
            data = self.transport.get_json(
                "/people", params={"page": page, "size": 500, "withHidden": include_hidden}
            )
            if not isinstance(data, dict):
                raise ImmichError(f"unexpected /people response: {type(data).__name__}")
            found.extend(Person.from_api(item) for item in data.get("people") or [])
            if not data.get("hasNextPage"):
                break
            page += 1
            if page > 100:  # safety valve on a pathological server
                break
        return [p for p in found if include_hidden or not p.hidden]

    def resolve_albums(self, names: Sequence[str]) -> list[Album]:
        """Match album names case-insensitively, then by substring."""
        return _resolve_named(self.albums(), names, kind="album")

    def resolve_people(self, names: Sequence[str]) -> list[Person]:
        return _resolve_named(self.people(), names, kind="person")

    # ----------------------------------------------------------------- search

    def random_image(
        self,
        *,
        album_ids: Sequence[str] = (),
        person_ids: Sequence[str] = (),
        favorites_only: bool = False,
        min_megapixels: float = 0.0,
        rng: random.Random | None = None,
        candidate_pool: int = 250,
    ) -> Asset:
        """Pick one random still image matching the filters.

        ``candidate_pool`` is asked for rather than a single asset so that the
        client-side quality filter (``min_megapixels``, stills only) has
        something to choose from instead of failing on an unlucky draw.
        """
        rng = rng or random.Random()
        size = max(1, min(candidate_pool, MAX_RANDOM_SIZE))
        criteria = _Criteria(
            album_ids=tuple(album_ids),
            person_ids=tuple(person_ids),
            favorites_only=favorites_only,
        )

        candidates = self._random_endpoint(criteria, size)
        if candidates is None:
            candidates = self._metadata_fallback(criteria, size)

        usable = [
            a
            for a in candidates
            if a.type == "IMAGE" and a.megapixels >= min_megapixels
        ]
        if not usable:
            if candidates:
                raise NoMatchingAssets(
                    f"{len(candidates)} asset(s) matched the filters but none were "
                    f"still images of at least {min_megapixels:g} MP"
                )
            raise NoMatchingAssets(
                f"no assets matched {criteria.describe()}"
            )
        rng.shuffle(usable)
        return usable[0]

    def _random_endpoint(self, criteria: _Criteria, size: int) -> list[Asset] | None:
        """Returns ``None`` when the server has no ``/search/random``."""
        if self._supports_random is False:
            return None
        payload = criteria.payload(self._effective_style(), size=size, with_people=True)
        try:
            data = self.transport.post_json("/search/random", payload)
        except HTTPStatusError as exc:
            if exc.status in (400, 404, 405):
                log.debug("/search/random unavailable (%s), falling back", exc.status)
                self._supports_random = False
                return None
            raise
        self._supports_random = True
        return [Asset.from_api(item) for item in _asset_items(data)]

    def _metadata_fallback(self, criteria: _Criteria, size: int) -> list[Asset]:
        collected: list[Asset] = []
        for page in range(1, MAX_FALLBACK_PAGES + 1):
            payload = criteria.payload(
                self._effective_style(), size=size, with_people=True, page=page
            )
            data = self.transport.post_json("/search/metadata", payload)
            items = _asset_items(data)
            collected.extend(Asset.from_api(item) for item in items)
            if not _has_next_page(data) or len(collected) >= size * 4:
                break
        return collected

    def _effective_style(self) -> FilterStyle:
        # "auto" means the flat form: it is what every currently released
        # server accepts, deprecation notwithstanding.
        return "nested" if self.filter_style == "nested" else "flat"

    # --------------------------------------------------------------- download

    def download(self, asset: Asset, *, prefer: str = "original") -> bytes:
        """Fetch image bytes.

        ``prefer="preview"`` grabs Immich's own downscaled JPEG, which is
        usually 1440-2160px on the long edge — plenty for a paint-by-number
        page and far kinder to a home server than pulling a 45 MP raw.
        """
        if prefer == "original":
            return self.transport.get_bytes(
                f"/assets/{asset.id}/original", headers={"Accept": "*/*"}
            )
        size = "fullsize" if prefer == "fullsize" else "preview"
        try:
            return self.transport.get_bytes(
                f"/assets/{asset.id}/thumbnail",
                params={"size": size},
                headers={"Accept": "*/*"},
            )
        except HTTPStatusError as exc:
            if exc.status == 404 and size == "fullsize":
                return self.transport.get_bytes(
                    f"/assets/{asset.id}/thumbnail",
                    params={"size": "preview"},
                    headers={"Accept": "*/*"},
                )
            raise

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> ImmichClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Criteria:
    """Filter set, renderable into either request dialect."""

    def __init__(
        self,
        *,
        album_ids: tuple[str, ...] = (),
        person_ids: tuple[str, ...] = (),
        favorites_only: bool = False,
    ):
        self.album_ids = album_ids
        self.person_ids = person_ids
        self.favorites_only = favorites_only

    def payload(
        self,
        style: FilterStyle,
        *,
        size: int,
        with_people: bool = False,
        page: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"size": size, "withExif": True}
        if with_people:
            body["withPeople"] = True
        if page is not None:
            body["page"] = page

        if style == "nested":
            filters: dict[str, Any] = {"type": {"eq": "IMAGE"}}
            if self.album_ids:
                filters["albumIds"] = {"in": list(self.album_ids)}
            if self.person_ids:
                filters["personIds"] = {"in": list(self.person_ids)}
            if self.favorites_only:
                filters["isFavorite"] = {"eq": True}
            filters["visibility"] = {"eq": "timeline"}
            body["filter"] = filters
            return body

        body["type"] = "IMAGE"
        body["visibility"] = "timeline"
        if self.album_ids:
            body["albumIds"] = list(self.album_ids)
        if self.person_ids:
            body["personIds"] = list(self.person_ids)
        if self.favorites_only:
            body["isFavorite"] = True
        return body

    def describe(self) -> str:
        parts = []
        if self.album_ids:
            parts.append(f"{len(self.album_ids)} album(s)")
        if self.person_ids:
            parts.append(f"{len(self.person_ids)} person(s)")
        if self.favorites_only:
            parts.append("favorites only")
        return " + ".join(parts) if parts else "no filters (whole library)"


def _asset_items(data: Any) -> list[dict[str, Any]]:
    """Normalise the several shapes Immich has returned search results in."""
    if data is None:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        assets = data.get("assets")
        if isinstance(assets, dict):
            return list(assets.get("items") or [])
        if isinstance(assets, list):
            return assets
        if isinstance(data.get("items"), list):
            return data["items"]
    return []


def _has_next_page(data: Any) -> bool:
    if isinstance(data, dict):
        assets = data.get("assets")
        if isinstance(assets, dict):
            return assets.get("nextPage") not in (None, "", 0)
        return data.get("nextPage") not in (None, "", 0)
    return False


def _resolve_named(items: Sequence[Any], names: Sequence[str], *, kind: str) -> list[Any]:
    if not names:
        return []
    by_lower = {getattr(i, "name", "").strip().lower(): i for i in items}
    resolved = []
    for raw in names:
        wanted = raw.strip()
        if not wanted:
            continue
        # A UUID straight from `immich-pbn albums` should work without a lookup.
        if len(wanted) == 36 and wanted.count("-") == 4:
            match = next((i for i in items if i.id == wanted), None)
            if match:
                resolved.append(match)
                continue
        exact = by_lower.get(wanted.lower())
        if exact is not None:
            resolved.append(exact)
            continue
        partial = [
            i for i in items if wanted.lower() in getattr(i, "name", "").lower()
        ]
        if len(partial) == 1:
            resolved.append(partial[0])
        elif len(partial) > 1:
            options = ", ".join(sorted(getattr(i, "name", "") for i in partial))
            raise ImmichError(
                f"{kind} {wanted!r} is ambiguous; it matches: {options}"
            )
        else:
            raise ImmichError(f"no {kind} named {wanted!r} on this server")
    return resolved
