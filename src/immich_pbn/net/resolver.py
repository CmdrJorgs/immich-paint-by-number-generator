"""Pluggable hostname resolution.

Today the only Immich server that matters is the one on the LAN, and the OS
resolver already knows how to find it. The point of this module is that the
*next* one might not be so easy: a hostname that only a specific nameserver
knows about, a ``.local`` name that needs mDNS, a split-horizon setup where the
same name means different things inside and outside the house.

Each strategy is a :class:`HostResolver` registered under a short name, which is
what a profile's ``resolver.kind`` selects. Adding a strategy means writing one
class and calling :func:`register_resolver` — no changes to the client, the
transport, or the CLI.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class ResolverError(RuntimeError):
    """Resolution failed, or the requested strategy is not usable here."""


class HostResolver:
    """Maps a hostname to the addresses a connection should actually dial.

    Returning ``None`` means "no opinion" — the transport connects by name and
    lets the operating system resolve it, which is the right answer far more
    often than not.
    """

    kind = "base"

    def resolve(self, hostname: str) -> list[str] | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind


class SystemResolver(HostResolver):
    """Delegate to the operating system. The default, and usually correct."""

    kind = "system"

    def resolve(self, hostname: str) -> list[str] | None:
        return None

    def describe(self) -> str:
        return "system (OS resolver / /etc/hosts / mDNS via nss)"


class StaticResolver(HostResolver):
    """A hosts-file that lives in the config rather than in ``/etc``.

    Useful right now for a server whose DNS name you have not gotten around to
    creating, and useful later as the escape hatch when a remote name resolves
    to the wrong side of a NAT.
    """

    kind = "static"

    def __init__(self, hosts: Mapping[str, str | Iterable[str]] | None = None, **_: Any):
        self._hosts: dict[str, list[str]] = {}
        for name, value in (hosts or {}).items():
            addrs = [value] if isinstance(value, str) else list(value)
            if not addrs:
                raise ResolverError(f"static resolver: no addresses for {name!r}")
            self._hosts[name.lower()] = [str(a) for a in addrs]

    def resolve(self, hostname: str) -> list[str] | None:
        return self._hosts.get(hostname.lower())

    def describe(self) -> str:
        if not self._hosts:
            return "static (empty)"
        entries = ", ".join(f"{k}->{v[0]}" for k, v in sorted(self._hosts.items()))
        return f"static ({entries})"


class SystemWithFallbackResolver(HostResolver):
    """Try the OS first; fall back to a static entry when it comes up empty.

    This is the shape most people actually want for a server that is reachable
    at ``immich.local`` at home and at a pinned address from anywhere else.
    """

    kind = "system+static"

    def __init__(self, hosts: Mapping[str, str | Iterable[str]] | None = None, **_: Any):
        self._static = StaticResolver(hosts)

    def resolve(self, hostname: str) -> list[str] | None:
        try:
            socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            fallback = self._static.resolve(hostname)
            if fallback is None:
                raise ResolverError(
                    f"{hostname!r} did not resolve and no static fallback is configured"
                ) from None
            return fallback
        return None

    def describe(self) -> str:
        return f"system with fallback to {self._static.describe()}"


@dataclass(frozen=True)
class PlannedResolver(HostResolver):
    """A strategy that is designed but not built.

    Registered on purpose: an unimplemented strategy that fails loudly with a
    note about what it would take beats a strategy nobody remembered wanting.
    """

    kind: str = "planned"
    summary: str = ""
    requires: tuple[str, ...] = ()

    def resolve(self, hostname: str) -> list[str] | None:
        extra = f" Requires: {', '.join(self.requires)}." if self.requires else ""
        raise ResolverError(
            f"resolver {self.kind!r} is scaffolding, not an implementation. "
            f"{self.summary}{extra} "
            f"Use resolver.kind = 'static' or 'system+static' until it lands."
        )

    def describe(self) -> str:
        return f"{self.kind} (not implemented: {self.summary})"


ResolverFactory = Callable[..., HostResolver]

_REGISTRY: dict[str, ResolverFactory] = {}


def register_resolver(kind: str, factory: ResolverFactory) -> None:
    _REGISTRY[kind] = factory


def available_resolvers() -> list[str]:
    return sorted(_REGISTRY)


def build_resolver(kind: str = "system", **options: Any) -> HostResolver:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        known = ", ".join(available_resolvers())
        raise ResolverError(
            f"unknown resolver kind {kind!r}; available: {known}"
        ) from None
    return factory(**options)


register_resolver("system", lambda **kw: SystemResolver())
register_resolver("static", StaticResolver)
register_resolver("system+static", SystemWithFallbackResolver)
register_resolver(
    "dns",
    lambda **kw: PlannedResolver(
        kind="dns",
        summary=(
            "Query an explicit nameserver instead of the system one, so a "
            "remote Immich name can be looked up against, say, a Tailscale or "
            "split-horizon resolver."
        ),
        requires=("dnspython", "a 'nameservers' option in the profile"),
    ),
)
register_resolver(
    "mdns",
    lambda **kw: PlannedResolver(
        kind="mdns",
        summary=(
            "Discover the Immich host by multicast DNS-SD rather than a "
            "hardcoded name, for a LAN where the box moves between addresses."
        ),
        requires=("zeroconf",),
    ),
)
