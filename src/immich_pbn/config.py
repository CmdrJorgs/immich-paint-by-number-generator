"""Configuration: named server profiles layered over env vars and CLI flags.

Precedence, highest first:

    1. explicit CLI flags
    2. environment variables (IMMICH_PBN_SERVER, IMMICH_URL, IMMICH_API_KEY, ...)
    3. the selected profile in the TOML config file
    4. built-in defaults

The profile indirection is what makes a second server cheap to add later: the
LAN box you have today is one profile, a reverse-proxied instance you reach from
outside is another, and nothing but ``--server`` changes at the call site.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_PORT = 2283
CONFIG_ENV = "IMMICH_PBN_CONFIG"

#: Searched in order; the first that exists wins.
CONFIG_SEARCH_PATH = (
    Path("immich-pbn.toml"),
    Path(".immich-pbn.toml"),
    Path.home() / ".config" / "immich-pbn" / "config.toml",
    Path.home() / ".immich-pbn.toml",
)


class ConfigError(ValueError):
    """Raised when a config file or profile is malformed or missing."""


@dataclass(frozen=True)
class ResolverSpec:
    """How to turn the profile's hostname into something connectable.

    ``kind`` names an entry in :mod:`immich_pbn.net.resolver`'s registry.
    Everything else is passed to that resolver as keyword options, so a future
    resolver can take whatever settings it needs without touching this file.
    """

    kind: str = "system"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Any, *, where: str) -> ResolverSpec:
        if data is None:
            return cls()
        if isinstance(data, str):
            return cls(kind=data)
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: resolver must be a string or a table")
        opts = {k: v for k, v in data.items() if k != "kind"}
        return cls(kind=str(data.get("kind", "system")), options=opts)


@dataclass(frozen=True)
class ServerProfile:
    """Everything needed to talk to one Immich instance."""

    name: str = "default"
    base_url: str = f"http://localhost:{DEFAULT_PORT}"
    api_key: str | None = None
    #: "lan" or "remote". Advisory today; it only relaxes/tightens defaults
    #: (a LAN box gets short timeouts, a remote one gets patient ones) but it is
    #: the hook a future transport can read to decide on TLS pinning, proxies,
    #: or a VPN dial-up step.
    scope: str = "lan"
    verify_tls: bool | str = True
    timeout: float = 30.0
    connect_timeout: float | None = None
    retries: int = 3
    #: Honour HTTP(S)_PROXY / NO_PROXY from the environment. Off for LAN
    #: profiles by default: a proxy set for internet traffic will happily
    #: swallow a request to ``immich.local``, because a hostname never matches
    #: the CIDR entries people put in NO_PROXY. ``None`` means "decide by scope".
    use_proxy: bool | None = None
    resolver: ResolverSpec = field(default_factory=ResolverSpec)
    #: Extra headers merged into every request (e.g. a Cloudflare Access token).
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in ("lan", "remote"):
            raise ConfigError(
                f"server '{self.name}': scope must be 'lan' or 'remote', got {self.scope!r}"
            )
        object.__setattr__(self, "base_url", normalise_base_url(self.base_url))

    @property
    def api_root(self) -> str:
        return f"{self.base_url}/api"

    @property
    def trust_env(self) -> bool:
        if self.use_proxy is not None:
            return self.use_proxy
        return self.scope == "remote"


@dataclass(frozen=True)
class Config:
    servers: dict[str, ServerProfile] = field(default_factory=dict)
    default_server: str = "default"
    #: Generation defaults, overridable per invocation. Kept as a plain dict so
    #: the CLI stays the single source of truth for option names.
    defaults: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def profile(self, name: str | None = None) -> ServerProfile:
        wanted = name or self.default_server
        try:
            return self.servers[wanted]
        except KeyError:
            known = ", ".join(sorted(self.servers)) or "<none>"
            raise ConfigError(
                f"unknown server profile {wanted!r}; configured profiles: {known}"
            ) from None


def normalise_base_url(url: str) -> str:
    """Trim trailing slashes and a trailing ``/api`` so callers can be sloppy."""
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        raise ConfigError("base_url must not be empty")
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    if cleaned.endswith("/api"):
        cleaned = cleaned[: -len("/api")]
    return cleaned


def _profile_from_table(name: str, table: dict[str, Any]) -> ServerProfile:
    if not isinstance(table, dict):
        raise ConfigError(f"servers.{name} must be a table")
    unknown = set(table) - {
        "base_url", "url", "api_key", "api_key_env", "api_key_file", "scope",
        "verify_tls", "timeout", "connect_timeout", "retries", "resolver", "headers",
        "use_proxy",
    }
    if unknown:
        raise ConfigError(f"servers.{name}: unknown keys {sorted(unknown)}")

    base_url = table.get("base_url") or table.get("url")
    if not base_url:
        raise ConfigError(f"servers.{name}: base_url is required")

    api_key = table.get("api_key")
    if not api_key and (env_name := table.get("api_key_env")):
        api_key = os.environ.get(str(env_name))
    if not api_key and (key_file := table.get("api_key_file")):
        path = Path(str(key_file)).expanduser()
        if path.exists():
            api_key = path.read_text(encoding="utf-8").strip()

    scope = str(table.get("scope", "lan"))
    return ServerProfile(
        name=name,
        base_url=str(base_url),
        api_key=api_key,
        scope=scope,
        verify_tls=table.get("verify_tls", True),
        timeout=float(table.get("timeout", 30.0 if scope == "lan" else 60.0)),
        connect_timeout=(
            float(table["connect_timeout"]) if table.get("connect_timeout") else None
        ),
        retries=int(table.get("retries", 3)),
        use_proxy=(
            bool(table["use_proxy"]) if table.get("use_proxy") is not None else None
        ),
        resolver=ResolverSpec.from_mapping(
            table.get("resolver"), where=f"servers.{name}"
        ),
        headers={str(k): str(v) for k, v in (table.get("headers") or {}).items()},
    )


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        return path
    if env_path := os.environ.get(CONFIG_ENV):
        path = Path(env_path).expanduser()
        if not path.exists():
            raise ConfigError(f"{CONFIG_ENV} points at a missing file: {path}")
        return path
    for candidate in CONFIG_SEARCH_PATH:
        if candidate.exists():
            return candidate
    return None


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load the TOML config, then let the environment override the active profile."""
    config_path = find_config_file(path)
    raw: dict[str, Any] = {}
    if config_path is not None:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

    servers = {
        name: _profile_from_table(name, table)
        for name, table in (raw.get("servers") or {}).items()
    }
    default_server = str(raw.get("default_server") or (next(iter(servers), "default")))
    defaults = dict(raw.get("defaults") or {})

    config = Config(
        servers=servers,
        default_server=default_server,
        defaults=defaults,
        source=config_path,
    )
    return apply_environment(config)


def apply_environment(config: Config, env: dict[str, str] | None = None) -> Config:
    """Fold ``IMMICH_*`` variables into the active profile.

    An env var never edits a *named* profile in place; it edits whichever
    profile is about to be used. That keeps a checked-in config file honest
    while still letting a shell (or a systemd unit) inject the API key.
    """
    env = os.environ if env is None else env
    servers = dict(config.servers)
    default_server = env.get("IMMICH_PBN_SERVER", config.default_server)

    active = servers.get(default_server)
    url = env.get("IMMICH_URL") or env.get("IMMICH_SERVER_URL")
    api_key = env.get("IMMICH_API_KEY")

    if active is None:
        if not url and not servers:
            # No file, no env: still hand back a usable object so `--url` alone works.
            active = ServerProfile(name=default_server)
        elif not url:
            return config  # named profile is simply absent; let profile() complain
        else:
            active = ServerProfile(name=default_server, base_url=url)

    updates: dict[str, Any] = {}
    if url:
        updates["base_url"] = normalise_base_url(url)
    if api_key:
        updates["api_key"] = api_key
    if verify := env.get("IMMICH_VERIFY_TLS"):
        updates["verify_tls"] = _coerce_verify(verify)
    if timeout := env.get("IMMICH_TIMEOUT"):
        updates["timeout"] = float(timeout)

    servers[default_server] = replace(active, **updates) if updates else active
    return replace(config, servers=servers, default_server=default_server)


def _coerce_verify(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return value  # a path to a CA bundle


def profile_with_overrides(
    profile: ServerProfile,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    verify_tls: bool | str | None = None,
) -> ServerProfile:
    """Apply CLI-level overrides, the last and highest layer."""
    updates: dict[str, Any] = {}
    if base_url:
        updates["base_url"] = normalise_base_url(base_url)
    if api_key:
        updates["api_key"] = api_key
    if timeout is not None:
        updates["timeout"] = timeout
    if verify_tls is not None:
        updates["verify_tls"] = verify_tls
    return replace(profile, **updates) if updates else profile
