import pytest

from immich_pbn.config import (
    Config,
    ConfigError,
    ServerProfile,
    apply_environment,
    load_config,
    normalise_base_url,
    profile_with_overrides,
)


@pytest.mark.parametrize(
    "given, expected",
    [
        ("http://immich.local:2283/", "http://immich.local:2283"),
        ("http://immich.local:2283/api", "http://immich.local:2283"),
        ("immich.local:2283", "http://immich.local:2283"),
        ("https://photos.example.com/api/", "https://photos.example.com"),
    ],
)
def test_base_url_is_forgiving(given, expected):
    assert normalise_base_url(given) == expected


def test_empty_base_url_rejected():
    with pytest.raises(ConfigError):
        normalise_base_url("   ")


def test_scope_is_validated():
    with pytest.raises(ConfigError):
        ServerProfile(base_url="http://x", scope="somewhere-else")


def test_lan_profiles_bypass_env_proxies_and_remote_ones_do_not():
    # A corporate HTTPS_PROXY silently eats requests to `immich.local`, because
    # a hostname never matches the CIDR entries people put in NO_PROXY.
    assert ServerProfile(base_url="http://immich.local", scope="lan").trust_env is False
    assert ServerProfile(base_url="https://p.example", scope="remote").trust_env is True
    assert ServerProfile(base_url="http://x", scope="lan", use_proxy=True).trust_env is True


def test_file_then_env_then_cli_precedence(tmp_path, monkeypatch):
    config_file = tmp_path / "c.toml"
    config_file.write_text(
        """
        default_server = "home"
        [servers.home]
        base_url = "http://from-file:2283"
        api_key = "file-key"
        timeout = 11.0
        [servers.away]
        base_url = "https://away.example"
        scope = "remote"
        """
    )
    monkeypatch.delenv("IMMICH_PBN_SERVER", raising=False)
    monkeypatch.setenv("IMMICH_URL", "http://from-env:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "env-key")

    config = load_config(config_file)
    profile = config.profile()
    assert profile.base_url == "http://from-env:2283"  # env beats file
    assert profile.api_key == "env-key"
    assert profile.timeout == 11.0  # untouched by env

    final = profile_with_overrides(profile, base_url="http://from-cli:2283")
    assert final.base_url == "http://from-cli:2283"  # cli beats env


def test_env_only_edits_the_active_profile(tmp_path, monkeypatch):
    config_file = tmp_path / "c.toml"
    config_file.write_text(
        """
        default_server = "home"
        [servers.home]
        base_url = "http://home:2283"
        [servers.away]
        base_url = "https://away.example"
        scope = "remote"
        """
    )
    monkeypatch.setenv("IMMICH_API_KEY", "shared-key")
    config = load_config(config_file)
    assert config.profile("home").api_key == "shared-key"
    assert config.profile("away").api_key is None


def test_api_key_env_indirection(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET", "indirect-key")
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    config_file = tmp_path / "c.toml"
    config_file.write_text(
        '[servers.home]\nbase_url = "http://h:2283"\napi_key_env = "MY_SECRET"\n'
    )
    assert load_config(config_file).profile("home").api_key == "indirect-key"


def test_unknown_profile_names_the_alternatives():
    config = Config(servers={"home": ServerProfile(base_url="http://h")}, default_server="home")
    with pytest.raises(ConfigError, match="home"):
        config.profile("nope")


def test_unknown_key_in_profile_is_an_error(tmp_path):
    config_file = tmp_path / "c.toml"
    config_file.write_text('[servers.home]\nbase_url = "http://h"\ntypo_here = 1\n')
    with pytest.raises(ConfigError, match="typo_here"):
        load_config(config_file)


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.toml")


def test_verify_tls_accepts_a_ca_bundle_path(monkeypatch):
    monkeypatch.setenv("IMMICH_VERIFY_TLS", "/etc/ssl/my-ca.pem")
    monkeypatch.setenv("IMMICH_URL", "https://p.example")
    config = apply_environment(Config(servers={}, default_server="default"))
    assert config.profile().verify_tls == "/etc/ssl/my-ca.pem"
