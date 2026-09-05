"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from .config import (
    Config,
    ConfigError,
    ServerProfile,
    load_config,
    profile_with_overrides,
)
from .immich.client import ImmichClient, ImmichError, NoMatchingAssets
from .immich.models import Asset
from .net.resolver import ResolverError, available_resolvers
from .net.transport import TransportError
from .pbn.page import PageLayout, legend_height_for
from .pbn.pipeline import PbnOptions, PipelineError, generate
from .pbn.render import RenderOptions, render_pdf, save_preview_png

log = logging.getLogger("immich_pbn")


SUBCOMMANDS = ("generate", "albums", "people", "doctor")
#: Global flags, accepted before *or* after the subcommand. Each is declared
#: twice under different dests and merged in :func:`main`, because argparse
#: subparser defaults silently clobber a parent's parsed value otherwise.
_GLOBAL_TAKING_A_VALUE = {"--config"}
_GLOBAL_SWITCHES = {"-v", "--verbose"}


def _add_global_flags(parser: argparse.ArgumentParser, suffix: str = "") -> None:
    parser.add_argument(
        "--config", dest=f"config{suffix}", default=None,
        help="path to a TOML config file",
    )
    parser.add_argument(
        "-v", "--verbose", dest=f"verbose{suffix}", action="count", default=0,
        help="repeat for debug logging",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="immich-pbn",
        description=(
            "Turn a random photo from an Immich server into a printable A4 "
            "paint-by-number page."
        ),
    )
    _add_global_flags(parser)
    subparsers = parser.add_subparsers(dest="command")

    def add_server_flags(sub: argparse.ArgumentParser) -> None:
        _add_global_flags(sub, suffix="_sub")
        group = sub.add_argument_group("server")
        group.add_argument("--server", help="named profile from the config file")
        group.add_argument("--url", help="override the profile's base URL")
        group.add_argument("--api-key", help="override the profile's API key")
        group.add_argument("--timeout", type=float, help="request timeout in seconds")
        group.add_argument(
            "--insecure",
            action="store_true",
            help="skip TLS verification (only ever sensible on a LAN with a self-signed cert)",
        )

    gen = subparsers.add_parser(
        "generate", help="make a paint-by-number page (the default command)"
    )
    add_server_flags(gen)

    picking = gen.add_argument_group("choosing the photo")
    picking.add_argument(
        "--album", action="append", default=[], metavar="NAME",
        help="restrict to an album, by name or UUID; repeatable",
    )
    picking.add_argument(
        "--person", action="append", default=[], metavar="NAME",
        help="restrict to photos of a named person; repeatable",
    )
    picking.add_argument("--favorites", "--favourites", action="store_true",
                         dest="favorites", help="restrict to favourites")
    picking.add_argument("--min-megapixels", type=float, default=1.0,
                         help="skip photos smaller than this (default: 1.0)")
    picking.add_argument("--seed", type=int,
                         help="make both the photo choice and the palette reproducible")
    picking.add_argument("--prefer", choices=("preview", "fullsize", "original"),
                         default="preview",
                         help="which rendition to download (default: preview -- "
                              "already big enough, and far kinder to the server)")
    picking.add_argument("--from-file", metavar="PATH",
                         help="skip Immich entirely and use a local image")

    art = gen.add_argument_group("the page itself")
    art.add_argument("-c", "--colors", "--colours", type=int, default=20,
                     dest="colours", metavar="N",
                     help="number of paint colours (default: 20)")
    art.add_argument("-r", "--resolution", type=int, default=1400, metavar="PX",
                     help="working image long edge; higher is more faithful, "
                          "never upscales past the source (default: 1400)")
    art.add_argument("--dpi", type=int, default=300,
                     help="print density for the outline raster (default: 300)")
    art.add_argument("--min-region-mm", type=float, default=3.0, metavar="MM",
                     help="dissolve anything narrower than this on paper (default: 3.0)")
    art.add_argument("--line-width-mm", type=float, default=0.28, metavar="MM")
    art.add_argument("--smoothing", type=int, default=2, metavar="R",
                     help="pre-quantization median radius; 0 disables (default: 2)")
    art.add_argument("--number-size-mm", type=float, default=2.6, metavar="MM")
    art.add_argument("--orientation", choices=("auto", "portrait", "landscape"),
                     default="auto")
    art.add_argument("--margin-mm", type=float, default=12.0)
    art.add_argument("--legend", choices=("inline", "page", "none"), default="inline",
                     help="colour key at the foot of the page, on its own page, or not at all")
    art.add_argument("--no-reference", action="store_true",
                     help="omit the posterised reference page")
    art.add_argument("--no-hex", action="store_true",
                     help="omit sRGB hex codes from the legend")

    out = gen.add_argument_group("output")
    out.add_argument("-o", "--output", default="paint-by-number.pdf")
    out.add_argument("--save-source", metavar="PATH",
                     help="also write the photo the page was made from")
    out.add_argument("--save-preview", metavar="PATH",
                     help="also write a PNG of the outlined, posterised image")

    albums = subparsers.add_parser("albums", help="list albums on the server")
    add_server_flags(albums)
    people = subparsers.add_parser("people", help="list named people on the server")
    add_server_flags(people)
    doctor = subparsers.add_parser(
        "doctor", help="check the config, reach the server, report what it supports"
    )
    add_server_flags(doctor)
    return parser


def _with_default_command(argv: list[str]) -> list[str]:
    """Insert ``generate`` when no subcommand was given.

    It has to go *after* any leading global flags, not at the front: putting it
    first turns ``--config x albums`` into ``generate --config x albums``,
    which argparse rejects with a baffling "unrecognized arguments".
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-h", "--help"):
            return argv
        if token in _GLOBAL_TAKING_A_VALUE:
            index += 2
        elif token.startswith("--config="):
            index += 1
        elif token in _GLOBAL_SWITCHES:
            index += 1
        else:
            break
    if index < len(argv) and argv[index] in SUBCOMMANDS:
        return argv
    return argv[:index] + ["generate"] + argv[index:]


def main(argv: list[str] | None = None) -> int:
    argv = _with_default_command(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = getattr(args, "config_sub", None) or args.config
    verbosity = args.verbose + getattr(args, "verbose_sub", 0)
    logging.basicConfig(
        level=(logging.DEBUG if verbosity > 1 else
               logging.INFO if verbosity else logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(config_path)
        profile = _resolve_profile(config, args)
    except ConfigError as exc:
        print(f"configuration problem: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "generate": cmd_generate,
        "albums": cmd_albums,
        "people": cmd_people,
        "doctor": cmd_doctor,
    }
    try:
        return handlers[args.command or "generate"](args, profile, config)
    except (ImmichError, TransportError, PipelineError, ResolverError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _resolve_profile(config: Config, args: argparse.Namespace) -> ServerProfile:
    profile = config.profile(getattr(args, "server", None))
    return profile_with_overrides(
        profile,
        base_url=getattr(args, "url", None),
        api_key=getattr(args, "api_key", None),
        timeout=getattr(args, "timeout", None),
        verify_tls=False if getattr(args, "insecure", False) else None,
    )


# --------------------------------------------------------------------- commands


def cmd_generate(args, profile: ServerProfile, config: Config) -> int:
    rng = random.Random(args.seed)
    asset: Asset | None = None

    if args.from_file:
        source: bytes | str = args.from_file
        if not Path(args.from_file).exists():
            print(f"error: no such file: {args.from_file}", file=sys.stderr)
            return 1
        caption_source = Path(args.from_file).name
    else:
        with _client(profile) as client:
            albums = client.resolve_albums(args.album)
            people = client.resolve_people(args.person)
            _report_filters(albums, people, args.favorites)
            try:
                asset = client.random_image(
                    album_ids=[a.id for a in albums],
                    person_ids=[p.id for p in people],
                    favorites_only=args.favorites,
                    min_megapixels=args.min_megapixels,
                    rng=rng,
                )
            except NoMatchingAssets as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"chose: {asset.describe()}")
            source = client.download(asset, prefer=args.prefer)
        caption_source = asset.original_file_name or asset.id
        if args.save_source:
            Path(args.save_source).write_bytes(source)
            print(f"wrote {args.save_source}")

    options = PbnOptions(
        colours=args.colours,
        resolution=args.resolution,
        dpi=args.dpi,
        seed=args.seed,
        min_region_mm=args.min_region_mm,
        smoothing=args.smoothing,
        line_width_mm=args.line_width_mm,
        number_size_mm=args.number_size_mm,
        orientation=args.orientation,
    )
    legend_height = (
        legend_height_for(args.colours, 8) if args.legend == "inline" else 0.0
    )
    layout = PageLayout(
        margin_mm=args.margin_mm,
        legend_height_mm=legend_height,
        caption_height_mm=6.0,
    )

    result = generate(source, options, layout)

    caption = f"{caption_source} - {len(result.palette)} colours"
    if asset is not None and asset.local_date_time:
        caption += f" - {asset.local_date_time[:10]}"
    render_options = RenderOptions(
        title=f"Paint by number - {caption_source}",
        caption=caption,
        legend=args.legend,
        reference_page=not args.no_reference,
        number_size_mm=args.number_size_mm,
        show_hex=not args.no_hex,
    )
    with open(args.output, "wb") as handle:
        drawn = render_pdf(result, render_options, handle)

    if args.save_preview:
        save_preview_png(result, args.save_preview)
        print(f"wrote {args.save_preview}")

    _report_result(result, drawn, args.output)
    return 0


def cmd_albums(args, profile: ServerProfile, config: Config) -> int:
    with _client(profile) as client:
        albums = sorted(client.albums(), key=lambda a: a.name.lower())
    if not albums:
        print("no albums on this server")
        return 0
    width = max(len(a.name) for a in albums)
    for album in albums:
        print(f"{album.name:<{width}}  {album.asset_count:>5} assets  {album.id}")
    return 0


def cmd_people(args, profile: ServerProfile, config: Config) -> int:
    with _client(profile) as client:
        people = [p for p in client.people() if p.name]
    if not people:
        print("no named people on this server (Immich only lists people you have named)")
        return 0
    width = max(len(p.name) for p in people)
    for person in sorted(people, key=lambda p: p.name.lower()):
        print(f"{person.name:<{width}}  {person.id}")
    return 0


def cmd_doctor(args, profile: ServerProfile, config: Config) -> int:
    print(f"config file:  {config.source or '(none found -- using defaults and environment)'}")
    print(f"profile:      {profile.name}  [{profile.scope}]")
    print(f"base URL:     {profile.base_url}")
    print(f"API key:      {'set' if profile.api_key else 'MISSING'}")
    print(f"TLS verify:   {profile.verify_tls}")
    print(f"env proxies:  {'honoured' if profile.trust_env else 'bypassed (LAN profile)'}")
    print(f"resolver:     {profile.resolver.kind}  (available: {', '.join(available_resolvers())})")

    if not profile.api_key:
        print("\nNo API key. Set IMMICH_API_KEY, or add api_key to the profile.")
        print("Immich issues keys under Account Settings -> API Keys.")
        return 1

    with _client(profile) as client:
        print(f"\nresolver in use: {_describe_resolver(profile)}")
        if not client.ping():
            print("could not reach the server (/server/ping did not answer 'pong')")
            return 1
        print(f"reachable:    yes, running {client.server_version()}")
        albums = client.albums()
        people = [p for p in client.people() if p.name]
        print(f"albums:       {len(albums)}")
        print(f"named people: {len(people)}")
        try:
            asset = client.random_image(candidate_pool=8)
            endpoint = client.random_endpoint_used or "an unknown endpoint"
            print(f"random pick:  works via {endpoint} -- e.g. {asset.describe()}")
        except NoMatchingAssets:
            print("random pick:  the server answered, but the library looks empty")
    return 0


# ---------------------------------------------------------------------- helpers


def _client(profile: ServerProfile) -> ImmichClient:
    if not profile.api_key:
        raise ImmichError(
            "no API key for this server. Set IMMICH_API_KEY, pass --api-key, or add "
            "api_key/api_key_env to the profile in your config file."
        )
    return ImmichClient(profile)


def _describe_resolver(profile: ServerProfile) -> str:
    from .net.resolver import build_resolver

    return build_resolver(profile.resolver.kind, **profile.resolver.options).describe()


def _report_filters(albums, people, favorites: bool) -> None:
    parts = []
    if albums:
        parts.append("albums: " + ", ".join(a.name for a in albums))
    if people:
        parts.append("people: " + ", ".join(p.name for p in people))
    if favorites:
        parts.append("favourites only")
    print("filters: " + ("; ".join(parts) if parts else "none (whole library)"))


def _report_result(result, drawn: dict[str, int], output: str) -> None:
    total = drawn["numbered"] + drawn["unnumbered"]
    print(
        f"wrote {output}: {result.stats['regions']} regions, "
        f"{result.stats['colours_used']} colours, "
        f"{drawn['numbered']}/{total} numbered"
    )
    if drawn["unnumbered"]:
        print(
            f"  {drawn['unnumbered']} region(s) were too narrow for a number "
            f"-- {drawn['unnumbered_area_percent']}% of the painted area. "
            f"Raise --min-region-mm to dissolve them."
        )
    if result.stats.get("resolution_capped"):
        print(
            "  note: the photo was smaller than --resolution, so that setting "
            "had no effect here."
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
