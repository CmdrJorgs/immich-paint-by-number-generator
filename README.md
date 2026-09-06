# immich-paint-by-number-generator

Pulls a random photo from your Immich library — filtered by album, by the
people in it, or by whether you starred it — and turns it into a printable A4
paint-by-number page: outlined regions, a number in each one, and a colour key
with the sRGB hex of every paint.

The page can follow the photo's own colour fields, or ignore them entirely and
impose a lattice — squares for a pixel mosaic, triangles for the low-poly look,
hexagons for something in between.

```
immich-pbn --album "Iceland 2024" --colours 20 -o iceland.pdf
immich-pbn --album "Iceland 2024" --style hex -o iceland-mosaic.pdf
```

```
filters: albums: Iceland 2024
chose: DSC_4471.jpg | 2024-07-14 | 6000x4000 | with Ada
wrote iceland.pdf: 174 regions, 20 colours, 144/174 numbered
  30 region(s) were too narrow for a number -- 2% of the painted area.
  Raise --min-region-mm to dissolve them.
```

## Install

Python 3.11 or newer.

```bash
pip install -e .
```

Pulls in numpy, Pillow, scipy, reportlab and requests. No scikit-learn — the
k-means is forty lines and doing it in-tree means the `--seed` genuinely
controls the result end to end.

## Point it at your server

An API key from **Account Settings → API Keys** in Immich, and a URL:

```bash
export IMMICH_URL=http://immich.local:2283
export IMMICH_API_KEY=...
immich-pbn doctor
```

`doctor` checks the config, reaches the server, reports its version, and tells
you which search endpoint your server actually supports. Start there when
something is off.

For anything beyond one server, copy `immich-pbn.example.toml` to
`~/.config/immich-pbn/config.toml` and use named profiles:

```bash
immich-pbn --album Garden            # the default profile
immich-pbn generate --server away    # a different one, same everything else
```

## Choosing the photo

```bash
immich-pbn albums     # names and asset counts
immich-pbn people     # named people (Immich only lists ones you have named)

immich-pbn --favourites
immich-pbn --person Ada --person Grace       # photos containing either
immich-pbn --album "Iceland 2024" --favourites
immich-pbn --seed 42                         # same photo and palette every time
```

Names are matched case-insensitively, then by unique substring; an ambiguous
name lists the candidates rather than guessing. UUIDs pasted from the listings
work too. Videos and anything under `--min-megapixels` are skipped.

`--from-file photo.jpg` skips Immich entirely, which is how to tune the drawing
settings without hammering your server.

## The dials

| Flag | Default | What it changes |
| --- | --- | --- |
| `--style` | contour | `contour`, `square`, `triangle` or `hex` (see below) |
| `--colours N` | 20 | Paints in the palette, and roughly how many regions |
| `--resolution PX` | 1400 | Working long edge — fidelity, not detail (see below) |
| `--cell-mm MM` | per style | Tiled styles only: cell size in printed mm |
| `--min-region-mm MM` | 3.0 | Contour only: narrowest shape allowed to survive |
| `--dpi N` | 300 | Print density of the outline raster |
| `--line-width-mm MM` | 0.28 | Outline weight |
| `--smoothing R` | 2 | Noise removal before quantization; 0 disables |
| `--number-size-mm MM` | 2.6 | Preferred digit height |
| `--orientation` | auto | `auto` picks whichever wastes less paper |
| `--legend` | inline | `inline` strip, its own `page`, or `none` |
| `--no-reference` | off | Drop the posterised reference page |

Two of these interact in a way worth knowing.

**`--colours` is the difficulty dial.** On a still life at default settings,
8 colours gave 89 regions, 12 gave 119, 20 gave 174 and 32 gave 240. More
paints means more, smaller shapes.

**`--resolution` runs backwards from most people's intuition.** Raising it
gives you *fewer* regions, not more — measured on one photo, 404 regions at
250 px against 304 at 752 px. A hard downscale aliases fine texture into blocky
noise that survives as dozens of separate blobs; sample the picture properly
and the same area resolves into one clean shape with a truthful edge. Region
count is governed by `--min-region-mm`, which is pinned to printed millimetres
and does not drift when you change resolution.

It also only ever downscales. Set it above the photo's own long edge and it
does nothing; the tool says so rather than pretending.

## Styles

`contour` traces the photo's own colour fields, so a region is a shadow or a
petal with an outline of whatever shape that turned out to be. The three tiled
styles do the opposite: they lay a fixed lattice over the picture and give each
cell the average colour of what falls inside it. Same photo, same palette, very
different thing to paint.

On one still life at 20 colours:

| Style | Cell | Cells | Numbered | Feel |
| --- | --- | --- | --- | --- |
| `contour` | — | 174 regions | 67% | Follows the picture |
| `square` | 8.1 mm | 506 | 100% | Pixel mosaic |
| `triangle` | 14.3 mm | 378 | 100% | Low-poly |
| `hex` | 8.9 mm | 495 | 99% | Honeycomb |

Three things fall out of that table.

**Tiled pages number nearly every cell**, where a contour page numbers about
two thirds of its regions. Nothing is wrong with the contour page — its
unnumbered third is slivers totalling a few percent of the painted area — but a
lattice has no slivers to begin with, so every cell has room for its digit.

**Cells snap to fit.** Ask for 8 mm and you get 8.09 mm, because a whole number
of cells has to span the width. Left ragged, the right-hand edge is a quarter of
a square hanging off the frame with no room for its number, which reads as a
printing mistake rather than a design. Whatever still does not divide evenly is
split between both edges instead of piling up on the right and bottom.

**The default cell size is per style, and so is the digit size.** An 8 mm
triangle has barely half the elbow room of an 8 mm square — the inscribed circle
is 0.29 of the side against 0.5 — so triangles default to 14 mm and digits are
sized off the inscribed circle rather than the nominal cell. One shared number
would work for one shape and crowd another.

`--min-region-mm` does nothing on a tiled page and `--cell-mm` does nothing on a
contour one; passing either to the wrong style says so rather than ignoring you.

### Why the lattice is evaluated twice

Each tiling is a pure function of position, so it is asked for the grid twice
per page: once at working resolution to choose colours and place numbers, once
at print resolution to draw the edges. The edges are then the true geometry
rather than an enlarged copy of a small drawing.

That only works if the two evaluations agree exactly, and getting there was the
whole difficulty. The two rasters are the same rectangle in different pixel
counts, so their aspect ratios disagree in the fourth decimal place. Round a row
count out of each independently and a page whose ideal answer is 34.5 rows gets
34 from one and 35 from the other — and prints a half-height row of cells along
the bottom edge, outlined, with no number in any of them. The fix is to derive
the counts from the page's millimetres, which is what the page actually *is*,
and clamp every lattice index into them. No tolerance to tune and no knife-edge
to land on. A test sweeps page shapes against cell sizes asserting the two
lattices come back identical.

## How the page is made

1. **Decode** once, honour EXIF rotation, downscale to the working resolution.
2. **Smooth** with a median filter, which removes JPEG mosquito noise and grain
   without dragging edges around the way a Gaussian does — and edges are
   exactly what becomes the outline.
3. **Quantize** by k-means in CIELAB. RGB distance does not match how different
   two colours look: cluster a sunset in RGB and you get four nearly identical
   oranges and one blue. Best of four restarts, kept by inertia, because one
   unlucky start spent two of six palette slots on near-identical greens and
   lost white from the picture entirely. Colours closer than 2.5 ΔE are then
   merged — below the just-noticeable difference they are the same paint, and
   only waste a legend slot.
4. **Region-build.** On a contour page: quantization alone gives a posterised
   photo whose "regions" include ten thousand two-pixel specks along every
   edge. Anything below the paintable minimum is absorbed into whichever
   neighbour it shares the most border with, nudged towards neighbours of a
   similar colour so a dark eyelash is not swallowed by a bright cheek.
   Absorption changes colours, which can merge further regions, so it runs to a
   fixed point. On a test image this took 44,672 raw regions down to 110.

   On a tiled page the cells *are* the regions, and nothing is merged — that is
   the point rather than an omission. A mosaic's whole appeal is the visible
   grid, and absorbing a cell into its same-coloured neighbour would grow back
   exactly the organic blobs the tiled styles exist to avoid. Cells are also
   clustered directly, rather than pixels being clustered and then voted on, so
   the palette is chosen to represent the things that will actually be painted.
5. **Place the numbers** at each region's pole of inaccessibility — the centre
   of the largest circle that fits inside it. The centroid is the obvious
   answer and the wrong one: on both a crescent and an L-shape it lands outside
   the region entirely. One distance transform serves every region, because
   boundaries are shared.
6. **Render.** Outlines go down as a 1-bit raster computed at the printer's own
   resolution; numbers go down as real vector text. At 300 dpi a pixel is
   0.085 mm — thinner than the pencil line anyone will draw over it — so
   tracing every boundary into a Bézier path would cost a lot and show nothing,
   whereas vector numbers stay sharp at any zoom and searchable in the PDF.

Page two is the photo reduced to the same palette, so you can see what you are
aiming at.

### Regions without numbers

Some regions are too narrow to hold a legible digit. They are left blank, which
is what commercial kits do — a sliver between two shapes gets painted by
inference, and a number crammed into it would foul the outline.

The tool reports this **by area, not by region count**, because the count
alarms without informing. Across the test images, pages came out 11–34%
unnumbered by region count and 2–4% by area — every one of them a sliver.
Raise `--min-region-mm` if you want them dissolved instead.

## Remote servers

Today's assumption is a box on the LAN. The parts that would otherwise have to
be rewritten for a server somewhere else are already factored out.

**Named profiles.** A second server is a second `[servers.*]` block; nothing but
`--server` changes at the call site. Each carries its own URL, key, TLS
setting, timeouts and retry budget.

**Proxy handling by scope.** `scope = "lan"` ignores environment proxies, which
is not a detail: an `HTTPS_PROXY` set for internet traffic will happily swallow
a request to `immich.local`, because a hostname never matches the CIDR entries
people put in `NO_PROXY`. This bit during development — a live request to a
local server died with a connection reset until the proxy was bypassed.
`scope = "remote"` honours them.

**Pluggable hostname resolution.** `resolver.kind` picks the strategy:

| kind | State |
| --- | --- |
| `system` | Implemented. Let the OS do it — the default, and usually right |
| `static` | Implemented. A hosts file that lives in the config, not `/etc` |
| `system+static` | Implemented. Try the OS, fall back to a pinned address |
| `dns` | Registered, **not implemented**. Query an explicit nameserver |
| `mdns` | Registered, **not implemented**. Discover the host by DNS-SD |

The two unimplemented strategies are registered on purpose and fail with a
message naming what they would need — a strategy that fails loudly beats one
nobody remembered wanting. Adding one means writing a class and calling
`register_resolver`; nothing in the client, transport or CLI changes.

The implemented ones are not decorative. When a resolver supplies an address,
the connection dials it while the `Host` header and the TLS server name stay on
the configured name, so virtual hosts still route and certificates still
validate. Both the plain and the TLS path are covered by tests that stand up a
real local server, including one with a self-signed certificate.

**API drift.** `/search/random` is preferred, with a fallback to paging
`/search/metadata` on servers too old to have it, and all four response
envelopes Immich has used across releases are accepted. The flat filter fields
(`isFavorite`, `personIds`, `albumIds`) are deprecated as of the 3.2 API in
favour of a nested `filter` object; deprecated is not removed, so flat stays
the default and nested is available for when that changes.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

189 tests. The interesting ones pin behaviour that is easy to get quietly
wrong: that the number lands inside a crescent where the centroid does not,
that the lattice drawn at print resolution is the same one that was coloured at
working resolution, that a triangle cell really is equilateral and not a
bisected square, that the resolver redirects a live connection without breaking
TLS identity, that a mid-olive swatch gets black digits rather than unreadable
white ones, and that every response shape Immich has ever returned still
parses.

## Known limitations

- **Not tested against a live Immich server.** Every endpoint, parameter and
  response shape was checked against the official OpenAPI specification, and
  the client is covered by tests using recorded response shapes — but no real
  server was available while writing this. `immich-pbn doctor` exists to make
  the first contact diagnosable rather than mysterious.
- On a **contour** page the label map is upscaled to print resolution with
  nearest-neighbour, so region boundaries carry a small stair-step. At 300 dpi
  with a 1400 px working image the steps are about 0.2 mm, under the outline's
  own weight. Raise `--resolution` toward the print pixel width if it ever
  shows. Tiled styles do not have this: their edges are re-derived at print
  resolution from the geometry itself.
- A hexagon lattice cannot meet a rectangle cleanly — alternate rows end in half
  cells at the left and right edges. That is what a hex grid in a frame looks
  like rather than a defect, but it does mean a handful of edge cells are
  smaller than the rest.
- A4 only. The page geometry is in millimetres throughout and `PageLayout`
  takes its size from two constants, so US Letter is a small change, not a
  rewrite — but it is not wired to a flag today.
- CMYK and 16-bit-per-channel sources are converted to 8-bit sRGB on load.
