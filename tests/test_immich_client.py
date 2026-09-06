import pytest

from immich_pbn.config import ServerProfile
from immich_pbn.immich.client import (
    ImmichClient,
    ImmichError,
    NoMatchingAssets,
    _asset_items,
    _Criteria,
)
from immich_pbn.immich.models import Album, Asset
from immich_pbn.net.transport import HTTPStatusError

PROFILE = ServerProfile(name="t", base_url="http://immich.test:2283", api_key="k")


def asset_json(asset_id, **overrides):
    data = {
        "id": asset_id,
        "type": "IMAGE",
        "originalFileName": f"{asset_id}.jpg",
        "width": 4000,
        "height": 3000,
        "isFavorite": False,
        "localDateTime": "2024-06-01T10:00:00.000Z",
    }
    data.update(overrides)
    return data


class FakeTransport:
    """Records calls and replays canned responses, keyed by path."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def _reply(self, path, payload=None):
        self.calls.append((path, payload))
        value = self.responses.get(path)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(payload)
        return value

    def get_json(self, path, **kwargs):
        self.calls.append((path, kwargs.get("params")))
        value = self.responses.get(path)
        if isinstance(value, Exception):
            raise value
        return value

    def post_json(self, path, payload, **kwargs):
        return self._reply(path, payload)

    def get_bytes(self, path, **kwargs):
        self.calls.append((path, kwargs.get("params")))
        return b"\xff\xd8fake-jpeg"

    def close(self):
        pass


# ------------------------------------------------------------------ dto shapes


@pytest.mark.parametrize(
    "payload",
    [
        [asset_json("a")],
        {"assets": {"items": [asset_json("a")], "nextPage": None}},
        {"assets": [asset_json("a")]},
        {"items": [asset_json("a")]},
    ],
    ids=["bare-list", "assets-items", "assets-list", "items"],
)
def test_every_search_response_shape_immich_has_used_is_accepted(payload):
    # /search/random has returned all of these across releases; a photo picker
    # has no business breaking because the envelope changed.
    assert [item["id"] for item in _asset_items(payload)] == ["a"]


def test_unrecognised_shapes_yield_nothing_rather_than_exploding():
    assert _asset_items({"unexpected": 1}) == []
    assert _asset_items(None) == []


def test_asset_falls_back_to_exif_for_dimensions():
    asset = Asset.from_api(
        {"id": "x", "exifInfo": {"exifImageWidth": 6000, "exifImageHeight": 4000}}
    )
    assert (asset.width, asset.height) == (6000, 4000)
    assert asset.megapixels == pytest.approx(24.0)


# ------------------------------------------------------------------- filtering


def test_flat_payload_is_the_default_dialect():
    payload = _Criteria(album_ids=("A",), person_ids=("P",), favorites_only=True).payload(
        "flat", size=50
    )
    assert payload["albumIds"] == ["A"]
    assert payload["personIds"] == ["P"]
    assert payload["isFavorite"] is True
    assert payload["type"] == "IMAGE"
    assert "filter" not in payload


def test_nested_payload_matches_the_newer_filter_object():
    payload = _Criteria(album_ids=("A",), favorites_only=True).payload("nested", size=50)
    assert payload["filter"]["albumIds"] == {"in": ["A"]}
    assert payload["filter"]["isFavorite"] == {"eq": True}
    assert "albumIds" not in payload


def test_no_filters_still_asks_only_for_still_images():
    payload = _Criteria().payload("flat", size=10)
    assert payload["type"] == "IMAGE"
    assert payload["visibility"] == "timeline"


# --------------------------------------------------------------------- picking


def test_random_uses_the_random_endpoint_when_it_exists():
    transport = FakeTransport({"/search/random": [asset_json("a"), asset_json("b")]})
    client = ImmichClient(PROFILE, transport)
    assert client.random_image().id in {"a", "b"}
    assert transport.calls[0][0] == "/search/random"


def test_random_falls_back_when_the_server_is_too_old():
    transport = FakeTransport(
        {
            "/search/random": HTTPStatusError(404, "/search/random", "Not Found"),
            "/search/metadata": {"assets": {"items": [asset_json("c")], "nextPage": None}},
        }
    )
    client = ImmichClient(PROFILE, transport)
    assert client.random_image().id == "c"
    assert [call[0] for call in transport.calls] == ["/search/random", "/search/metadata"]


def test_a_real_server_error_is_not_swallowed_as_a_missing_endpoint():
    transport = FakeTransport(
        {"/search/random": HTTPStatusError(500, "/search/random", "boom")}
    )
    with pytest.raises(HTTPStatusError):
        ImmichClient(PROFILE, transport).random_image()


def test_videos_and_tiny_images_are_filtered_out_client_side():
    transport = FakeTransport(
        {
            "/search/random": [
                asset_json("vid", type="VIDEO"),
                asset_json("small", width=300, height=200),
                asset_json("good"),
            ]
        }
    )
    asset = ImmichClient(PROFILE, transport).random_image(min_megapixels=1.0)
    assert asset.id == "good"


def test_nothing_usable_explains_which_test_rejected_everything():
    transport = FakeTransport({"/search/random": [asset_json("vid", type="VIDEO")]})
    with pytest.raises(NoMatchingAssets, match="still images"):
        ImmichClient(PROFILE, transport).random_image()


def test_an_empty_result_reports_the_filters_not_a_rejection():
    transport = FakeTransport({"/search/random": []})
    with pytest.raises(NoMatchingAssets, match="album"):
        ImmichClient(PROFILE, transport).random_image(album_ids=("A",))


def test_the_same_seed_picks_the_same_photo():
    import random

    payload = [asset_json(f"a{i}") for i in range(20)]
    picks = set()
    for _ in range(3):
        transport = FakeTransport({"/search/random": list(payload)})
        picks.add(ImmichClient(PROFILE, transport).random_image(rng=random.Random(7)).id)
    assert len(picks) == 1


# ---------------------------------------------------------------- name lookups


ALBUMS = [
    Album(id="11111111-1111-4111-8111-111111111111", name="Iceland 2024", asset_count=90),
    Album(id="22222222-2222-4222-8222-222222222222", name="Iceland 2019", asset_count=40),
    Album(id="33333333-3333-4333-8333-333333333333", name="Garden", asset_count=5),
]


def _client_with_albums():
    client = ImmichClient(PROFILE, FakeTransport())
    client.albums = lambda: ALBUMS  # type: ignore[method-assign]
    return client


def test_exact_name_beats_substring():
    assert _client_with_albums().resolve_albums(["Iceland 2019"])[0].asset_count == 40


def test_a_unique_substring_is_enough():
    assert _client_with_albums().resolve_albums(["garden"])[0].name == "Garden"


def test_an_ambiguous_name_lists_the_candidates_instead_of_guessing():
    with pytest.raises(ImmichError, match="Iceland 2019"):
        _client_with_albums().resolve_albums(["Iceland"])


def test_a_uuid_pasted_from_the_listing_works_without_a_name():
    resolved = _client_with_albums().resolve_albums(["11111111-1111-4111-8111-111111111111"])
    assert resolved[0].name == "Iceland 2024"


def test_a_name_that_matches_nothing_is_an_error_not_a_silent_no_op():
    with pytest.raises(ImmichError, match="Norway"):
        _client_with_albums().resolve_albums(["Norway"])


# ------------------------------------------------------------------- downloads


def test_preview_is_the_default_rendition():
    transport = FakeTransport()
    client = ImmichClient(PROFILE, transport)
    client.download(Asset(id="a"), prefer="preview")
    path, params = transport.calls[-1]
    assert path == "/assets/a/thumbnail" and params == {"size": "preview"}


def test_original_uses_the_original_endpoint():
    transport = FakeTransport()
    ImmichClient(PROFILE, transport).download(Asset(id="a"), prefer="original")
    assert transport.calls[-1][0] == "/assets/a/original"


def test_people_paging_stops_at_the_last_page():
    pages = iter(
        [
            {"people": [{"id": "p1", "name": "Ada"}], "hasNextPage": True},
            {"people": [{"id": "p2", "name": "Grace"}], "hasNextPage": False},
        ]
    )
    transport = FakeTransport()
    transport.get_json = lambda path, **kw: next(pages)  # type: ignore[method-assign]
    names = [p.name for p in ImmichClient(PROFILE, transport).people()]
    assert names == ["Ada", "Grace"]


def test_hidden_people_are_excluded_by_default():
    transport = FakeTransport()
    transport.get_json = lambda path, **kw: {  # type: ignore[method-assign]
        "people": [{"id": "p1", "name": "Ada", "isHidden": True},
                   {"id": "p2", "name": "Grace"}],
        "hasNextPage": False,
    }
    assert [p.name for p in ImmichClient(PROFILE, transport).people()] == ["Grace"]


def test_doctor_can_report_which_search_endpoint_was_used():
    client = ImmichClient(PROFILE, FakeTransport({"/search/random": [asset_json("a")]}))
    assert client.random_endpoint_used is None  # nothing tried yet
    client.random_image()
    assert client.random_endpoint_used == "/search/random"


def test_the_fallback_is_reported_as_such():
    transport = FakeTransport(
        {
            "/search/random": HTTPStatusError(404, "/search/random", "Not Found"),
            "/search/metadata": {"assets": {"items": [asset_json("c")], "nextPage": None}},
        }
    )
    client = ImmichClient(PROFILE, transport)
    client.random_image()
    assert "fallback" in client.random_endpoint_used
