import pytest

from immich_pbn.cli import build_parser, main


def test_generate_is_the_default_command(tmp_path, photo_like):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    output = tmp_path / "page.pdf"
    # No subcommand, straight to the flags.
    assert main(["--from-file", str(source), "-c", "8", "-r", "320", "-o", str(output)]) == 0
    assert output.exists() and output.stat().st_size > 1000


def test_generate_can_also_be_named(tmp_path, photo_like):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    output = tmp_path / "page.pdf"
    assert main(
        ["generate", "--from-file", str(source), "-c", "8", "-r", "320", "-o", str(output)]
    ) == 0
    assert output.exists()


def test_british_and_american_spellings_both_work():
    parser = build_parser()
    assert parser.parse_args(["generate", "--colours", "9"]).colours == 9
    assert parser.parse_args(["generate", "--colors", "9"]).colours == 9
    assert parser.parse_args(["generate", "--favourites"]).favorites is True
    assert parser.parse_args(["generate", "--favorites"]).favorites is True


def test_filters_accumulate():
    args = build_parser().parse_args(
        ["generate", "--album", "Iceland", "--album", "Norway", "--person", "Ada"]
    )
    assert args.album == ["Iceland", "Norway"]
    assert args.person == ["Ada"]


def test_a_missing_local_file_is_reported_not_traced(tmp_path, capsys):
    assert main(["--from-file", str(tmp_path / "nope.jpg")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_no_api_key_gives_advice_rather_than_a_stack_trace(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    monkeypatch.setenv("IMMICH_PBN_CONFIG", str(tmp_path / "empty.toml"))
    (tmp_path / "empty.toml").write_text('[servers.home]\nbase_url = "http://h:2283"\n')
    # Subcommand first, then its flags -- `--server` belongs to the subcommand.
    assert main(["albums", "--server", "home"]) == 1
    assert "API key" in capsys.readouterr().err


def test_global_flags_work_before_or_after_the_subcommand(tmp_path):
    config = tmp_path / "c.toml"
    config.write_text('[servers.home]\nbase_url = "http://h:2283"\n')
    parser = build_parser()
    before = parser.parse_args(["--config", str(config), "-v", "albums"])
    after = parser.parse_args(["albums", "--config", str(config), "-v"])
    assert (before.config or before.config_sub) == str(config)
    assert (after.config or after.config_sub) == str(config)
    assert before.verbose + before.verbose_sub == 1
    assert after.verbose + after.verbose_sub == 1


def test_the_default_command_is_inserted_after_global_flags():
    from immich_pbn.cli import _with_default_command

    assert _with_default_command(["--from-file", "x.jpg"]) == ["generate", "--from-file", "x.jpg"]
    assert _with_default_command(["--config", "c.toml", "albums"]) == ["--config", "c.toml", "albums"]
    assert _with_default_command(["--config", "c.toml", "--from-file", "x"]) == [
        "--config", "c.toml", "generate", "--from-file", "x",
    ]
    assert _with_default_command(["-v", "doctor"]) == ["-v", "doctor"]
    assert _with_default_command([]) == ["generate"]
    assert _with_default_command(["--help"]) == ["--help"]


def test_a_broken_config_file_exits_two_not_one(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[servers.home]\nbase_url = "http://h"\nnonsense = 1\n')
    assert main(["--config", str(bad), "albums"]) == 2
    assert "configuration problem" in capsys.readouterr().err


def test_doctor_reports_a_missing_key_without_touching_the_network(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    config = tmp_path / "c.toml"
    config.write_text('[servers.home]\nbase_url = "http://immich.invalid:2283"\n')
    assert main(["--config", str(config), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "bypassed" in out


def test_legend_and_reference_flags_reach_the_pdf(tmp_path, photo_like):
    pypdfium2 = pytest.importorskip("pypdfium2")
    source = tmp_path / "photo.png"
    photo_like.save(source)
    output = tmp_path / "bare.pdf"
    assert main(
        ["--from-file", str(source), "-c", "6", "-r", "320", "--legend", "none",
         "--no-reference", "-o", str(output)]
    ) == 0
    assert len(pypdfium2.PdfDocument(str(output))) == 1


# ------------------------------------------------------------------- styles


@pytest.mark.parametrize("style", ["contour", "square", "triangle", "hex"])
def test_every_style_is_reachable_from_the_command_line(style, tmp_path, photo_like):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    output = tmp_path / f"{style}.pdf"
    assert main(
        ["--from-file", str(source), "--style", style, "-c", "8", "-r", "320",
         "-o", str(output)]
    ) == 0
    assert output.exists() and output.stat().st_size > 1000


def test_an_unknown_style_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate", "--style", "pentagon"])


def test_cell_size_defaults_per_style():
    from immich_pbn.pbn.tessellation import build_tessellation

    # Not one shared number: an 8 mm triangle has half the room of an 8 mm square.
    assert build_tessellation("triangle").default_cell_mm > build_tessellation("square").default_cell_mm


def test_min_region_mm_on_a_tiled_page_says_it_does_nothing(tmp_path, photo_like, capsys):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    assert main(
        ["--from-file", str(source), "--style", "square", "--min-region-mm", "5",
         "-c", "6", "-r", "320", "-o", str(tmp_path / "o.pdf")]
    ) == 0
    assert "--min-region-mm applies only to --style contour" in capsys.readouterr().err


def test_cell_mm_on_a_contour_page_says_it_does_nothing(tmp_path, photo_like, capsys):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    assert main(
        ["--from-file", str(source), "--cell-mm", "12", "-c", "6", "-r", "320",
         "-o", str(tmp_path / "o.pdf")]
    ) == 0
    assert "--cell-mm applies only to a tiled style" in capsys.readouterr().err


def test_a_tiled_page_reports_cells_rather_than_regions(tmp_path, photo_like, capsys):
    source = tmp_path / "photo.png"
    photo_like.save(source)
    main(["--from-file", str(source), "--style", "hex", "-c", "6", "-r", "320",
          "-o", str(tmp_path / "o.pdf")])
    assert "cells" in capsys.readouterr().out
