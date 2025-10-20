import itertools as it

import pytest

from fxutil.plotting import SaveFigure


@pytest.mark.parametrize("latex,gridspec", it.product(*[[False, True]] * 2))
def test_basic_plotting(latex, gridspec, tmpdir, plot_fn_factory):
    sf = SaveFigure(
        tmpdir,
        interactive_mode=None,
        subfolder_per_filetype=True,
        width=100,
        output_dpi=300,
    )
    if gridspec:
        plot = plot_fn_factory(latex=latex, sf=sf)
    else:
        plot = plot_fn_factory(latex=latex)

    sf(plot, "basic plot")

    for ext, style in it.product(["png", "pdf"], ["light", "dark"]):
        assert (tmpdir / ext / f"basic-plot-{style}.{ext}").exists()


@pytest.mark.parametrize("filetypes", [None, "png", ["png"], ["png", "pdf"]])
def test_filetype_combi_args(filetypes, tmpdir, plot_fn_factory):
    sf = SaveFigure(
        tmpdir,
        interactive_mode=None,
        width=100,
        output_dpi=300,
        filetypes=filetypes,
        use_styles=["light"],
    )

    plot = plot_fn_factory(latex=False, sf=sf)

    sf(plot, "basic plot")

    if filetypes is None:
        filetypes_parsed = ["png", "pdf"]
    elif isinstance(filetypes, str):
        filetypes_parsed = [filetypes]
    else:
        filetypes_parsed = filetypes

    for ext in filetypes_parsed:
        assert (tmpdir / f"basic-plot-light.{ext}").exists()
