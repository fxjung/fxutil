import itertools as it
import numpy as np

import pytest

from fxutil.plotting import SaveFigure, evf


@pytest.mark.parametrize("latex,gridspec", it.product(*[[False, True]] * 2))
def test_basic_plotting(latex, gridspec, tmpdir, plot_fn_factory):
    sf = SaveFigure(
        tmpdir,
        interactive_mode=None,
        subfolder_per_filetype=True,
        width=100,
        output_dpi=300,
        filetypes=["png", "pdf"],
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
        filetypes_parsed = ["png"]
    elif isinstance(filetypes, str):
        filetypes_parsed = [filetypes]
    else:
        filetypes_parsed = filetypes

    for ext in filetypes_parsed:
        assert (tmpdir / f"basic-plot-light.{ext}").exists()


def test_evf():
    x, y = evf(np.r_[0:5:10j], lambda x: x / 2)

    assert np.array_equal(x, np.r_[0:5:10j])
    assert np.array_equal(y, x / 2)

    x, y = evf[0:5:10j, lambda x: x / 2]

    assert np.array_equal(x, np.r_[0:5:10j])
    assert np.array_equal(y, x / 2)
