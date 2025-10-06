import pytest

from matplotlib import pyplot as plt
import numpy as np
import itertools as it

from fxutil import evf
from fxutil.plotting import SaveFigure, figax


@pytest.mark.parametrize("latex,gridspec", it.product(*[[False, True]] * 2))
def test_basic_plotting(latex, gridspec, tmpdir, plot_fn_factory):
    sf = SaveFigure(tmpdir, interactive_mode=None)
    if gridspec:
        plot = plot_fn_factory(latex=latex, sf=sf)
    else:
        plot = plot_fn_factory(latex=latex)

    sf(plot, "basic plot")

    for ext, style in it.product(["png", "pdf"], ["light", "dark"]):
        assert (tmpdir / ext / f"basic-plot-{style}.{ext}").exists()
