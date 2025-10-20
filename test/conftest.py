import pytest

from fxutil import SaveFigure


@pytest.fixture(scope="session")
def plot_fn_factory():
    import numpy as np

    from fxutil.plotting import evf, figax

    def make_plotter(latex=False, sf: SaveFigure | None = None):
        def _plot():
            if sf is None:
                fig, ax = figax()
            else:
                fig, ax = sf.figax()
            for b in [1, 2, 3]:
                ax.plot(*evf(np.r_[0:1:300j], lambda x: x**b), label="curve")
                ax.axvline(0.5, color="contrast", label=r"line $x=0.5$")

            if latex:
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
            else:
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            ax.legend(loc=2)

        return _plot

    return make_plotter
