import matplotlib.pyplot as plt
import itertools as it
import functools as ft
import numpy as np
import operator as op


from pathlib import Path
from cycler import cycler


class SaveFigure:
    def __init__(
        self,
        plot_dir,
        suffix: str = "",
        output_dpi: int = 250,
        output_transparency: bool = True,
    ):
        plot_dir = Path(plot_dir)
        self.plot_dirs = {}
        for ext in ["pdf", "png"]:
            self.plot_dirs[ext] = plot_dir / ext
            self.plot_dirs[ext].mkdir(exist_ok=True, parents=True)

        self.output_dpi = output_dpi
        self.output_transparency = output_transparency
        self.suffix = suffix

    def __call__(self, name, fig=None):
        name = (name + self.suffix).replace(" ", "_")
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout()
        for ext, plot_dir in self.plot_dirs.items():
            fig.savefig(
                plot_dir / f"{name}.{ext}",
                bbox_inches="tight",
                dpi=self.output_dpi,
                transparent=self.output_transparency,
            )


def easy_prop_cycle(ax, N=10, cmap="cividis", markers=None):
    cyclers = []
    if cmap is not None:

        cyclers.append(
            cycler(
                "color",
                (plt.cm.get_cmap(cmap)(i) for i in np.r_[0 : 1 : N * 1j]),
            )
        )

    if markers is not None:
        cyclers.append(cycler(marker=it.islice(it.cycle(markers), N)))

    ax.set_prop_cycle(ft.reduce(op.__add__, cyclers))
    return ax


evf = lambda S, f, **arg: (S, f(S, **arg))
