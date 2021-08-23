import warnings

import matplotlib.pyplot as plt
import itertools as it
import functools as ft
import numpy as np
import operator as op

from collections.abc import Iterable, Sequence
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


solarized_colors = dict(
    base03="#002b36",
    base02="#073642",
    base01="#586e75",
    base00="#657b83",
    base0="#839496",
    base1="#93a1a1",
    base2="#eee8d5",
    base3="#fdf6e3",
    yellow="#b58900",
    orange="#cb4b16",
    red="#dc322f",
    magenta="#d33682",
    violet="#6c71c4",
    blue="#268bd2",
    cyan="#2aa198",
    green="#859900",
)


def easy_prop_cycle(ax, N=10, cmap="cividis", markers=None):
    cyclers = []
    if cmap is not None:
        cycle = []
        if isinstance(cmap, str):
            if cmap == "solarized":
                scs = (
                    # "base1",
                    # "base2",
                    "yellow",
                    "orange",
                    "red",
                    "magenta",
                    "violet",
                    "blue",
                    "cyan",
                    "green",
                )
                cycle = [solarized_colors[sc] for sc in scs]
            else:
                cycle = [plt.cm.get_cmap(cmap)(i) for i in np.r_[0 : 1 : N * 1j]]
        elif isinstance(cmap, Iterable):
            cycle = list(cmap)
        else:
            raise TypeError(f"incompatible cmap type: {type(cmap)}")

        if len(cycle) != N:
            warnings.warn(
                f"{N=}, but number of colors in cycle is {len(cycle)}.", UserWarning
            )
        cyclers.append(
            cycler(
                "color",
                cycle,
            )
        )

    if markers is not None:
        cyclers.append(cycler(marker=it.islice(it.cycle(markers), N)))

    ax.set_prop_cycle(ft.reduce(op.__add__, cyclers))
    return ax


evf = lambda S, f, **arg: (S, f(S, **arg))
"""
Use like 
`ax.plot(*evf(np.r_[0:1:50j], lambda x, c: x ** 2 + c, c=5))`
"""


def figax(figsize=(4, 3), dpi=130):
    return plt.subplots(figsize=figsize, dpi=dpi)
