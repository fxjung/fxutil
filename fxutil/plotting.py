import warnings

import matplotlib.pyplot as plt
import itertools as it
import functools as ft
import numpy as np
import operator as op

from typing import Optional
from collections.abc import Iterable, Sequence
from pathlib import Path
from cycler import cycler


def set_plot_dark():
    default_cycler = plt.rcParams["axes.prop_cycle"]
    plt.style.use("dark_background")
    plt.rcParams["axes.prop_cycle"] = default_cycler
    plt.rcParams["axes.facecolor"] = (1, 1, 1, 0)
    plt.rcParams["figure.facecolor"] = (1, 1, 1, 0)
    plt.rcParams["legend.framealpha"] = None
    plt.rcParams["legend.facecolor"] = (0, 0, 0, 0.1)


class SaveFigure:
    def __init__(
        self,
        plot_dir,
        suffix: str = "",
        output_dpi: int = 250,
        output_transparency: bool = True,
        make_tex_safe: bool = True,
        dark: bool = True,
        filetypes=None,
        name_str_space_replacement_char: str = "_",
    ):
        plot_dir = Path(plot_dir)
        self.plot_dirs = {}
        for ext in filetypes or ["pdf", "png"]:
            self.plot_dirs[ext] = plot_dir / ext
            self.plot_dirs[ext].mkdir(exist_ok=True, parents=True)

        self.output_dpi = output_dpi
        self.output_transparency = output_transparency
        self.suffix = suffix
        self.make_tex_safe = make_tex_safe
        self.dark = dark
        self.name_str_space_replacement_char = name_str_space_replacement_char

        if self.dark:
            set_plot_dark()

    def __call__(
        self,
        name,
        fig=None,
        panel: Optional[str] = None,
        extra_artists: Optional[list] = None,
    ):
        name = (name + self.suffix).replace(" ", self.name_str_space_replacement_char)
        if fig is None:
            fig = plt.gcf()

        extra_artists = extra_artists or []

        # TODO multiple axes
        axs = fig.get_axes()
        if isinstance(axs, plt.Axes):
            axs = [[axs]]
        elif len(np.shape(axs)) == 1:
            axs = [axs]

        for ax in np.ravel(axs):
            legend = ax.get_legend()

            if self.make_tex_safe:
                if "$" not in (label := ax.get_xlabel()):
                    ax.set_xlabel(label.replace("_", " "))

                if "$" not in (label := ax.get_ylabel()):
                    ax.set_ylabel(label.replace("_", " "))

                if "$" not in (label := ax.get_title()):
                    ax.set_title(label.replace("_", " "))

                if legend is not None:
                    for text in legend.texts:
                        if "$" not in (label := text.get_text()):
                            text.set_text(label.replace("_", " "))

                    if "$" not in (label := legend.get_title().get_text()):
                        legend.set_title(label.replace("_", " "))

            # if panel is not None:
            #     ax.text(
            #         ax.get_xlim()[0],
            #         ax.get_ylim()[1],
            #         panel,
            #         va="top",
            #         ha="left",
            #         backgroundcolor="k" if self.dark else "w",
            #         color="w" if self.dark else "k",
            #     )

            if legend is not None:
                extra_artists.append(legend)

        if fig._suptitle is not None:
            extra_artists.append(fig._suptitle)

        # TODO this still needs to be called beforehand sometimes (in the calling code) WHY??
        fig.tight_layout()

        for ext, plot_dir in self.plot_dirs.items():
            fig.savefig(
                plot_dir / f"{name}.{ext}",
                bbox_inches="tight",
                dpi=self.output_dpi,
                transparent=self.output_transparency,
                bbox_extra_artists=extra_artists,
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


def figax(figsize=(4, 3), dpi=130, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
    return fig, ax


def set_aspect(ratio=3 / 4, axs=None):
    """
    Set "viewport aspect ratio" (i.e. axes aspect ratio) to the desired value,
    for all axes of the current figure.

    If some axes need to be excluded (like colorbars), supply the axes objects manually
    using the ``axs`` parameter.

    Parameters
    ----------
    ratio
    axs

    Returns
    -------

    """
    if axs is None:
        axs = plt.gcf().get_axes()
    else:
        axs = np.ravel(axs)

    for ax in axs:
        ax.set_aspect(1 / ax.get_data_ratio() * ratio)
