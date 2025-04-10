from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
import functools as ft
import numpy as np
import operator as op

from typing import Optional, Callable, List
from collections.abc import Iterable, Sequence
from pathlib import Path
from cycler import cycler


from fxutil.common import get_git_repo_path


class SaveFigure:
    """
    A class to save figures in different styles and formats.
    """

    def __init__(
        self,
        plot_dir: str | Path = None,
        suffix: str = "",
        output_dpi: int = 250,
        output_transparency: bool = True,
        make_tex_safe: bool = True,
        show_dark: bool = True,
        save_dark: bool = True,
        save_light: bool = True,
        filetypes=None,
        name_str_space_replacement_char: str = "-",
        fig_width_half: float = None,
        fig_width_full: float = None,
        fig_height_max: float = None,
    ):
        """
        Initialize a SaveFigure object.

        Parameters
        ----------
        plot_dir
            Output directory for the figures. If None, the directory is determined
            automatically. If we are inside a git repository `<repo>`, plots are stored
            to `<repo>/data/figures`. If we are not inside a git repository, plots are
            stored to `./figures`, i.e. starting from the current working directory.
        suffix
        output_dpi
        output_transparency
        make_tex_safe
        show_dark
        save_dark
        save_light
        filetypes
        name_str_space_replacement_char
        fig_width_half
            mm
        fig_width_full
            mm
        fig_height_max
            mm
        """
        # TODO: OPACITY!

        if plot_dir is not None:
            plot_dir = Path(plot_dir)
        else:
            try:
                plot_dir = get_git_repo_path() / "data/figures"
            except ValueError:
                plot_dir = Path("./figures")

        self.plot_dirs = {}
        if filetypes is None:
            filetypes = ["pdf", "png"]
        elif isinstance(filetypes, str):
            filetypes = [filetypes]
        if len(filetypes) == 1:
            self.plot_dirs[filetypes[0]] = plot_dir
        else:
            for ext in filetypes or ["pdf", "png"]:
                self.plot_dirs[ext] = plot_dir / ext
                self.plot_dirs[ext].mkdir(exist_ok=True, parents=True)

        self.output_dpi = output_dpi
        self.output_transparency = output_transparency
        self.suffix = suffix
        self.make_tex_safe = make_tex_safe
        self.show_dark = show_dark
        self.save_dark = save_dark
        self.save_light = save_light
        self.name_str_space_replacement_char = name_str_space_replacement_char

        self.fig_width_half = (fig_width_half or 170 / 2) / 25.4
        self.fig_width_full = (fig_width_full or 170) / 25.4
        self.fig_height_max = (fig_height_max or 195) / 25.4

        if self.show_dark:
            plt.style.use(
                [
                    "dark_background",
                    "fxutil.mplstyles.tex",
                    "fxutil.mplstyles.dark",
                ]
            )
        else:
            plt.style.use(
                [
                    "default",
                    "fxutil.mplstyles.tex",
                    "fxutil.mplstyles.light",
                ]
            )

        # stolen from https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/7
        try:
            __IPYTHON__
            self._in_ipython_session = True
            from IPython.display import display

            self._display_plot = lambda fig: display(fig)
        except NameError:
            self._in_ipython_session = False
            self._display_plot = lambda fig: plt.show()

    def __call__(
        self,
        plot_function: Callable,
        name=None,
        fig=None,
        panel: Optional[str] = None,
        extra_artists: Optional[list] = None,
        left=0,
        bottom=0,
        right=1,
        top=1,
        w_pad=0.04167 * 25.4,
        h_pad=0.04167 * 25.4,
        wspace=0.02,
        hspace=0.02,
    ):
        plt.ioff()
        layout_engine_params = dict(
            rect=(left, bottom, right - left, top - bottom),
            w_pad=w_pad / 25.4,
            h_pad=h_pad / 25.4,
            wspace=wspace,
            hspace=hspace,
        )

        plot_function()

        if fig is None:
            fig = plt.gcf()

        fig.get_layout_engine().set(**layout_engine_params)

        styles = {}

        if self.save_dark:
            styles["dark"] = [
                "dark_background",
                "fxutil.mplstyles.tex",
                "fxutil.mplstyles.dark",
            ]
        if self.save_light:
            styles["light"] = [
                "default",
                "fxutil.mplstyles.tex",
                "fxutil.mplstyles.light",
            ]

        for style_name, style in styles.items():
            self._save_figure(
                plot_function=plot_function,
                style_name=style_name,
                style=style,
                name=name,
                fig=fig,
                panel=panel,
                extra_artists=extra_artists,
                layout_engine_params=layout_engine_params,
            )
        plt.ion()
        fig.canvas.draw()
        self._display_plot(fig)

    def _save_figure(
        self,
        plot_function: Callable,
        style_name: str,
        style: [str],
        name=None,
        fig=None,
        panel: Optional[str] = None,
        extra_artists: Optional[list] = None,
        layout_engine_params: Optional[dict] = None,
    ):
        with plt.style.context(style, after_reset=True):
            plot_function()

            if fig is None:
                fig = plt.gcf()

            if layout_engine_params is not None:
                fig.get_layout_engine().set(**layout_engine_params)

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

            name = (name + self.suffix).replace(
                " ", self.name_str_space_replacement_char
            )
            name += self.name_str_space_replacement_char + style_name

            for ext, plot_dir in self.plot_dirs.items():
                fig.savefig(
                    plot_dir / f"{name}.{ext}",
                    dpi=self.output_dpi,
                    transparent=self.output_transparency,
                    bbox_extra_artists=extra_artists,
                )
            plt.close(fig)

    def figax(
        self,
        n_panels=None,
        *,
        n_rows: int | None = None,
        n_cols: int | None = None,
        width_ratios: Sequence[float] = None,
        height_ratios: Sequence[float] = None,
        panel_labels: Optional[bool] = None,
        width=None,
        height=None,
    ):
        if n_panels is None:
            n_rows = n_rows or 1
            n_cols = n_cols or 1
            n_panels = n_rows * n_cols
        else:
            if n_rows is not None and n_cols is None:
                n_cols = n_panels // n_rows + (1 if n_panels % n_rows else 0)
            elif n_rows is None and n_cols is not None:
                n_rows = n_panels // n_cols + (1 if n_panels % n_cols else 0)
            elif n_rows is None and n_cols is None:
                n_cols = 2
                n_rows = n_panels // n_cols + (1 if n_panels % n_cols else 0)
            else:
                if n_rows * n_cols < n_panels:
                    raise ValueError(
                        "n_rows * n_cols must not be smaller "
                        "than n_panels (for obvious reasons)"
                    )

        if panel_labels is None:
            panel_labels = n_rows * n_cols > 1

        fig = plt.figure(
            figsize=(
                width / 25.4 if width is not None else self.fig_width_full,
                (
                    height / 25.4
                    if height is not None
                    else min(self.fig_width_full / n_cols, self.fig_height_max)
                ),
            ),
            dpi=130,
            constrained_layout=True,
        )
        width_ratios = width_ratios if width_ratios is not None else [1] * n_cols
        gs = fig.add_gridspec(
            nrows=n_rows,
            ncols=n_cols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        axs = [*map(fig.add_subplot, gs)]
        if panel_labels:
            for i, ax in enumerate(axs, 97):
                ax.text(
                    -0.15,
                    1.1,
                    rf"\textbf{{({chr(i)})}}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                )
        return fig, axs if len(axs) > 1 else axs[0]


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


def figax(
    figsize: tuple[float, float] = (4, 3), dpi: int = 130, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience function to create matplotlib figure and axis objects with the given
    parameters.

    Parameters
    ----------
    figsize
        Figure size in inches (width, height).
    dpi
        Resolution of the figure (if rasterized output).
    kwargs
        Additional arguments passed to `plt.subplots`.
    Returns
    -------
    fig, ax

    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="compressed", **kwargs)
    return fig, ax


def set_aspect(ratio=3 / 4, axs=None) -> None:
    """
    Set "viewport aspect ratio" (i.e. axes aspect ratio) to the desired value,
    for all axes of the current figure.

    If some axes need to be excluded (like colorbars), supply the axes objects to be
    adjusted manually using the ``axs`` parameter.

    Parameters
    ----------
    ratio
        The desired aspect ratio.
    axs
        The axes objects to be adjusted. If None, all axes of the current figure are
        adjusted.

    """
    if axs is None:
        axs = plt.gcf().get_axes()
    else:
        axs = np.ravel(axs)

    for ax in axs:
        ax.set_aspect(1 / ax.get_data_ratio() * ratio)
