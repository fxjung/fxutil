from __future__ import annotations

import logging
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
import functools as ft
import numpy as np
import operator as op
import matplotlib.colors as mpc

from typing import Optional, Callable
from collections.abc import Iterable, Sequence
from pathlib import Path
from cycler import cycler


from fxutil.common import get_git_repo_path

log = logging.getLogger(__name__)


def evf(S, f, **kwargs):
    """
    Use like
    `ax.plot(*evf(np.r_[0:1:50j], lambda x, c: x ** 2 + c, c=5))`

    Parameters
    ----------
    S
        Space
    f
        function
    **kwargs
        Additional function args
    """
    return S, f(S, **kwargs)


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


def pad_range(left, right, pad=0.03, log=False):
    """
    Pad plots ranges to achieve equal-distance padding on both sides.

    Use like

    ```python
        ax.set_xlim(*pad_range(*minmax(values)))
    ```

    Parameters
    ----------
    left
        Left unpadded bound
    right
        Right unpadded bound
    pad
        Padding to  be added on both sides as a fraction.
    log
        If True, assume log scale

    Returns
    -------
    bounds
        Tuple of left and right bounds

    """
    if log:
        left = np.log(left)
        right = np.log(right)

    d = right - left
    p = d / (1 / pad - 2)

    lp = left - p
    rp = right + p

    if log:
        lp = np.exp(lp)
        rp = np.exp(rp)

    return lp, rp


class SaveFigure:
    """
    A class to save figures in different styles and formats.
    """

    _base_plot_dir: Path
    _in_ipython_session: bool | None = None

    styles: dict[str, list[str]] = {
        "dark": [
            "dark_background",
            "fxutil.mplstyles.tex",
            "fxutil.mplstyles.dark",
        ],
        "light": [
            "default",
            "fxutil.mplstyles.tex",
            "fxutil.mplstyles.light",
        ],
    }

    def __init__(
        self,
        plot_dir: str | Path = None,
        suffix: str = "",
        output_dpi: int = 250,
        output_transparency: bool = True,
        make_tex_safe: bool = True,
        interactive_mode: str | None = "dark",
        use_styles: list[str] | None = None,
        filetypes=None,
        name_str_space_replacement_char: str = "-",
        fig_width_half: float = None,
        fig_width_full: float = None,
        fig_height_max: float = None,
        # TODO what about these guys?
        left=0,
        bottom=0,
        right=1,
        top=1,
        w_pad=0.04167 * 25.4,
        h_pad=0.04167 * 25.4,
        wspace=0.02,
        hspace=0.02,
        layout: str | mpl.layout_engine.LayoutEngine = "constrained",
        subfolder_per_filetype=False,
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
        interactive_mode
            Valid values are styles such as "dark", "light", and None to not show anything.
        use_styles
            Styles to save plots in. Valid values are at least "dark" and "light".
            ``None`` saves plots in all availabel styles.
        save_dark
        save_light
        filetypes
            Filetypes to save plots as. Defaults to ["pdf", "png"]
        name_str_space_replacement_char
        fig_width_half
            mm
        fig_width_full
            mm
        fig_height_max
            mm
        subfolder_per_filetype
            If True, create a subfolder for each filetype in the plot_dir.
        """
        # TODO: OPACITY!

        if plot_dir is not None:
            plot_dir = Path(plot_dir)
        else:
            try:
                plot_dir = get_git_repo_path() / "data/figures"
            except ValueError:
                plot_dir = Path("./figures")

        self._base_plot_dir = plot_dir

        self.filetypes = self._parse_filetypes(filetypes)
        self.subfolder_per_filetype = subfolder_per_filetype

        self.output_dpi = output_dpi
        self.output_transparency = output_transparency
        self.suffix = suffix
        self.make_tex_safe = make_tex_safe
        self.name_str_space_replacement_char = name_str_space_replacement_char
        self.use_styles = use_styles or [*self.styles.keys()]
        self.interactive_mode = interactive_mode

        self.fig_width_half = (fig_width_half or 170 / 2) / 25.4
        self.fig_width_full = (fig_width_full or 170) / 25.4
        self.fig_height_max = (fig_height_max or 195) / 25.4

        self.layout = layout

        # stolen from https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/7
        try:
            __IPYTHON__
            self._in_ipython_session = True

            from IPython.display import display

            self._display_plot = lambda fig: display(fig)
        except NameError:
            self._in_ipython_session = False
            self._display_plot = lambda fig: plt.show()

        self.layout_engine_params = dict(
            rect=(left, bottom, right - left, top - bottom),
            w_pad=w_pad / 25.4,
            h_pad=h_pad / 25.4,
            wspace=wspace,
            hspace=hspace,
        )

    def __call__(
        self,
        plot_function: Callable,
        name=None,
        fig=None,
        panel: Optional[str] = None,
        extra_artists: Optional[list] = None,
        filetypes=None,
    ):
        for style_name in self.use_styles:
            style = self.styles[style_name]
            log.info(f"Saving figure in {style_name} style...")
            self._save_figure(
                plot_function=plot_function,
                style_name=style_name,
                style=style,
                name=name,
                fig=fig,
                panel=panel,
                extra_artists=extra_artists,
                layout_engine_params=self.layout_engine_params,
                filetypes=filetypes,
            )

        if (style_name := self.interactive_mode) is not None:
            self.register_contrast_color(style_name)
            with plt.style.context(self.styles[style_name], after_reset=True):
                plot_function()
                fig = plt.gcf()
                # fig.canvas.draw()  # Does not seem to be necessary
                if style_name == "dark" and not self._in_ipython_session:
                    if (backend := mpl.get_backend()) == "qtagg":
                        fig.canvas.manager.window.setStyleSheet(
                            "background-color: black;"
                        )
                    elif backend == "tkagg":
                        fig.canvas.manager.window.config(bg="black")
                self._display_plot(fig)
                plt.close(fig)

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
        filetypes=None,
    ):
        self.register_contrast_color(style_name)
        with plt.style.context(style, after_reset=True):
            plot_function()

            if fig is None:
                fig = plt.gcf()

            # if layout_engine_params is not None:
            #     fig.set_layout_engine(self.layout)
            #     fig.get_layout_engine().set(**layout_engine_params)

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

            for ext in (
                self._parse_filetypes(filetypes) if filetypes else self.filetypes
            ):
                fig.savefig(
                    self._get_plot_dir(ext) / f"{name}.{ext}",
                    dpi=self.output_dpi,
                    transparent=self.output_transparency,
                    bbox_extra_artists=extra_artists,
                )
            plt.close(fig)

    @staticmethod
    def register_contrast_color(style_name: str):
        match style_name:
            case "light":
                contrast = "#000000"  # can't be named color
                acontrast = "#ffffff"  # can't be named color
            case "dark":
                contrast = "#ffffff"  # can't be named color
                acontrast = "#000000"  # can't be named color
            case _:
                raise ValueError(f"Invalid style name {style_name}")

        mpc._colors_full_map["contrast"] = contrast
        mpc._colors_full_map["acontrast"] = acontrast

    def figax(
        self,
        n_panels=None,
        *,
        n_rows: int | None = None,
        n_cols: int | None = None,
        left=None,
        right=None,
        top=None,
        bottom=None,
        wspace=None,  # FIXME are these actually used, or are we obeying the layout_engine's as set through the __init__ kwargs?
        hspace=None,
        width_ratios: Sequence[float] = None,
        height_ratios: Sequence[float] = None,
        panel_labels: Optional[bool] = None,
        width=None,
        height=None,
    ):
        """

        Parameters
        ----------
        n_panels
            Number of panels to create
        n_rows
            Number of rows to squeeze the panels into
        n_cols
            Number of columns to squeeze the panels into
        width_ratios
        height_ratios
        panel_labels
        width
        height

        Returns
        -------

        fig
            Figure
        axs
            A single axes or a tuple of multiple axes
        """
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
            dpi=self.output_dpi,
            constrained_layout=True,
        )
        width_ratios = width_ratios if width_ratios is not None else [1] * n_cols
        gs = fig.add_gridspec(
            nrows=n_rows,
            ncols=n_cols,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            wspace=wspace,
            hspace=hspace,
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

    @ft.cache
    def _get_plot_dir(self, filetype: str):
        plot_dir = self._base_plot_dir
        if self.subfolder_per_filetype:
            plot_dir /= filetype
        plot_dir.mkdir(exist_ok=True, parents=True)
        return plot_dir

    def _parse_filetypes(self, filetypes: str | list[str] | None) -> list[str]:
        if not filetypes:
            filetypes = ["pdf", "png"]
        elif isinstance(filetypes, str):
            filetypes = [filetypes]

        return filetypes

    @property
    def output_dir(self):
        return self._base_plot_dir
