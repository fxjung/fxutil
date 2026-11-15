"""Render representative presentation plots and check their actual artist layout.

For visual review (both PNG and PDF, light and dark)::

    MPLBACKEND=Agg python test/test_presentation_rendering.py /tmp/fxutil-preview
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fxutil.plotting import SaveFigure


def draw_examples(sf: SaveFigure, panels: int) -> tuple[plt.Figure, np.ndarray]:
    """Exercise curves, symbols, uncertainty, log ticks, and explicit grids."""
    fig, axes = sf.figax(
        n_rows=2 if panels == 4 else 1, n_cols=2 if panels == 4 else 1, panel_labels=[]
    )
    x = np.linspace(0, 6, 150)
    for index, ax in enumerate(np.atleast_1d(axes)):
        if index == 0:
            ax.plot(x, np.exp(-x / 3), label="Theory")
            sample = np.linspace(0.3, 5.7, 8)
            ax.errorbar(
                sample,
                np.exp(-sample / 3) + 0.02 * np.sin(sample * 2),
                yerr=0.035,
                fmt="o",
                label="Experiment",
            )
            ax.set_title("Relaxation")
            ax.legend(loc="upper right")
        elif index == 1:
            for offset, marker in [(0, "o"), (0.5, "s"), (1, "^")]:
                ax.plot(x, np.sin(x + offset), marker=marker, markevery=25)
            ax.set_title("Oscillations")
            ax.grid(True)
        elif index == 2:
            (line,) = ax.semilogy(x, np.exp(-x), marker="s", markevery=25)
            ax.fill_between(
                x,
                0.7 * np.exp(-x),
                1.3 * np.exp(-x),
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )
            ax.grid(True, axis="y")
            ax.set_title("Decay")
        else:
            ax.plot(x, x / (1 + x), marker="D", markevery=25)
            ax.axhline(0.8, linestyle="--", color="contrast", alpha=0.4)
            ax.minorticks_on()
            ax.set_title("Saturation")
        ax.set_xlabel(r"Time $t$ (s)")
        ax.set_ylabel(r"Response $R(t)$")
        # A restrained number of labels is easier to read on smaller panels.
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        if index != 2:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    return fig, np.atleast_1d(axes)


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("panels", [1, 4])
def test_rendered_presentation(scheme, panels, tmp_path, monkeypatch):
    sf = SaveFigure(
        tmp_path,
        presentation=True,
        use_styles=[scheme],
        interactive_mode=None,
        filetypes=["png", "pdf"],
        output_dpi=100,
    )
    before = mpl.rcParams.copy()
    savefig = plt.Figure.savefig

    def inspect_export(fig, *args, **kwargs):
        # Check the actual export, after SaveFigure's text sanitization.
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        assert titles == ["Relaxation", "Oscillations", "Decay", "Saturation"][:panels]
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", inspect_export)

    def plot():
        # Exercise native mathtext without requiring LaTeX or a GUI.
        mpl.rcParams["text.usetex"] = False
        fig, axes = draw_examples(sf, panels)
        fig.set_layout_engine("constrained")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            assert ax.lines[0].get_linewidth() == 3.2
            assert ax.spines["left"].get_linewidth() == 1.6
            # Titles align with their data viewport, and grids stay under fills.
            assert ax.get_title(loc="left")
            title_bounds = ax._left_title.get_window_extent(renderer)
            assert abs(title_bounds.x0 - ax.bbox.x0) < 1
            assert ax.xaxis.get_zorder() < ax.lines[0].get_zorder()
            if ax.collections:
                assert ax.yaxis.get_zorder() < ax.collections[0].get_zorder()
            tick = ax.xaxis.get_major_ticks()[0]
            assert tick.tick1line.get_markersize() == 7
            assert tick.tick1line.get_markeredgewidth() == 1.6
            # Locators also create ticks outside the visible data limits; these
            # are not drawn and should not be included in the clipping check.
            visible_ticks = [
                tick.label1
                for axis in [ax.xaxis, ax.yaxis]
                for tick in axis.get_major_ticks()
                if min(axis.get_view_interval())
                <= tick.get_loc()
                <= max(axis.get_view_interval())
            ]
            # Labels must fit inside the nominal canvas before tight export.
            for text in [
                ax.xaxis.label,
                ax.yaxis.label,
                ax._left_title,
                *visible_ticks,
            ]:
                bounds = text.get_window_extent(renderer)
                assert bounds.x0 >= 0 and bounds.y0 >= 0
                assert bounds.x1 <= fig.bbox.width
                assert bounds.y1 <= fig.bbox.height
            # Text should leave a substantial viewport for the data.
            position = ax.get_position()
            assert position.width > (0.25 if panels == 4 else 0.65)
            assert position.height > (0.22 if panels == 4 else 0.65)
        markers = axes[0].lines[1]
        assert markers.get_markersize() == 9
        assert markers.get_markeredgewidth() == 1.2
        caps = axes[0].containers[0].lines[1]
        assert len(caps) == 2
        assert all(cap.get_markersize() == 6 for cap in caps)
        if panels == 4:
            minor_tick = axes[3].xaxis.get_minor_ticks()[0]
            assert minor_tick.tick1line.get_markersize() == 4
            assert minor_tick.tick1line.get_markeredgewidth() == 1.1
        legend = axes[0].get_legend()
        assert legend.legend_handles[0].get_linewidth() == 3.2

    sf(plot, f"gallery-{panels}")
    assert dict(mpl.rcParams) == dict(before)
    pixels = plt.imread(tmp_path / f"gallery-{panels}-{scheme}.png")
    assert pixels.shape[0] > 500 and pixels.shape[1] > 800
    assert np.std(pixels[:, :, :3]) > 0.05
    assert (tmp_path / f"gallery-{panels}-{scheme}.pdf").stat().st_size > 1000


@pytest.mark.parametrize("scheme,background", [("light", 1), ("dark", 0)])
def test_presentation_foreground_contrast(scheme, background):
    sf = SaveFigure(presentation=True)

    def contrast(color):
        rgb = np.array(mpl.colors.to_rgb(color))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = linear @ [0.2126, 0.7152, 0.0722]
        return (max(luminance, background) + 0.05) / (min(luminance, background) + 0.05)

    with plt.style.context(sf.styles[scheme], after_reset=True):
        text_contrast = contrast(mpl.rcParams["text.color"])
        assert text_contrast >= 7
        for key in [
            "axes.labelcolor",
            "axes.titlecolor",
            "xtick.labelcolor",
            "ytick.labelcolor",
        ]:
            assert contrast(mpl.rcParams[key]) >= 7
        for key in ["axes.edgecolor", "xtick.color", "ytick.color"]:
            assert 3 <= contrast(mpl.rcParams[key]) < text_contrast


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1])
    for count in [1, 4]:
        saver = SaveFigure(
            output,
            presentation=True,
            interactive_mode=None,
            output_transparency=False,
            filetypes=["png", "pdf"],
        )
        # Composite preview exports onto a slide background; production exports
        # retain their existing transparent backgrounds.
        saver.styles["dark"].append({"figure.facecolor": "black"})
        saver(
            lambda sf=saver, panels=count: draw_examples(sf, panels),
            f"gallery-{count}",
        )
    print(f"Presentation previews written to {output.resolve()}")
