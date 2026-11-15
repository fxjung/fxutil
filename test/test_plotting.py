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


def test_publication_defaults():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sf = SaveFigure(presentation=False)
    assert sf.styles == SaveFigure.styles
    assert sf.styles is not SaveFigure.styles
    assert sf.output_dpi == 250
    fig, _ = sf.figax()
    np.testing.assert_allclose(fig.get_size_inches() * 25.4, [170, 127.5])
    plt.close(fig)
    with plt.style.context(sf.styles["light"], after_reset=True):
        assert plt.rcParams["font.size"] == 10
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["text.usetex"]
        assert (
            mpl.colors.to_hex(plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
            == "#1f77b4"
        )


@pytest.mark.parametrize(
    "scheme,first_color", [("dark", "#4CC9F0"), ("light", "#0072B2")]
)
def test_presentation_save_and_display(scheme, first_color, tmp_path, monkeypatch):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    sf = SaveFigure(
        tmp_path, presentation=True, use_styles=[scheme], interactive_mode=scheme
    )
    before = mpl.rcParams.copy()
    calls = []
    saved = []

    def plot():
        assert mpl.rcParams["font.size"] == 48
        for key in ["axes.titlesize", "axes.labelsize", "figure.titlesize"]:
            assert mpl.rcParams[key] == 52
        for key in ["xtick.labelsize", "ytick.labelsize", "legend.fontsize"]:
            assert mpl.rcParams[key] == 48
        assert mpl.rcParams["lines.linewidth"] == 3.5
        assert mpl.rcParams["lines.markersize"] == 10
        assert mpl.rcParams["axes.linewidth"] == 1.5
        assert mpl.rcParams["font.family"] == ["sans-serif"]
        assert mpl.rcParams["text.usetex"]
        assert "lmodern" in mpl.rcParams["text.latex.preamble"]
        assert not mpl.rcParams["axes.spines.top"]
        assert not mpl.rcParams["axes.spines.right"]
        assert not mpl.rcParams["axes.grid"]
        assert not mpl.rcParams["legend.frameon"]
        assert mpl.rcParams["xtick.direction"] == "out"
        fig, ax = sf.figax()
        (line,) = ax.plot([0, 1], [0, 1])
        assert line.get_color() == first_color
        assert mpl.colors.to_hex(mpl.colors.to_rgba("contrast")) == (
            "#ffffff" if scheme == "dark" else "#000000"
        )
        ax.set_xlabel(r"Time $t$")
        ax.grid(True)
        calls.append(fig)

    # Inspect both paths without drawing LaTeX or opening a GUI.
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kw: saved.append(path))
    monkeypatch.setattr(sf, "_display_plot", lambda fig: None)
    sf(plot, "example")
    assert len(calls) == 2
    assert saved == [tmp_path / f"example-{scheme}.png"]
    assert dict(mpl.rcParams) == dict(before)
    assert all(not plt.fignum_exists(fig.number) for fig in calls)


@pytest.mark.parametrize(
    "rows,cols,width,height,expected",
    [
        (1, 1, None, None, [270, 202.5]),
        (2, 2, None, None, [270, 202.5]),
        (2, 1, None, None, [270, 405]),
        (1, 1, 240, None, [240, 240 * 3 / 4]),
        (1, 1, None, 100, [270, 100]),
        (2, 1, 200, 120, [200, 120]),
    ],
)
def test_presentation_dimensions(rows, cols, width, height, expected):
    import matplotlib.pyplot as plt

    sf = SaveFigure(presentation=True)
    for create in [sf.figax, sf.get_figure]:
        result = create(n_rows=rows, n_cols=cols, width=width, height=height)
        fig = result[0] if isinstance(result, tuple) else result
        np.testing.assert_allclose(fig.get_size_inches() * 25.4, expected)
        plt.close(fig)

    sf = SaveFigure(presentation=True, width=200)
    fig, axes = sf.figax(n_panels=3)
    assert len(axes) == 4  # Existing automatic 2 × 2 arrangement.
    np.testing.assert_allclose(fig.get_size_inches() * 25.4, [200, 200 * 3 / 4])
    plt.close(fig)


def test_presentation_styles_are_isolated():
    publication = SaveFigure()
    presentation = SaveFigure(presentation=True)
    other = SaveFigure()
    assert publication.styles == other.styles == SaveFigure.styles
    assert len(presentation.styles["dark"]) == 5
    presentation.styles["dark"].append({"axes.grid": True})
    assert len(publication.styles["dark"]) == 3
    assert len(SaveFigure(presentation=True).styles["dark"]) == 5


@pytest.mark.parametrize(
    "name", ["presentation", "presentation_dark", "presentation_light"]
)
def test_presentation_style_resources(name):
    from importlib.resources import files

    import matplotlib as mpl

    resource = files("fxutil.mplstyles").joinpath(f"{name}.mplstyle")
    assert resource.is_file()
    assert mpl.rc_params_from_file(
        str(resource), fail_on_error=True, use_default_template=False
    )


@pytest.mark.parametrize("presentation", [False, True])
@pytest.mark.parametrize(
    "defaults,overrides,expected",
    [
        ({"width": 240, "aspect_ratio": 2}, {}, [240, 120]),
        ({"width": 240, "aspect_ratio": 2}, {"width": 200}, [200, 100]),
        ({"width": 240, "height": 160, "aspect_ratio": 2}, {}, [240, 160]),
        ({"height": 160}, {"width": 200, "height": 100}, [200, 100]),
        ({"width": 240, "height": 160}, {"width": 200}, [200, 160]),
    ],
)
def test_constructor_dimensions(presentation, defaults, overrides, expected):
    import matplotlib.pyplot as plt

    sf = SaveFigure(presentation=presentation, **defaults)
    for create in [sf.figax, sf.get_figure]:
        result = create(n_rows=1, n_cols=1, **overrides)
        fig = result[0] if isinstance(result, tuple) else result
        np.testing.assert_allclose(fig.get_size_inches() * 25.4, expected)
        plt.close(fig)
    fig = sf.get_figure(2, 1)
    expected_height = defaults.get("height", sf.fig_width_mm * 2 / sf.aspect_ratio)
    np.testing.assert_allclose(fig.get_size_inches()[1] * 25.4, expected_height)
    plt.close(fig)


@pytest.mark.parametrize("name", ["width", "height", "aspect_ratio"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_constructor_dimensions(name, value):
    with pytest.raises(ValueError, match=name):
        SaveFigure(**{name: value})


def test_presentation_compact_spacing():
    import matplotlib.pyplot as plt

    sf = SaveFigure(presentation=True)
    with plt.style.context(sf.styles["light"], after_reset=True):
        assert plt.rcParams["figure.constrained_layout.w_pad"] == 0.03
        assert plt.rcParams["figure.constrained_layout.h_pad"] == 0.03
        assert plt.rcParams["legend.labelspacing"] == 0.35
        assert plt.rcParams["axes.labelpad"] == 3
