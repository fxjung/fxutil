[![Tests](https://github.com/fxjung/fxutil/actions/workflows/test.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/test.yml)
[![PyPI release](https://github.com/fxjung/fxutil/actions/workflows/release.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/release.yml)

# f(x)util---A bunch of utilities to do things.

Scarcely [documented](https://fjung.com/fxutil).

## Some hints

### Getting started

Needs Python 3.10 or later.

```bash
pip install fxutil
```

### Contributing

Happy to take pull requests!

```bash
git clone git@github.com:fxjung/fxutil.git
cd fxutil
pip install -e .
pre-commit install
pytest
```

Code style is ruff.

### CLI commands

In general, try `fxutil <cmd> --help` to get more help.

- `fxutil manuscript package` -- Package a scientfic manuscript into a
  journal-digestible zip file

### `SaveFigure`

Use like:

```python
from fxutil.imports.general import *

sf = SaveFigure()


def draw_plot():
    fig, ax = figax()
    ax.plot(*evf(np.r_[-1:1:100j], lambda x: x**2))


sf(draw_plot, "my cute figure")
```


### Releasing a new version

```bash
git push
gh release create v<version> --generate-notes
```
### Presentation figures

Enable presentation mode for readable figures in conference talks, lectures, and
PhD defenses:

```python
import numpy as np

from fxutil.plotting import SaveFigure

sf = SaveFigure(
    presentation=True,
    use_styles=["dark"],
    interactive_mode="dark",
    filetypes=["png", "pdf"],
)


def plot():
    fig, ax = sf.figax()

    x = np.linspace(0, 10, 200)

    ax.plot(x, np.sin(x), label="Simulation")
    ax.plot(x, np.cos(x), label="Theory")

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Response $R(t)$")
    ax.legend()


sf(plot, "example")
```

Presentation mode uses 26 pt text, 30 pt titles and axis labels, 9 pt markers,
3.2 pt lines, outward ticks, and restrained high-contrast palettes. Axes and major
ticks use a moderate 1.6 pt stroke; minor ticks remain lighter. Neutral axes and
high-contrast text give colored data visual emphasis. Titles align to the left of
each panel, with compact legends and a little breathing room around labels.
Top and right spines, legend frames, and grids are hidden; enable a subtle grid
behind the data with `ax.grid(True)`. Publication defaults remain unchanged
unless you enable this mode.

A single panel defaults to 270 × 202.5 mm (4:3) at the existing 250 dpi output resolution.
Automatic height scales with the row/column arrangement for multiple panels.
Override dimensions in millimetres with `sf.figax(width=240, height=135)` or
`sf.get_figure(1, 1, width=240, height=135)`. Set defaults on the constructor, for example
`SaveFigure(presentation=True, width=240, aspect_ratio=4 / 3)` or
`SaveFigure(presentation=True, width=240, height=180)`.
`aspect_ratio` means single-panel width divided by height; automatic height is
`width / aspect_ratio * n_rows / n_cols`. A constructor `height` fixes the total
height for any panel arrangement and takes precedence over `aspect_ratio`.
Per-figure `width` and `height` override their respective constructor defaults.
Presentation mode also tightens layout padding and legend spacing to use the
available space more densely while keeping the large text.

Use `use_styles=["light"]` and `interactive_mode="light"` for a light background;
the two options independently select saved and displayed styles. The existing
transparent output default is retained: place dark figures on dark slides, or
set `output_transparency=False` to save the background color.

Presentation text uses Matplotlib's native renderer with Atkinson Hyperlegible
Next when installed, falling back to DejaVu Sans, and Computer Modern mathtext for
`$...$` mathematical notation. No additional font installation is required.
For full LaTeX expressions, put this inside your plotting callback, before
creating the figure:

```python
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.sans-serif": ["Computer Modern Sans Serif"],
})
```

This uses Latin Modern Sans through the retained `lmodern` LaTeX preamble and
requires a working LaTeX installation. Font and renderer overrides inside the
callback stay within SaveFigure's style context. Publication mode continues to
use its existing LaTeX settings.

The defaults balance readable text and clear curves with enough room for the
data, using compact legend and label spacing. Presentation exports use a tight
bounding box with a small safety margin so axis labels and outside legends are
included. The nominal canvas remains 4:3; the exported bounds can differ slightly
as they fit the plot's labels. Publication exports retain their existing behavior.

Check readability at the final size on your slide. Shrinking a figure also shrinks
its text and lines; for a smaller slide region, use a smaller source canvas
(e.g. `SaveFigure(presentation=True, width=180)`) so text occupies a larger fraction
of the figure, rather than choosing oversized fonts for every plot.

To render the visual review gallery (single- and four-panel examples, including
error bars, uncertainty bands, log axes, markers, and grids):

```bash
MPLBACKEND=Agg python test/test_presentation_rendering.py /tmp/fxutil-preview
```

It writes light and dark PNG/PDF previews; dark previews include a black slide
background for inspection. Actual figure exports retain their transparency setting.
