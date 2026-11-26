# fxutil

[![Tests](https://github.com/fxjung/fxutil/actions/workflows/test.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/test.yml)
[![PyPI release](https://github.com/fxjung/fxutil/actions/workflows/release.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/release.yml)

fxutil provides Matplotlib figure export for papers and slides, a LaTeX manuscript packager, and small utilities for scientific Python. It also provides an interactive namespace and a figure-saving cell magic for IPython. [Documentation](https://fjung.com/fxutil/)

## Install

Use Python 3.13 or later.

```bash
python -m pip install fxutil
```

## Save a figure

This example writes `plots/decay-curve-light.png` and `plots/decay-curve-light.pdf`. It runs headlessly and uses Matplotlib's native text renderer.

```python
import numpy as np
from fxutil import SaveFigure

t = np.linspace(0, 4, 101)
response = np.exp(-t)
sf = SaveFigure("plots", presentation=True, use_styles=["light"],
                filetypes=["png", "pdf"], interactive_mode=None)

def draw():
    fig, ax = sf.figax()
    ax.plot(t, response)
    ax.set(xlabel="Time (s)", ylabel="Response")

sf(draw, "decay curve")
```

The saver calls `draw` once per style, writes all formats from that figure, and draws again for an enabled preview. Its defaults are light and dark PNGs, a dark preview, and publication typography that requires LaTeX. Presentation mode uses a larger canvas and native mathtext. [Figure guide](https://fjung.com/fxutil/figures.html)

## Package a manuscript

```bash
fxutil manuscript package paper --no-delete-existing
```

Given a supported `paper/manuscript.tex`, the command builds `paper-packaged/` and `paper-packaged.zip` beside the source directory. It copies referenced graphics, inlines table fragments and the compiled bibliography, and rebuilds the PDF. It requires `latexmk`, a working TeX toolchain and `zip`. The flag shown skips the default deletion of previous outputs. The guide describes the literal TeX commands and layout the packager recognizes. [Manuscript guide](https://fjung.com/fxutil/manuscript.html)

## Smaller tools

- Format a LaTeX number with `scinum`, check for a common value with `bunny`, or accept either one argument value or an iterable with `Combi` and `@parse_combi_args`. [Values and collections](https://fjung.com/fxutil/utilities.html)
- Importing `fxutil` in IPython installs names such as `np`, `pd` and `plt` directly into the user namespace. Use `%%s figure name` to render a cell with a `SaveFigure` instance. [Interactive imports](https://fjung.com/fxutil/ipython.html)

## Contributing

From a source checkout, install the locked environment with uv and run the checks below. The full test suite renders publication figures and needs LaTeX with `commath`, `amsfonts` and `lmodern`. CI installs `texlive-full`.

```bash
uv sync --dev
uv run pre-commit install
MPLBACKEND=Agg uv run pytest
uv run sphinx-build -W -b html docs docs/_build/html
uv run sphinx-build -W -b doctest docs docs/_build/doctest
```

Open `docs/_build/html/index.html` for the local guide. The pre-commit hooks run Ruff's linter and formatter. Publishing a GitHub release tagged `v<version>` triggers the tests, PyPI publication and documentation deployment. [Contributor instructions](https://github.com/fxjung/fxutil/blob/master/AGENTS.md)
