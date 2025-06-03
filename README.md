[![Tests](https://github.com/fxjung/fxutil/actions/workflows/test.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/test.yml)
[![PyPI](https://github.com/fxjung/fxutil/actions/workflows/release.yml/badge.svg)](https://github.com/fxjung/fxutil/actions/workflows/release.yml)

# f(x)util---A bunch of utilities to do things.

Scarcely documented.

## Some hints

### Getting started

Needs Python 3.8 or later.

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

Code style is black.

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

