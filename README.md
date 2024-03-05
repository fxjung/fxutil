# f(x)util---A bunch of utilities to do things.


## Some hints

### CLI commands

In general, try `fxutil <cmd> --help` to get more help.

- `fxutil manuscript package` -- Package a scientfic manuscript into a
  journal-digestible zip file

### `SaveFigure`

Use like:

```python
from fxutil.imports.datascience import *

sf = SaveFigure()

def draw_plot():
    fig, ax = figax()
    ax.plot(*evf(np.r_[-1:1:100j], lambda x: x**2))

sf(draw_plot, "my cute figure")
```
