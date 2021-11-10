import math as m

from pympler.asizeof import asizeof

from fxutil.plotting import SaveFigure, evf, easy_prop_cycle

fmt_bytes = lambda s: (
    lambda s, k: (s * 2 ** (-k * 10), ["B", "KiB", "MiB", "GiB", "TiB", "PiB"][k])
)(s, round(m.log(s) / m.log(2) / 10))

described_size = lambda desc, obj: (
    lambda desc, fmtd: f"{desc}{fmtd[0]:.2f} {fmtd[1]}"
)(desc, fmt_bytes(asizeof(obj)))
"""
Use like
```py
>>> print(described_size("my huge-ass object's size: ", my_huge_ass_object))
my huge-ass object's size: 5.32 TiB
```

Parameters
----------
desc : str
    string to print along 
obj : object
    thing to get the size of 
    
Returns
-------
str
"""


def scinum(a, force_pref=False) -> str:
    """
    Return LaTeX-formatted string representation of number in scientific notation.

    Parameters
    ----------
    a
        number to format
    force_pref
        force prepending sign prefix

    Returns
    -------

    """
    s = fr"{'' if not force_pref else ('+' if a>= 0 else '')}"
    e = m.floor(m.log10(abs(a)))
    if abs(e) > 3:
        s += fr"{a*10**(-e):.2f}\times 10^{{{e}}}"
    else:
        s += fr"{a:.2f}"
    s += "\,"
    return s
