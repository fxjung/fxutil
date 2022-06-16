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


def round_by_method(x, ndigits, round_method: str = "round"):
    """

    Parameters
    ----------
    x
    ndigits
    round_method
        One of 'round', 'floor', 'ceil'

    Returns
    -------

    """
    if round_method == "round":
        return round(x, ndigits)
    elif round_method == "floor":
        e = 10**ndigits
        return m.floor(x * e) / e
    elif round_method == "ceil":
        e = 10**ndigits
        return m.ceil(x * e) / e
    else:
        raise ValueError


def scinum(a, force_pref: bool = False, round_method: str = "round", ndigits=2) -> str:
    """
    Return LaTeX-formatted string representation of number in scientific notation.

    Parameters
    ----------
    a
        number to format
    force_pref
        force prepending sign prefix
    round_method
        One of 'round', 'floor', 'ceil'
    ndigits
        Number of decimal places

    Returns
    -------

    """
    s = rf"{'' if not force_pref else ('+' if a>= 0 else '')}"
    e = m.floor(m.log10(abs(a)))

    if abs(e) > 2:
        m_ = round_by_method(a * 10 ** (-e), ndigits=ndigits, round_method=round_method)
        s += rf"{m_:.{ndigits}f}\times 10^{{{e}}}"
    else:
        s += rf"{round_by_method(a, ndigits=ndigits, round_method=round_method):.{ndigits}f}"
    s += "\,"
    return s
