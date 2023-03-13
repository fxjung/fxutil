import os
import pygit2

import math as m

from pathlib import Path
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


def scinum(
    a,
    force_pref: bool = False,
    round_method: str = "round",
    ndigits: int = 2,
    force_mode: str | None = None,
) -> str:
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
    force_mode
        'e', 'f'

    Returns
    -------

    """

    if a == 0:
        s = "0"
        if ndigits > 0:
            s += "."
            s += "0" * ndigits
        return s

    if m.isinf(a):
        s = "\infty"
        if a > 0 and force_pref:
            s = f"+{s}"
        elif a < 0:
            s = f"-{s}"
        return s

    s = rf"{'' if not force_pref else ('+' if a>= 0 else '')}"
    e = m.floor(m.log10(abs(a)))

    if (abs(e) > 2 or force_mode == "e") and not force_mode == "f":
        m_ = round_by_method(a * 10 ** (-e), ndigits=ndigits, round_method=round_method)
        s += rf"{m_:.{ndigits}f}\times 10^{{{e}}}"
    else:
        s += rf"{round_by_method(a, ndigits=ndigits, round_method=round_method):.{ndigits}f}"
    s += "\,"  # TODO: should this always be appended??
    return s


def get_git_repo_path():
    """
    Returns the path to the root of the inner most git repository that the
    working directory resides in, if any. Raises if not contained in any
    repository.

    Returns
    -------
    repository_path: Path
    """

    working_dir = os.getcwd()
    repository_path = pygit2.discover_repository(working_dir)
    if repository_path is None:
        raise ValueError(f"{working_dir} is not part of a git repository")
    else:
        return Path(repository_path).parent
