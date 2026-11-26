import math as m
import os
from pathlib import Path
from typing import Iterable, Optional

import operator as op
import numpy as np
import pandas as pd
import pygit2
from pympler.asizeof import asizeof


def fmt_bytes(s):
    """Return a (value, unit) pair for a positive byte count.

    Choose the nearest power of 1024 on a logarithmic scale, using units B
    through PiB. For example, 512 bytes yields ``(0.5, "KiB")``. Zero and
    counts beyond the supported unit indices raise an error.
    """
    return (
        lambda s, k: (s * 2 ** (-k * 10), ["B", "KiB", "MiB", "GiB", "TiB", "PiB"][k])
    )(s, round(m.log(s) / m.log(2) / 10))


def described_size(desc, obj):
    """Estimate the in-memory size of an object with Pympler's ``asizeof``.

    Return the supplied prefix followed by the size to two decimal places and
    a binary unit. Include any desired space or punctuation in ``desc``.
    """
    return (lambda desc, fmtd: f"{desc}{fmtd[0]:.2f} {fmtd[1]}")(
        desc, fmt_bytes(asizeof(obj))
    )


def round_by_method(x, ndigits, round_method: str = "round"):
    """Round a number to ``ndigits`` decimal places.

    Choose Python's ``"round"``, or ``"floor"`` / ``"ceil"`` for rounding
    toward negative / positive infinity. Negative precisions round to tens,
    hundreds, and so on. An unknown method raises ``ValueError``.
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
    no_trailing_zeros=True,
    force_mode: Optional[str] = None,
    suffix=r"\,",
    thousands_sep=r"\,",
    inf_lim=None,
) -> str:
    r"""Format a real number as a LaTeX math fragment, without dollar signs.

    Let *e* be the floor of the base-10 logarithm of the absolute value.
    Scientific notation is used when ``abs(e) > 2``. Set ``force_mode="e"``
    for scientific notation or ``"f"`` for fixed point. The exponent is
    chosen before rounding.

    ``ndigits`` counts decimal places in the displayed number or mantissa.
    ``round_method`` accepts the methods of :func:`round_by_method`.
    Fractional trailing zeros are stripped by default. With zero decimal
    places this also strips significant integer zeros, so use
    ``no_trailing_zeros=False`` at that precision.

    Fixed-point groups and the trailing suffix both default to the LaTeX
    thin space ``\,``. Set ``thousands_sep=""`` or ``suffix=""`` to omit
    either. ``force_pref=True`` prefixes positive values with a plus sign.
    Zero and infinity omit the suffix. A positive ``inf_lim`` formats
    magnitudes at or above the threshold as infinity. NaN is unsupported.
    """

    def strip_trailing_zeros(s):
        return s.rstrip("0").rstrip(".")

    if a == 0:
        s = "0"
        if ndigits > 0 and not no_trailing_zeros:
            s += "."
            s += "0" * ndigits
        return s

    if m.isinf(a) or (inf_lim and abs(a) >= inf_lim):
        s = r"\infty"
        if a > 0 and force_pref:
            s = rf"+{s}"
        elif a < 0:
            s = rf"-{s}"
        return s

    s = rf"{'' if not force_pref else ('+' if a >= 0 else '')}"
    e = m.floor(m.log10(abs(a)))

    if (abs(e) > 2 or force_mode == "e") and not force_mode == "f":
        m_ = round_by_method(a * 10 ** (-e), ndigits=ndigits, round_method=round_method)
        s += rf"{m_:.{ndigits}f}"
        if no_trailing_zeros:
            s = strip_trailing_zeros(s)
        s += rf"\times 10^{{{e}}}"
    else:
        s += rf"{round_by_method(a, ndigits=ndigits, round_method=round_method):_.{ndigits}f}".replace(
            "_", thousands_sep if thousands_sep else ""
        )
        if no_trailing_zeros:
            s = strip_trailing_zeros(s)
    s += suffix
    return s


def get_git_repo_path():
    """Return the parent of the Git directory discovered from the working directory.

    In an ordinary checkout this is the repository root. Linked worktrees and
    bare repositories have different Git directory layouts. Raise ``ValueError``
    if discovery finds no repository.
    """

    working_dir = os.getcwd()
    repository_path = pygit2.discover_repository(working_dir)
    if repository_path is None:
        raise ValueError(f"{working_dir} is not part of a git repository")
    else:
        return Path(repository_path).parent


def minmax(ser, min_quantile: float | None = None, max_quantile: float | None = None):
    """Return the lower and upper bounds of a nonempty sequence.

    Each optional quantile (0 to 1) replaces the corresponding extreme with
    a NumPy quantile. Supply data that can be traversed twice.
    """
    if min_quantile is None:
        _min = min(ser)
    else:
        _min = np.quantile(ser, min_quantile)

    if max_quantile is None:
        _max = max(ser)
    else:
        _max = np.quantile(ser, max_quantile)

    return _min, _max


def mmr(ser):
    """Return (minimum, maximum, maximum - minimum) for a nonempty numeric sequence."""
    min_, max_ = minmax(ser)
    return min_, max_, max_ - min_


def nixt(thing: Iterable):
    """Return the first item. An empty iterable raises ``StopIteration``."""
    return next(iter(thing))


def onixt(thing: Iterable):
    """Return the sole item. An empty iterable raises ``StopIteration``.

    Raise ``ValueError`` upon reading a second item. On an iterator, the
    items inspected are consumed.
    """
    iter_ = iter(thing)
    val = next(iter_)
    try:
        next(iter_)
    except StopIteration:
        return val
    else:
        raise ValueError("Iterable yielded more than one element.")


def thing(which: Optional[str]):
    """Build example data for the requested container.

    ``None`` or ``"dataframe"`` (aliases ``"pandas", "pd", "df", "d"``)
    gives columns a = 10…19 and b = 100…109. ``"array"`` (``"np", "numpy",
    "n"``) gives the same two columns as a NumPy array. ``"series"`` (``"ser",
    "s"``), ``"list"`` (``"l"``), ``"tuple"`` (``"t"``) and ``"set"`` use just
    the first column. ``"dict"`` maps letters k through t to 10 through 19.
    Supply the argument explicitly. Unknown names raise ``ValueError``.
    """
    tabular_data = np.c_[10:20, 100:110]
    iterable_1d = map(op.methodcaller("item"), tabular_data[:, 0])
    which = which or "dataframe"
    if which in ["dataframe", "pandas", "pd", "df", "d"]:
        return pd.DataFrame(tabular_data, columns=["a", "b"])
    elif which in ["series", "ser", "s"]:
        return pd.Series(iterable_1d, name="a")
    elif which in ["list", "l", "s"]:
        return [*iterable_1d]
    elif which in ["tuple", "t"]:
        return (*iterable_1d,)
    elif which in ["set", "s"]:
        return {*iterable_1d}
    elif which in ["dict"]:
        return {chr(97 + v): v for v in iterable_1d}
    elif which in ["array", "np", "numpy", "n"]:
        return tabular_data
    else:
        raise ValueError()


def get_unique_with_bang(iterable):
    """Return the first value if all items compare equal.

    Repeated values are allowed. Unequal values raise ``ValueError`` for
    ordinary hashable scalars. Empty input raises ``IndexError``. An iterator
    is materialized. Multi-row numeric Series and single-column numeric
    frames work directly. For a one-row Series or one containing Python
    strings, pass ``series.to_list()`` to avoid errors from pandas scalar
    handling.
    """
    if isinstance(iterable, (pd.Series, pd.DataFrame)):
        ser = iterable.squeeze()
        val = iterable.iloc[0].squeeze()
        if not all(ser == val):
            raise ValueError(
                f"Series not unique. Unique values: {', '.join(map(str, ser.unique()))}"
            )
    else:
        left = [*iterable]
        val = left[0]
        if not all(x == val for x in left):
            raise ValueError(
                f"{type(iterable).__name__} not unique. Unique values: "
                f"{', '.join(map(str, set(iterable)))}"
            )

    return val


bunny = get_unique_with_bang


def permute(iterable, perm):
    """Return a list of items from an indexable object in the order given by ``perm``.

    Indices can repeat or omit items. Indexing follows the input object's
    own rules.
    """
    return [iterable[i] for i in perm]
