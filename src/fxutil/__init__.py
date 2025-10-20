import logging

from IPython import get_ipython

from fxutil.common import (
    bunny,
    described_size,
    fmt_bytes,
    get_git_repo_path,
    get_unique_with_bang,
    nixt,
    round_by_method,
    scinum,
    thing,
)
from fxutil.plotting import SaveFigure, easy_prop_cycle, evf, figax, pad_range

from .meta import in_ipython_session
from .typing import Combi, parse_combi_args


__all__ = [
    "in_ipython_session",
    "SaveFigure",
    "evf",
    "figax",
    "pad_range",
    "easy_prop_cycle",
    "fmt_bytes",
    "described_size",
    "get_git_repo_path",
    "round_by_method",
    "scinum",
    "nixt",
    "thing",
    "get_unique_with_bang",
    "bunny",
    "Combi",
    "parse_combi_args",
]

logging.getLogger("").setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logging.getLogger("").addHandler(ch)


if in_ipython_session():
    exec("from fxutil.imports.general import *", get_ipython().user_ns)
