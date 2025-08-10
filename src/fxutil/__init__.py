import logging

logging.getLogger("").setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logging.getLogger("").addHandler(ch)


from fxutil.plotting import SaveFigure, evf, easy_prop_cycle, figax, pad_range
from fxutil.common import (
    fmt_bytes,
    described_size,
    get_git_repo_path,
    round_by_method,
    scinum,
    nixt,
    thing,
    get_unique_with_bang,
    bunny,
)

from .typing import Combi, parse_combi_args
