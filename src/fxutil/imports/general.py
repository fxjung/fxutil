# ruff: noqa

import dataclasses
import h5py
import scipy
import copy
import random
import json

import itertools as it
import functools as ft
import operator as op

import math as m
import numpy as np

import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from IPython.display import display
from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from cycler import cycler

from fxutil.meta import in_ipython_session

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

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
    minmax,
    mmr,
)

from fxutil.typing import Combi, parse_combi_args

# ruff:enable

if in_ipython_session():

    @register_cell_magic
    def s(line, cell):
        cls = SaveFigure
        ip = get_ipython()
        ns = ip.user_ns
        sfs = [v for v in ns.values() if isinstance(v, cls)]
        if (n_sfs := len(sfs)) > 1:
            raise RuntimeError("Must have no more than one SaveFigure instance")
        elif n_sfs == 1:
            sf = sfs[0]
        elif n_sfs == 0:
            # print("Creating SaveFigure instance")
            sf = SaveFigure()
            ns["sf"] = sf
        sf(lambda globals_=ns: exec(cell, globals_), name=line)
