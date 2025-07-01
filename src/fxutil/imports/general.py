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
from cycler import cycler

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from fxutil.plotting import SaveFigure, evf, easy_prop_cycle, figax
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
