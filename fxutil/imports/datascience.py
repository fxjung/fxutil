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

plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"""
\usepackage{nicefrac}
\usepackage{commath}
\usepackage{amsfonts}
\usepackage{lmodern}
%\usepackage{cmbright}
""",
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 10,
        "font.family": "lmodern",
        # 'font.sans-serif': 'sans-serif',
    }
)

plt.rc("pdf", fonttype=42)

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

from fxutil.plotting import evf, set_plot_dark
