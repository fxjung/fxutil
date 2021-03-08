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
plt.rc("text.latex", preamble=r"\usepackage{nicefrac}\usepackage{commath}")
