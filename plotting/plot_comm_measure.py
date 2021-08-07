import itertools
from numpy import mat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path, PosixPath
from typing import List, Tuple
from pandas import DataFrame
from chunked_writer import TidyReader
import matplotlib
import os
from matplotlib.gridspec import GridSpec

from scipy import stats


path: str = ""