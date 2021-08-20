import os
from pathlib import PosixPath, Path
from reader.chunked_writer import TidyReader
from typing import List
import pandas as pd


def series_to_mean(df, threshold=4800, add_params=()):
    df = df[df["Step"] > threshold]
    groups = df.groupby(
        ["Rank", "Metric", "Type", "Agent", "Epoch", *add_params], as_index=False
    ).mean()
    return groups


def stem_to_params(stem: str):
    """
    Helper function which takes in a path stem of the form "param:value-param2:value2..."
    and returns a dictionary of parameters, e.g. "{"param":value,...}
    Args:
        stem: String. The name of the folder.
    """
    params = {k: v for d in map(eval_d, stem.split("-")[:-1]) for k, v in d.items()}
    return params


def eval_d(string: str):
    """
    Helper function which parses a string of the form 'x:y' into
    a dictionary {"x": "y"}. This is used for inferring the hyperparameters
    from the folder names.
    Args:
        string: String. The input string to evaluate.
    """
    k, v = string.split(":")
    return {k: v}


def load_df_and_params(
    posixpath: PosixPath, tag: str, columns: List[str], datafolder="data"
):
    """
    Args:
        posixpath: Posixpath. Path to one specific epxeriment
        tag: String. The name of the metric which we want to retrieve
        columns: List[String]. The column headers of the resulting dataframe
    Returns:
        df: DataFrame
    """
    reader = TidyReader(os.path.join(posixpath, datafolder))
    df = reader.read(tag=tag, columns=columns)
    params = stem_to_params(posixpath.name)
    return df, params


def load_data(path, tag, keys, stop_after=None):
    paths = list(Path(path).glob("*"))
    dfs = []
    i = 0
    for path in paths:
        i += 1
        df, params = load_df_and_params(path, tag, keys)
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
        if stop_after is not None:
            if i > stop_after:
                break

    return pd.concat(dfs)


def plot_over(set_params, by=None, ax=None, title="Title", path=""):
    path_candidates = sorted(Path(path).glob("*"))
    dfs = []
    for path in path_candidates:
        suffix = str(path).split("/")[-1]
        params = stem_to_params(suffix)
        passing = False
        for p, v in set_params.items():
            if params[p] != v:
                passing = True
        if not passing:
            reader = TidyReader(str(path) + "/data")
            df = reader.read(
                "pred_from_latent",
                ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
            )
            df[by] = params[by]
            dfs.append(df)
    df = pd.concat(dfs)
    df = df[
        (df["Epoch"] == 39999)
        & (df["Metric"] == "Accuracy")
        & (df["Type"] == "Latent")
        & (df["Agent"] != "baseline_1")
        & (df["Agent"] != "baseline_2")
    ]
    df["Title"] = title
    return df
