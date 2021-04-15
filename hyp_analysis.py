from ast import Str
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from chunked_writer import TidyReader


def load_df_and_params(posixpath, tag, columns):
    """
    Args:
        posixpath: Posixpath. Path to one specific epxeriment
        tag: String. The name of the metric which we want to retrieve
        columns: List[String]. The column headers of the resulting dataframe
    Returns:
        df: DataFrame
    """
    reader = TidyReader(os.path.join(posixpath, "data"))
    df = reader.read(tag=tag, columns=columns)
    params = stem_to_params(posixpath.name)
    return df, params


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


def stem_to_params(stem) -> dict:
    """
    Helper function which takes in a path stem of the form "param:value-param2:value2..."
    and returns a dictionary of parameters, e.g. "{"param":value,...}
    Args:
        stem: String. The name of the folder.
    """
    params = {k: v for d in map(eval_d, stem.split("-")[:-1]) for k, v in d.items()}
    return params


def series_to_mean(df):
    groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
    return groups.apply(lambda x: x[x["Step"] >= 4000].mean())


def load_data(path, type="Reconstruction"):
    paths = Path(path).glob("*")
    dfs = []
    i = 0
    for path in paths:
        i += 1
        df, params = load_df_and_params(
            path,
            "pred_from_latent",
            ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
        )
        df = df[(df["Metric"] == "Accuracy") & (df["Type"] == type)]
        df = series_to_mean(df)
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
    return pd.concat(dfs)


def get_best_params(df, tolerance=0.00, hparams=[]):
    # compute the mean across ranks
    groups = df.groupby([*hparams, "Agent", "Epoch"], as_index=False).mean()
    # filter groups with lower final performance
    res = groups.groupby([*hparams, "Epoch"]).filter(
        lambda x: x[x["Agent"] == "baseline"].Value
        < x[x["Agent"] != "baseline"].Value.max() + tolerance
    )
    return res


def compute_reg_coefs(X, y):
    reg = LinearRegression().fit(X, y)
    return reg.coef_


def load_data_raw(path):
    paths = Path(path).glob("*")
    dfs = []
    i = 0
    for path in paths:
        i += 1
        df, params = load_df_and_params(
            path,
            "pred_from_latent",
            ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
        )
        df = df[(df["Metric"] == "Accuracy")]
        df = series_to_mean(df)
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
        if i == -1:
            break
    return pd.concat(dfs)


def compute_and_save_reg_coefs(df, hparams, tag):
    # compute impact of hparams on prediction

    ## filter out baseline because most parameters have no influence on it
    ## only look at last epoch
    df = df[(df["Agent"] != "baseline") & (df["Epoch"] == 49999.0)]

    ## compute the mean across ranks and agents to arrive at 1 acc. value per set of hparams
    groups = df.groupby(hparams, as_index=False).mean()

    coefs = compute_reg_coefs(groups.loc[:, hparams], groups.loc[:, "Value"])
    columns = list(map(lambda x: f"beta_{x}", hparams))
    df_coefs = pd.DataFrame((coefs, coefs), columns=columns)
    df_coefs.to_csv(f"plots/{'-'.join(hparams)}_params_{tag}.csv")


def compute_barplots(df, hparams, tag):
    df = get_best_params(df, 0.0, hparams)
    df = df[df["Epoch"] == 49999.0]
    param_col_name = ""
    for param in hparams:
        param_col_name += param + "=" + df[param] + " "

    df["params"] = param_col_name
    if len(df) > 0:
        g = sns.catplot(
            data=df, kind="bar", col="params", x="Agent", y="Value", col_wrap=3
        )
        g.set_titles("{col_name}")
        plt.savefig(f"plots/bar_{tag}.pdf")
    else:
        print("Arsch")


def compute_pcoords(df, hparams):
    pass


def compute_best_vs_base(df, hparams, tag):
    df = get_best_params(df, 1.0, hparams)
    trans = df.groupby([*hparams, "Epoch"], as_index=False).apply(
        lambda x: x.assign(
            max_diff=x[x.Agent != "baseline"].Value.max()
            - x[x.Agent == "baseline"].Value,
        ),
    )
    trans = trans[trans.Agent == "baseline"]
    trans = trans[trans.Epoch == 49999.0]
    trans = trans.loc[
        :, ["eta_ae", "eta_lsa", "eta_dsa", "eta_msa", "sigma", "Epoch", "max_diff"]
    ].sort_values(by="max_diff", axis=0, ascending=False)

    plt.bar(["Best", "Mean"], [trans.max_diff.iloc[0], trans.max_diff.mean()])
    plt.annotate(str(round(trans.max_diff.iloc[0], 2)), (0, trans.max_diff.iloc[0]))
    plt.annotate(str(round(trans.max_diff.mean(), 2)), (1, trans.max_diff.mean()))
    plt.show()

    exit(1)


def compute_plots_rec(df, hparams):
    # reconstruction
    tag = "Reconstruction"
    df_rec = df[df["Type"] == tag]

    ## reg coefs
    compute_and_save_reg_coefs(df_rec, hparams, tag)
    ## barplots
    compute_barplots(df_rec, hparams, tag)
    ## diff between best agent and baseline
    compute_best_vs_base(df_rec, hparams, tag)
    ## pcoords
    # compute_pcoords(df_rec, hparams, tag)


def compute_plots_latent(df, hparams):
    # latent
    tag = "Latent"
    df_lat = df[df["Type"] == tag]
    compute_best_vs_base(df_lat, hparams, tag)

    ## reg coefs
    compute_and_save_reg_coefs(df_lat, hparams, tag)
    ## barplots
    compute_barplots(df_lat, hparams, tag)
    ## diff between best agent and baseline
    compute_best_vs_base(df_lat, hparams, tag)
    ## pcoords
    # compute_pcoords(df_rec, hparams, tag)


def make_plots(path: Str, hparams: List):
    df = load_data_raw(path)

    compute_plots_latent(df, hparams)
    compute_plots_rec(df, hparams)

    # t-sne in latent space

    # reconstruction from good marl agents vs. baseline agents for some digits

    # covariance matric between hparams and losses (final?)


if __name__ == "__main__":
    make_plots(
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-12/21-04-14",
        ["eta_ae", "eta_lsa", "eta_dsa", "eta_msa", "sigma"],
    )

    # gt_than_base = get_best_params(
    #     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-12/21-04-14",
    #     "Latent",
    #     hparams=["eta_ae", "eta_lsa", "eta_dsa", "eta_msa", "sigma"],
    #     tolerance=0.00,
    # )
    # print(gt_than_base)
