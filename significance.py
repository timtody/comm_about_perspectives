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

EPOCH = 24999.0
hparams = ["eta_ae", "eta_lsa", "eta_msa", "eta_dsa", "sigma"]

path_perspective_msa = (
    "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/"
    "21-26-45/eta_ae:0.0-eta_lsa:0.0-eta_msa:1-eta_dsa:0.0-sigma:0.67-"
)
path_perspective_baseline = (
    "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/"
    "21-26-45/eta_ae:1-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-"
)

path_no_perspective_msa = (
    "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-21/13-39-24/"
    "eta_ae:0.0-eta_lsa:0.0-eta_msa:1.0-eta_dsa:0.0-sigma:0.67-"
)
path_no_perspective_base = (
    "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-21/13-39-24/"
    "eta_ae:1.0-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-"
)

# def series_to_mean(df, threshold=4000):
#     groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
#     return groups.apply(lambda x: x[x["Step"] >= threshold].mean())


def plot_over_noise(set_params, ax=None, title="Title"):
    if ax is not None:
        # ax.set_title(title)
        ax.axhline(0.8781, color="k", linestyle="--")
    res_path = "results/2021-05-20/21-26-45"
    path_candidates = sorted(Path(res_path).glob("*"))
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
            df["sigma"] = params["sigma"]
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
    return df


# def load_data_raw(path):
#     df, params = load_df_and_params(
#         path,
#         "pred_from_latent",
#         ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
#     )
#     df = df[(df["Metric"] == "Accuracy")]
#     df = series_to_mean(df, threshold=0)
#     for param, value in params.items():
#         df[param] = value
#     dfs.append(df)
#     return pd.concat(dfs)


def series_to_mean(df, threshold=4000):
    groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
    return groups.apply(lambda x: x[x["Step"] >= threshold].mean())


def prepare_df(path: str):
    df = load_df_and_params(
        path,
        "pred_from_latent",
        ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
    )

    df = df[
        (df["Agent"] != "baseline_1")
        & (df["Agent"] != "baseline_2")
        & (df["Agent"] != "baseline")
        & (df["Epoch"] == EPOCH)
        & (df["Metric"] == "Accuracy")
        & (df["Type"] == "Latent")
    ]
    return series_to_mean(df)


hatches = itertools.cycle([r"\\", "/"])


def change_width(ax, new_value, nclasses=2):
    hatch = next(hatches)
    for i, patch in enumerate(ax.patches):
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)
        if i % nclasses == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


if __name__ == "__main__":
    from statannot import add_stat_annotation

    width = 397.48499

    fig_w, fig_h = set_size(width, subplots=(2, 2), fraction=0.98)

    # print(matplotlib.get_data_psath())
    # matplotlib.style.use(
    #     matplotlib.get_data_path() + "/stylelib/apa.mplstyle"
    # )  # selecting
    # sns.set_theme(context="paper", style="white")
    sns.set_palette(sns.color_palette("Set1"))

    df_p_msa = prepare_df(path_perspective_msa)
    df_p_base = prepare_df(path_perspective_baseline)

    df_p_msa["Perspective"] = "Yes"
    df_p_msa["Agent"] = "Multi"
    df_p_base["Perspective"] = "Yes"
    df_p_base["Agent"] = "Single"

    df_nop_msa = prepare_df(path_no_perspective_msa)
    df_nop_base = prepare_df(path_no_perspective_base)

    df_nop_msa["Perspective"] = "No"
    df_nop_msa["Agent"] = "Multi"
    df_nop_base["Perspective"] = "No"
    df_nop_base["Agent"] = "Single"

    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    plt.rcParams.update(tex_fonts)

    G = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(G[:, 0])
    ax2 = fig.add_subplot(G[0, 1])
    ax3 = fig.add_subplot(G[1, 1])

    # ax.bar(["Yes", "No"], )

    df_noise = plot_over_noise(
        {"eta_ae": "0.0", "eta_msa": "1", "eta_lsa": "0.0", "eta_dsa": "0.0"},
        title="Plot 1",
        ax=ax2,
    )
    df_noise_2 = plot_over_noise(
        {"eta_ae": "0.53", "eta_msa": "0.74", "eta_lsa": "0.01", "eta_dsa": "0.84"},
        title="Plot 2",
        ax=ax3,
    )
    sns.lineplot(
        data=df_noise,
        x="sigma",
        y="Value",
        ax=ax2,
        err_style="bars",
        markers=True,
        dashes=False,
        style="Title",
        err_kws=dict(
            capsize=3,
            capthick=2,
        ),
    )
    sns.lineplot(
        data=df_noise_2,
        x="sigma",
        y="Value",
        ax=ax3,
        err_style="bars",
        markers=True,
        dashes=False,
        style="Title",
        err_kws=dict(
            capsize=3,
            capthick=2,
        ),
    )

    df = pd.concat([df_p_msa, df_p_base, df_nop_msa, df_nop_base])
    sns.barplot(
        data=df,
        x="Perspective",
        y="Value",
        hue="Agent",
        ci="sd",
        edgecolor=".2",
        capsize=0.01,
        errwidth=2.5,
        ax=ax1,
    )
    change_width(ax1, 0.35)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)

    ax1.legend(loc="lower left")

    ax1.set_ylim((0.5, 1))

    add_stat_annotation(
        ax1,
        data=df,
        x="Perspective",
        y="Value",
        hue="Agent",
        test="Mann-Whitney",
        box_pairs=[
            (("Yes", "Multi"), ("Yes", "Single")),
            (("No", "Multi"), ("No", "Single")),
        ],
    )

    fig.savefig("example_1.pdf", format="pdf", bbox_inches="tight")

    # print(df_p_msa.Value, df_p_base.Value)
    # print(stats.ttest_ind(df_p_msa.Value, df_p_base.Value, equal_var=False))
    # plt.show()
