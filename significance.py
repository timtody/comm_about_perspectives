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
from utils import stem_to_params, load_df_and_params
from plotting_helpers import set_size, set_tex_fonts, set_palette

from scipy import stats

EPOCH = 19999.0
hparams = ["eta_ae", "eta_lsa", "eta_msa", "eta_dsa", "sigma"]

# path_perspective_msa = (
#     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/"
#     "21-26-45/eta_ae:0.0-eta_lsa:0.0-eta_msa:1-eta_dsa:0.0-sigma:0.67-"
# )
# path_perspective_baseline = (
#     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/"
#     "21-26-45/eta_ae:1-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-"
# )
#
# path_no_perspective_msa = (
#     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-21/13-39-24/"
#     "eta_ae:0.0-eta_lsa:0.0-eta_msa:1.0-eta_dsa:0.0-sigma:0.67-"
# )
# path_no_perspective_base = (
#     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-21/13-39-24/"
#     "eta_ae:1.0-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-"
# )

path_perspective_msa = (
    "results/gridsweep/"
    "sigma:0.0-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.67-"
)
path_perspective_baseline = (
    "results/gridsweep/"
    "sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
)

path_perspective_base_lsa = (
    "results/gridsweep/"
    "sigma:1.0-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
)

path_no_perspective_msa = (
    "results/gridsweep/"
    "sigma:0.0-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.67-"
)
path_no_perspective_base = (
    "results/gridsweep/"
    "sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
)

path_no_perspective_base_lsa = (
    "results/gridsweep/"
    "sigma:1.0-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
)


# def series_to_mean(df, threshold=4000):
#     groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
#     return groups.apply(lambda x: x[x["Step"] >= threshold].mean())


def plot_over(set_params, by=None, ax=None, title="Title", path=""):
    path_candidates = sorted(Path(path).glob("*"))
    dfs = []
    for path in path_candidates:
        print(path)
        suffix = str(path).split("/")[-1]
        print(suffix)
        params = stem_to_params(suffix)
        print(params)
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
        (df["Epoch"] == EPOCH)
        & (df["Metric"] == "Accuracy")
        & (df["Type"] == "Latent")
        & (df["Agent"] != "baseline_1")
        & (df["Agent"] != "baseline_2")
    ]
    df["Title"] = title
    return df


def plot_over_noise(set_params, ax=None, title="Title"):
    if ax is not None:
        pass
        # ax.set_title(title)
        # ax.axhline(0.8781, color="k", linestyle="--")
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


def series_to_mean(df, threshold=4800, add_params=[]):
    groups = df.groupby(
        ["Rank", "Metric", "Type", "Agent", "Epoch", *add_params], as_index=False
    )
    return groups.apply(lambda x: x[x["Step"] >= threshold])


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


hatches = itertools.cycle(["/", r"\\", "X"])
hatches_list = [["/", r"\\", "X"]]


def change_width(ax, new_value, nclasses=2):
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



if __name__ == "__main__":
    from statannot import add_stat_annotation

    fig_w, fig_h = set_size("neurips", subplots=(1, 2), fraction=0.98)
    set_tex_fonts()
    set_palette()

    df_p_msa = prepare_df(path_perspective_msa)
    df_p_base = prepare_df(path_perspective_baseline)
    df_p_lsa = prepare_df(path_perspective_base_lsa)

    df_p_msa["Perspective"] = "Yes"
    df_p_msa["Agent"] = "MTI"
    df_p_lsa["Perspective"] = "Yes"
    df_p_lsa["Agent"] = "AE+MTM"
    df_p_base["Perspective"] = "Yes"
    df_p_base["Agent"] = "AE"

    df_nop_msa = prepare_df(path_no_perspective_msa)
    df_nop_base = prepare_df(path_no_perspective_base)
    df_nop_lsa = prepare_df(path_no_perspective_base_lsa)

    df_nop_msa["Perspective"] = "No"
    df_nop_msa["Agent"] = "MTI"
    df_nop_base["Perspective"] = "No"
    df_nop_base["Agent"] = "AE"
    df_nop_lsa["Perspective"] = "No"
    df_nop_lsa["Agent"] = "AE+MTM"


    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))

    G = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(G[0, 0])
    ax2 = fig.add_subplot(G[0, 1])
    # ax3 = fig.add_subplot(G[1, 1])

    # ax.bar(["Yes", "No"], )

    # df_noise = plot_over_noise(
    #     {"eta_ae": "0.0", "eta_msa": "1", "eta_lsa": "0.0", "eta_dsa": "0.0"},
    #     title="Plot 1",
    #     ax=ax2,
    # )

    # print(df_noise)
    # df_noise = series_to_mean(df_noise, add_params=["sigma", "Title"])
    # print(df_noise)

    df_nagents = plot_over(
        {"eta_lsa": "0.3", },
        by="nagents",
        path="results/nagents",
    )
    print(df_nagents)
    df_nagents["nagents"] = df_nagents["nagents"].map(str)
    print(df_nagents)
    df_nagents = series_to_mean(df_nagents, add_params=["nagents", "Title"])
    print(df_nagents)

    # lineax = sns.lineplot(
    #     data=df_noise,
    #     x="sigma",
    #     y="Value",
    #     ax=ax2,
    #     err_style="bars",
    #     markers=True,
    #     dashes=False,
    #     style="Title",
    #     legend=False,
    #     err_kws=dict(
    #         capsize=3,
    #         capthick=2,
    #     ),
    # )
    # lineax.set_xlabel(r"Noise level ($\sigma$)")
    # lineax.set_ylabel("Accuracy")

    sns.lineplot(
        data=df_nagents,
        x="nagents",
        y="Value",
        ax=ax2,
        err_style="bars",
        markers=True,
        dashes=False,
        style="Title",
        legend=False,
        err_kws=dict(
            capsize=3,
            capthick=2,
        ),
    )

    df = pd.concat([df_p_msa, df_p_base, df_p_lsa, df_nop_msa, df_nop_base, df_nop_lsa])
    sns.barplot(
        data=df,
        x="Perspective",
        y="Value",
        hue="Agent",
        ci="sd",
        edgecolor=".2",
        capsize=0.01,
        errwidth=1.5,
        ax=ax1,
    )
    ax1.set_ylabel(r"Accuracy (\%)")
    ax2.set_ylabel(r"Accuracy (\%)")
    ax2.set_xlabel("N agents")

    import matplotlib.patches as mpatches


    handles, labels = ax1.get_legend_handles_labels()
    print(handles, labels)
    for hatch, handle in zip(hatches_list, handles):
        print(handle)
        print(dir(handle))
        handle.hatch = hatch

    ax1.legend(handles=handles, labels=labels, title="")
    # ax3.set_ylabel(r"Accuracy (\%)")
    # ax3.set_xlabel("N agents")
    change_width(ax1, 0.2)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    # sns.despine(ax=ax3)

    #ax1.legend(loc="lower left")

    ax2.set_ylim((0.89, 0.91))

    add_stat_annotation(
        ax1,
        data=df,
        x="Perspective",
        y="Value",
        hue="Agent",
        test="Mann-Whitney",
        box_pairs=[
            (("Yes", "AE+MTM"), ("No", "AE+MTM")),
            (("Yes", "AE"), ("Yes", "AE+MTM"))
            #(("No", "Multi"), ("No", "Single")),
        ],
    )

    fig.savefig("example_1_decentralised.pdf", format="pdf", bbox_inches="tight")

    # print(df_p_msa.Value, df_p_base.Value)
    # print(stats.ttest_ind(df_p_msa.Value, df_p_base.Value, equal_var=False))
    # plt.show()
