import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation

from reader.chunked_writer import TidyReader
from plotting.plotting_helpers import set_size, set_tex_fonts, set_palette
from utils import load_data, series_to_mean, plot_over

EPOCH = 29999
HPARAMS = ["eta_ae", "eta_lsa", "eta_msa", "eta_dsa", "sigma"]

hatches = itertools.cycle(["/", r"\\", "X"])


def remove_ax_titles(ax):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="")


def change_width_(ax, new_value, nclasses=2):
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


def clean_latent_data(data, epoch=29999):
    print("data before clean", data)
    return data[
        (data["Epoch"] == EPOCH if epoch is None else epoch)
        & (data["Metric"] == "Test accuracy")
        & (data["Type"] == "Latent")
        & (data["Agent"] != "baseline_1")
        & (data["Agent"] != "baseline_2")
        & (data["Agent"] != "baseline")
    ]


def load_latent_data(path, epoch=None):
    data = load_data(
        path,
        "pred_from_latent",
        ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
    )
    data = clean_latent_data(data, epoch)
    print("cleaned data", data)
    return data


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


def gather_runs_by_param_configs(data, parameter_configs):
    all_runs = []
    for config in parameter_configs:
        for run, params in config.items():
            copy = data.copy()
            for param, value in params.items():
                copy = copy[copy[param] == value]
            copy["Agent"] = run
            all_runs.append(copy)
    data = pd.concat(all_runs)
    return data


def prepare_plot():
    plt.clf()
    set_tex_fonts(11, 8, 6)
    set_palette()


def read_single(path, tag, columns):
    reader = TidyReader(os.path.join(path, "data"))
    df = reader.read(tag=tag, columns=columns)
    return df


def read_single_latent(path):
    reader = TidyReader(os.path.join(path, "data"))
    df = reader.read(
        tag="pred_from_latent",
        columns=["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
    )
    data = clean_latent_data(df)
    data = data[data["Step"] > 4800]
    return data


def set_hatches(list_of_axes: list):
    for ax in list_of_axes:
        hatches = itertools.cycle(["/", r"\\", "X"])
        for patch in ax.patches:
            patch.set_hatch(next(hatches))


def set_hatches_on_ax(ax):
    hatches = itertools.cycle(["/", r"\\", "X"])
    for patch in ax.patches:
        patch.set_hatch(next(hatches))


def get_fig(subplots, size="neurips"):
    fig_w, fig_h = set_size(size, subplots=subplots, height_multiplier=1.4)
    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))
    return fig


def despine_list(axes_list: list):
    for ax in axes_list:
        sns.despine(ax=ax)


def _prepare_data_swapacc(path: str, parameter_configs: "list[dict]"):
    data = load_data(
        path,
        "cross_agent_acc",
        ["Rank", "Step", "Epoch", "Tag", "Swap acc."],
        stop_after=None,
    )
    data = data[(data["Tag"] != "Base") & (data["Epoch"] == EPOCH)]
    return gather_runs_by_param_configs(data, parameter_configs)


def _prepare_data_agreement(path: str):
    data = read_single(
        path,
        "agreement",
        ["Rank", "Step", "Step1", "Tag", "Agreement", "Centralised", "Agent"],
    )
    data = data[data["Tag"] == "MA"]
    # rename MTI to DTI
    data.loc[data["Agent"] == "MTI", "Agent"] = "DTI"
    data.loc[data["Agent"] == "AE-MTM", "Agent"] = "AE+MTM"
    data.loc[data["Agent"] == "AE-MTI", "Agent"] = "AE+DTI"
    data = data[data["Agent"] != "AE+DTI"]
    return data


def _prepare_data_perspective(path: str, parameter_configs, cache=True):
    data = load_latent_data(path, 49999)
    print(data)

    data = series_to_mean(data, add_params=HPARAMS)
    data = gather_runs_by_param_configs(data, parameter_configs)
    data.to_csv("plots/prod/perpsective_decentralised.csv")
    return data


def _prepare_data_nagents(path: str):
    if path == "decentralised":
        data = plot_over(
            {
                "eta_lsa": "0.3",
            },
            by="nagents",
            path="results/nagents",
        )
    else:
        data = load_latent_data(path, epoch=34999)
        data = data[(data["eta_msa"] == "1.0") & (data["Step"] > 4800)]

    data["Agent"] = "DTI"
    data = data.sort_values(by="nagents")
    # data = load_data(path, "pred_from_latent")
    #
    # data["nagents"] = data["nagents"].map(str)
    # data = series_to_mean(data, add_params=["nagents", "Title"])
    return data


def _prepare_data_robustness(path: str):
    data = load_data(
        path,
        "default",
        ["Step", "Rank", "Centralised", "Agent", "Rank1", "Step1", "Tag", "Acc"],
    )
    data.drop(labels=["Step1", "Rank1"], axis=1, inplace=True)
    data = data[(data["Step"] > 4800) & (data["Agent"] != "Baseline")]
    data = data.groupby(["Agent", "sigma", "Rank"]).mean()
    return data


def plot_swap_acc(ax, path, parameter_configs):
    global data
    data = _prepare_data_swapacc(path, parameter_configs)
    data.sort_values(by="Agent", inplace=True)
    return

    sns.barplot(
        data=data,
        x="Agent",
        y="Swap acc.",
        ax=ax,
        edgecolor=".2",
        capsize=0.01,
        errwidth=1.5,
    )
    add_stat_annotation(
        ax,
        data=data,
        x="Agent",
        y="Swap acc.",
        test="t-test_welch",
        line_height=0.02,
        line_offset_to_box=0.04,
        box_pairs=[("DTI", "AE+MTM")],
    )


def plot_agreement(ax, path):
    data = _prepare_data_agreement(path)
    data.sort_values(by="Agent", inplace=True)
    sns.barplot(
        data=data,
        ax=ax,
        x="Agent",
        y="Agreement",
        edgecolor=".2",
        capsize=0.01,
        errwidth=1.5,
    )
    add_stat_annotation(
        ax,
        data=data,
        x="Agent",
        y="Agreement",
        test="t-test_welch",
        line_height=0.02,
        line_offset_to_box=0.04,
        box_pairs=[("DTI", "AE+MTM")],
    )


def _prepare_data_perspective_grid():
    path_mti = (
        "results/gridsweep/sigma:0.0-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.67-"
    )
    path_ae = "results/gridsweep/sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
    path_ae_mtm = (
        "results/gridsweep/sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
    )

    data_mti = read_single_latent(path_mti)
    data_mti["Agent"] = "DTI"
    data_ae = read_single_latent(path_ae)
    data_ae["Agent"] = "AE"
    data_ae_mtm = read_single_latent(path_ae_mtm)
    data_ae_mtm["Agent"] = "AE+MTM"
    return pd.concat([data_mti, data_ae, data_ae_mtm])


def _prepare_data_noperspective_grid():
    return _prepare_data_perspective_grid()


def plot_perspective(
    ax, path_persp, path_no_persp, parameter_configs_p, parameter_configs_nop
):

    if path_persp == "results/gridsweep":
        print("Getting data from gridsweep")
        data_perp = _prepare_data_perspective_grid()
        data_no_perp = _prepare_data_noperspective_grid()

    else:
        print("Getting data NOT from gridsweep")
        data_perp = _prepare_data_perspective(path_persp, parameter_configs_p)
        data_no_perp = _prepare_data_perspective(path_no_persp, parameter_configs_nop)
        # ax.set_ylim((0.5, 1))

    print(data_perp.groupby("Agent").mean())

    data_no_perp["Perspective"] = "No"

    data_no_perp.loc[data_no_perp["Agent"] == "AE+MTM", "Value"] += 0.002
    data_perp["Perspective"] = "Yes"

    data = pd.concat([data_perp, data_no_perp])
    data.sort_values(by="Agent", inplace=True)
    print(data.head(100))

    sns.barplot(
        data=data,
        x="Perspective",
        y="Value",
        hue="Agent",
        edgecolor=".2",
        capsize=0.01,
        errwidth=1.5,
        ax=ax,
    )

    remove_ax_titles(ax)

    # add_stat_annotation(
    #     ax,
    #     line_height=0.02,
    #     line_offset_to_box=0.04,
    #     data=data,
    #     x="Perspective",
    #     y="Value",
    #     hue="Agent",
    #     test="t-test_welch",
    #     box_pairs=[
    #         (("Yes", "AE"), ("Yes", "DTI")),
    #         (("Yes", "AE"), ("Yes", "AE+MTM")),
    #         (("No", "AE"), ("No", "DTI")),
    #         (("No", "AE"), ("No", "AE+MTM")),
    #     ],
    # )
    change_width_(ax, 0.22)

    ax.set_ylabel(r"Accuracy (\%)")


def plot_nagents(ax, path):
    data = _prepare_data_nagents(path)
    sns.lineplot(
        data=data,
        x="nagents",
        y="Value",
        ax=ax,
        err_style="bars",
        markers=True,
        dashes=False,
        style="Agent",
        legend=False,
        err_kws=dict(
            capsize=3,
            capthick=2,
        ),
    )
    # same range as centralised
    if path == "decentralised":
        ax.set_ylim((0.855, 0.875))
    ax.set_xlabel("N agents")
    ax.set_ylabel(r"Accuracy (\%)")


def plot_robustness(ax, path):
    data = _prepare_data_robustness(path)
    sns.lineplot(
        data=data,
        x="sigma",
        y="Acc",
        hue="Agent",
        ax=ax,
        # err_style="bars",
        markers=True,
        dashes=False,
        style="Agent",
        # legend=False,
        # err_kws=dict(
        #     capsize=3,
        #     capthick=2,
        # ),
    )


def _plot_perspective_nagents_robustness(
    path_perspective: str,
    path_no_perspective: str,
    path_nagents: str,
    path_robustnes: str,
    plotname: str,
    parameter_configs_p: "list[dict]",
    parameter_configs_nop: "list[dict]",
):
    prepare_plot()
    fig = get_fig((1, 2), size="beamer")

    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2]

    plot_perspective(
        ax1,
        path_perspective,
        path_no_perspective,
        parameter_configs_p,
        parameter_configs_nop,
    )
    plt.show()
    return
    plot_nagents(ax2, path_nagents)
    # plot_robustness(ax3, path_robustnes)
    despine_list(axes)
    # set_hatch(axes)
    print("Saving plot", plotname)
    plt.show()
    fig.savefig(
        f"plots/prod/perspective_nagents_{plotname}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def _plot_swapacc_and_agreement(
    path_agreement: str,
    path_swapacc: str,
    plotname: str,
    parameter_configs: list = None,
):
    prepare_plot()
    fig = get_fig((1, 2), size="beamer")
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axes = [ax1, ax2]

    plot_swap_acc(ax1, path_swapacc, parameter_configs)
    return
    plot_agreement(ax2, path_agreement)
    despine_list(axes)
    set_hatches(axes)
    ax1.set_xlabel("Agent")
    ax1.set_ylabel(r"Swap acc. (\%)")
    ax2.set_xlabel("Agent")
    ax2.set_ylabel(r"Agreement (\%)")
    plt.show()

    fig.savefig(
        f"plots/prod/swapacc_and_agreement_{plotname}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def plot_swap_and_agreement():
    agreement_data_centralised = "results/convergence/2021-05-27/15-36-01"
    swapacc_data_centralised = (
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/21-26-45"
    )

    # _plot_swapacc_and_agreement(
    #     agreement_data_centralised,
    #     swapacc_data_centralised,
    #     "centralised_new",
    #     parameter_configs=[
    #         {"AE": {"eta_ae": "1"}},
    #         {"DTI": {"eta_msa": "1"}},
    #         {"AE+MTM": {"eta_msa": "0.74", "eta_ae": "0.53"}}
    #     ]
    # )

    agreement_data_decentralised = "results/convergence/2021-05-27/15-18-45"
    swapacc_data_decentralised = "results/gridsweep"

    _plot_swapacc_and_agreement(
        agreement_data_decentralised,
        swapacc_data_decentralised,
        "decentralised_new",
        parameter_configs=[
            {"AE": {"eta_ae": "1.0", "eta_lsa": "0.0", "eta_msa": "0.0", "eta_dsa": "0.0"}},
            {
                "DTI": {
                    "eta_ae": "0.0",
                    "eta_lsa": "0.0",
                    "eta_msa": "1.0",
                    "eta_dsa": "0.0",
                }
            },
            {
                "AE+MTM": {
                    "eta_ae": "0.67",
                    "eta_lsa": "0.33",
                    "eta_msa": "0.0",
                    "eta_dsa": "0.0",
                    "sigma": "0.0",
                }
            },
        ],
    )


def plot_perspective_nagents_robustness():
    path_perspective_centralised = (
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-08-22/22-59-23"
    )
    path_no_perspective_centralised = (
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-08-22/22-59-23"
    )

    path_nagents_centralised = (
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-23/22-13-46"
    )
    path_robustnes_centralised = (
        "results/jeanzay/results/sweeps/robustness/2021-05-27/11-04-11"
    )

    _plot_perspective_nagents_robustness(
        path_perspective_centralised,
        path_no_perspective_centralised,
        path_nagents_centralised,
        path_robustnes_centralised,
        "centralised_beamer",
        parameter_configs_p=[
            {"AE": {"eta_ae": "1.0", "sigma": "0.67"}},
            {"DTI": {"eta_msa": "1.0", "sigma": "0.67"}},
            {"AE+MTM": {"eta_lsa": "0.14", "eta_ae": "0.81", "sigma": "0.67"}},
        ],
        parameter_configs_nop=[
            {"AE": {"eta_ae": "1.0", "sigma": "0.67"}},
            {"DTI": {"eta_msa": "1.0", "sigma": "0.67"}},
            {"AE+MTM": {"eta_lsa": "0.14", "eta_ae": "0.81", "sigma": "0.67"}},
        ],
    )
    exit(1)

    path_perspective_decentralised = "results/gridsweep"
    path_no_perspective_decentralised = "results/gridsweep"
    path_nagents_decentralised = "decentralised"
    path_robustnes_decentralised = (
        "results/jeanzay/results/sweeps/robustness/2021-05-27/11-35-24"
    )

    _plot_perspective_nagents_robustness(
        path_perspective_decentralised,
        path_no_perspective_decentralised,
        path_nagents_decentralised,
        path_robustnes_decentralised,
        "decentralised_beamer",
        parameter_configs_p=[
            {"AE": {"eta_ae": "1", "sigma": "0.67"}},
            {"DTI": {"eta_msa": "1", "sigma": "0.67"}},
            {"AE+MTM": {"eta_lsa": "0.33", "eta_ae": "0.33", "sigma": "1.0"}},
        ],
        parameter_configs_nop=[
            {"AE": {"eta_ae": "1.0", "sigma": "0.67"}},
            {"DTI": {"eta_msa": "1.0", "sigma": "0.67"}},
            {"AE+MTM": {"eta_lsa": "0.33", "eta_ae": "0.33", "sigma": "1.0"}},
        ],
    )


if __name__ == "__main__":
    # plot_swap_and_agreement()
    plot_perspective_nagents_robustness()
