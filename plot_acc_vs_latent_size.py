import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, map_params_to_name
from plotting_functions import prepare_plot
from plotting.plotting_helpers import get_size


def series_to_dict(series) -> dict:
    return {
        "eta_ae": series["eta_ae"],
        "eta_lsa": series["eta_lsa"],
        "eta_msa": series["eta_msa"],
        "eta_dsa": series["eta_dsa"],
    }


def add_run_column(df):
    df["Run"] = map_params_to_name(series_to_dict(df))
    return df


def filter_unwanted_rows(df):
    df = df[
        (df.Epoch == 9999)
        & (df.Metric == "Test accuracy")
        & (df.Agent != "baseline_2")
        & (df.Agent != "baseline")
    ]
    return df


def main(args):
    df = load_data(
        args.results_path,
        "pred_from_latent",
        ["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
    )
    df = filter_unwanted_rows(df)
    # all but Step and Value
    grouping_indices = list(df.columns[(df.columns != "Step") & (df.columns != "Value")])
    # filter out all but last steps, then groupby run and compute mean
    df = df[df.Step >= 20000].groupby(grouping_indices, as_index=False).mean()
    # add run column and remove explicit hparams
    df = df.apply(lambda x: add_run_column(x), axis=1).drop(
        ["eta_ae", "eta_lsa", "eta_msa", "eta_dsa", "Step"], axis=1
    )
    df = df[df.Run != "All"]
    df = df[df.Run != "AE+MTM-pure"]
    df["Latent size"] = pd.to_numeric(df["latent_dim"])
    df[r"Validation accuracy (\%)"] = df["Value"]
    df["Perspective"] = df["samedigit"].map({"True": "With", "False": "Without"})

    prepare_plot()
    fig_w, fig_h = get_size("neurips", subplots=(2, 2), height_multiplier=0.7)
    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.lineplot(
        ax=ax1,
        data=df[df.samedigit == "False"],
        x="Latent size",
        y=r"Validation accuracy (\%)",
        hue="Run",
        style="Run",
        markers=True,
        dashes=False,
        legend=False,
    )
    sns.despine(ax=ax1)

    ax1.set_title("With perspective")

    sns.lineplot(
        ax=ax2,
        data=df[df.samedigit == "True"],
        x="Latent size",
        y=r"Validation accuracy (\%)",
        hue="Run",
        style="Run",
        markers=True,
        dashes=False,
    )
    ax2.set_title("Without perspective")
    sns.despine(ax=ax2)
    ax2.set_ylabel("")

    fig.savefig(
        "plots/acc_vs_latent_size.pdf",
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, type=str)
    main(parser.parse_args())
