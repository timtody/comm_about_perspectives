import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, map_params_to_name
from plotting_functions import prepare_plot


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
    print(args.results_path)
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

    prepare_plot()
    sns.relplot(
        data=df,
        x="Latent size",
        y=r"Validation accuracy (\%)",
        hue="Run",
        style="Run",
        markers=True,
        dashes=False,
        col="samedigit",
        kind="line",
    )
    plt.suptitle("Validation accuracy against latent size")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, type=str)
    main(parser.parse_args())
