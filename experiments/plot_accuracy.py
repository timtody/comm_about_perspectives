from typing import Any

import matplotlib.pyplot as plt
from pandas.io.formats import style
import seaborn as sns
from chunked_writer import TidyReader

from experiments.experiment import BaseExperiment

sns.set(style="whitegrid")


class Experiment(BaseExperiment):
    def load_data(reader: TidyReader) -> Any:
        df = reader.read(
            tag="pred_from_latent",
            columns=["Epoch", "Rank", "Step", "Value", "Metric", "Type", "Agent"],
        )
        df = df[df.Epoch == 49999.0]
        df = df[df.Metric == "Accuracy"]
        df.Agent = df.Agent.map(
            {
                "A": "MA",
                "B": "MA",
                "C": "MA",
                "baseline": "Baseline",
            }
        )
        df = df.groupby(["Rank", "Agent", "Type"], as_index=False).apply(
            lambda x: x[::4]
        )
        df_latent = df[df.Type == "Latent"]
        df_rec = df[df.Type == "Reconstruction"]

        return df_latent, df_rec

    def plot(dataframes, path) -> None:
        df_latent, df_rec = dataframes
        plot_line(df_latent, path + "/accuracy_latent")
        plot_line(df_rec, path + "/accuracy_rec")


def plot_line(df, path):
    sns.lineplot(
        data=df,
        x="Step",
        y="Value",
        hue="Agent",
        dashes=False,
        markers=True,
        style="Agent",
        hue_order=["Baseline", "MA"],
    )
    df.to_csv(path + "_results.csv")
    plt.savefig(path + ".pdf")
    plt.savefig(path + ".svg")
    plt.close()