from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from reader.chunked_writer import TidyReader
from pandas.core.frame import DataFrame

from experiments.experiment import BaseExperiment

# matplotlib.style.use(matplotlib.get_data_path() + "/stylelib/apa.mplstyle")


class Experiment(BaseExperiment):
    def load_data(reader: TidyReader) -> Any:
        return reader.read(
            tag="loss", columns=["Step", "Rank", "Loss", "Type", "Agent_i", "Agent_j"]
        )

    def plot(df: DataFrame, plot_path: str) -> None:
        df = df[(df["Type"] == "AE") | (df["Type"] == "LSA") | (df["Type"] == "MSA")]
        df = df[(df["Agent_j"] != "baseline") & (df["Agent_i"] != "baseline")]
        sns.relplot(
            data=df,
            x="Step",
            y="Loss",
            hue="Type",
            kind="line",
            col="Agent_i",
            col_wrap=2,
            ci=None,
            facet_kws=dict(sharey=False),
        )
        plt_name = f"{plot_path}/ae_msa_end"
        plt.xlim((45000, 50000))
        plt.savefig(plt_name + ".pdf")
        plt.savefig(plt_name + ".svg")
        plt.savefig(plt_name + ".png")
