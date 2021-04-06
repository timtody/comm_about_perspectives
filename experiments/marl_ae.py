from typing import Tuple

import c_types
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_line(df):
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Loss")
    plt.savefig("line.pdf")
    plt.close()


def plot_scatter(df):
    plt.figure()
    sns.relplot(data=df, x="x", y="y", col="Rank")
    plt.savefig("scatter.pdf")
    plt.close()


def filter_df(df: c_types.DataFrame, datapoints: int = 50, sort: bool = False):
    if sort:
        df = df.sort_values(by=["Step"])
    nsteps = df.groupby("Rank").size()[0]
    return df.groupby("Rank").apply(lambda x: x[:: nsteps // datapoints])


class Experiment(c_types.BaseExperiment):
    @staticmethod
    def load_data(
        reader: c_types.TidyReader,
    ) -> Tuple[c_types.DataFrame, c_types.DataFrame]:
        df1 = filter_df(reader.read("sin", ["Rank", "Step", "Loss"]), sort=True)
        df2 = filter_df(reader.read("scat", ["Rank", "x", "y"]))
        return df1, df2

    @staticmethod
    def plot(args) -> None:
        df1, df2 = args
        plot_line(df1)
        plot_scatter(df2)

    def run(self):
        for i in range(10000):
            self.writer.add((i, np.sin(i) * 0.01 + np.random.randn() * 0.5), tag="sin")
            self.writer.add((np.random.randn(), np.random.randn()), tag="scat")

            if i % 1000 == 999:
                self.log(i)
