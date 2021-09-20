import glob

import pandas as pd
from pandas import DataFrame
from reader import TidyReader
import matplotlib.pyplot as plt
import seaborn as sns
from plotting.plotting_helpers import get_size, set_tex_fonts, set_palette
from utils import stem_to_params


def extract_rec_and_acc_from_path(path: str, experiment: int) -> DataFrame:
    reader = TidyReader(path + "/data")
    params = stem_to_params(path.split("/")[-1])
    df_loss = (
        reader.read("loss", ["Epoch", "Rank", "Loss", "Type", "Agent_A", "Agent_B"])
        .query(
            "Epoch == 50000 and Type == 'AE' and "
            "Agent_A != 'baseline' and Agent_A != 'baseline_2' and Agent_B != 'baseline' and Agent_B != 'baseline_2'"
        )
        .drop(["Epoch", "Type"], axis=1)
        .groupby(["Rank"])
        .mean()
    )
    df_loss["Experiment"] = experiment
    df_loss[r"$\eta_{DTI}$"] = float(params["eta_msa"])
    df_acc = (
        reader.read(
            "pred_from_latent", ["Epoch", "Rank", "Step", "Acc", "Type", "Layer", "Agent"]
        )
        .query(
            "Layer == 'Latent' and Type == 'Test accuracy' and Agent != 'baseline' and Agent != 'baseline_2' "
            "and Epoch > 49000 and Step > 4000"
        )
        .drop(["Epoch", "Type", "Layer"], axis=1)
        .groupby(["Rank"])
        .mean()
    )
    df = pd.concat([df_loss, df_acc], axis=1)
    return df


def main(path: str):
    dfs = []
    for i, path in enumerate(glob.glob(path + "/*")):
        df = extract_rec_and_acc_from_path(path, i)
        dfs.append(df)

    df = pd.concat(dfs)
    # df = df.groupby("Experiment").agg(["mean", "std"]).round(4)

    fig_w, fig_h = get_size("neurips")
    fig, ax = plt.subplots(constrained_layout=True, figsize=(fig_w, fig_h))
    # ax.errorbar(
    #     df.Acc["mean"], y=df.Loss["mean"], yerr=df.Loss["std"], xerr=df.Acc["std"], fmt="o"
    # )
    # ax.errorbar()
    sns.regplot(ax=ax, data=df, x="Acc", y="Loss", scatter=False)
    sns.scatterplot(ax=ax, data=df, x="Acc", y="Loss", hue=r"$\eta_{DTI}$")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Accuracy (%)")
    sns.despine(ax=ax)
    set_tex_fonts(11, 8, 6)
    set_palette()
    # plt.show()
    fig.savefig(
        f"plots/prod/rec_vs_acc_reg.pdf",
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    data_path = "results/ae-dti-sweep"
    main(data_path)
