from io import DEFAULT_BUFFER_SIZE
import itertools
import os
import string
from pathlib import Path, PosixPath
from typing import Callable, Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from torch.types import Number

from autoencoder import AutoEncoder
from chunked_writer import TidyReader
from mnist import MNISTDataset

EPOCH = 49999.0
DATA_LEN = 99999
# sns.set(style="whitegrid")


def load_df_and_params(
    posixpath: PosixPath, tag: str, columns: List[str], datafolder="data"
) -> Tuple[DataFrame, Dict[str, str]]:
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


def stem_to_params(stem: str) -> Dict[str, str]:
    """
    Helper function which takes in a path stem of the form "param:value-param2:value2..."
    and returns a dictionary of parameters, e.g. "{"param":value,...}
    Args:
        stem: String. The name of the folder.
    """
    params = {k: v for d in map(eval_d, stem.split("-")[:-1]) for k, v in d.items()}
    return params


def plot_pcoords(df, labels, tag, path_to_plot):
    columns_to_drop = [col for col in df.columns if col not in labels]
    df = df.drop(axis=1, labels=columns_to_drop)
    _, axes = plt.subplots(ncols=len(labels) - 1, sharey=False, figsize=(20, 8))

    for i, ax in enumerate(axes):
        for ix in df.index:
            ax.plot(
                [0, 1],
                df.loc[ix, labels[i] : labels[i + 1]].astype(float),
                c=cm.Spectral(df.loc[ix, "Value"]),
            )
            ax.set_xlim((0, 1))
            ax.set_ylim((-0.05, 1.05))
            try:
                label = labels[i].split("_")[1]
            except:
                label = labels[i]
            ax.set_xticklabels([label])

    ax = plt.twinx(axes[-1])
    ax.set_xticks([0, 1])
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlim((0, 1))
    ax.set_xticklabels([labels[-2], labels[-1]])
    plt.subplots_adjust(wspace=0)
    plt.savefig(f"{path_to_plot}/pcoords_{tag}.pdf")
    plt.close()


def series_to_mean(df, threshold=4000):
    groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
    return groups.apply(lambda x: x[x["Step"] >= threshold].mean())


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


def get_best_params(df: DataFrame, tolerance: float = 0.00, hparams: List[str] = []):
    """
    Filters out results associated with hyperparameters which don't match
    specific performance criteria, controlled by tolerance.

    A set of hyperparameter gets filtered out if the mean performance (across ranks)
    of the best agent (across ranks and agents) is worse than the mean performance
    of the baseline (across ranks), adjusted by :tolerance:.
    """
    filter_fn: Callable = (
        lambda x: x[x["Agent"] == "baseline"].mean().Value
        < x[x["Agent"] != "baseline"].groupby("Agent").mean().Value.max() + tolerance
    )
    res = df.groupby([*hparams, "Epoch"]).filter(filter_fn)
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
        df = series_to_mean(df, threshold=0)
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
        if i > DATA_LEN:
            break
    return pd.concat(dfs)


def load_loss_data(path, threshold=2000):
    paths = Path(path).glob("*")
    dfs = []
    i = 0
    for path in paths:
        i += 1
        df, params = load_df_and_params(
            path,
            "loss",
            ["Step", "Rank", "Loss", "Type", "Agent_i", "Agent_j"],
        )
        groups = df.groupby(["Type", "Agent_i"], as_index=False)
        df = groups.apply(
            lambda x: x[x["Step"] >= 45000].drop(axis=1, labels="Agent_j").mean()
        )
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
        if i > DATA_LEN:
            break
    return pd.concat(dfs)


def load_acc_data(path):
    paths = Path(path).glob("*")
    dfs = []
    i = 0
    for path in paths:
        i += 1
        df, params = load_df_and_params(
            path,
            "cross_agent_accuracy_override",
            ["Rank", "Tag", "Accuracy"],
            "",
        )
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
        if i > DATA_LEN:
            break
    return pd.concat(dfs)


def compute_and_save_reg_coefs(df, hparams, tag, path_to_plot):
    # compute impact of hparams on prediction

    ## filter out baseline because most parameters have no influence on it
    ## only look at last epoch
    df = df[(df["Agent"] != "baseline") & (df["Epoch"] == EPOCH)]
    ## compute the mean across ranks and agents to arrive at 1 acc. value per set of hparams
    groups: DataFrame = df.groupby(hparams, as_index=False).mean()

    # write to latex before regressing
    # this contains a table of all parameters
    with open(f"{path_to_plot}/{tag}_table_view.txt", "w") as f:
        f.writelines(
            groups.drop(["Epoch", "Rank", "Step"], axis=1)
            .sort_values(by="Value", ascending=False)
            .to_latex()
        )

    X, y = groups.loc[:, hparams], groups.loc[:, "Value"]
    coefs = compute_reg_coefs(X, y)
    columns = list(map(lambda x: f"beta_{x}", hparams))
    df_coefs = pd.DataFrame((coefs,), columns=columns)
    df_coefs.to_csv(f"{path_to_plot}/{'-'.join(hparams)}_params_{tag}.csv")


def compute_barplots(df, hparams, tag, path_to_plot):
    df = get_best_params(df, 0.0, hparams)
    df = df[df["Epoch"] == EPOCH]
    param_col_name = ""
    for param in hparams:
        param_col_name += param + "=" + df[param] + " "

    df["params"] = param_col_name
    if len(df) > 0:
        g = sns.catplot(
            data=df, kind="bar", col="params", x="Agent", y="Value", col_wrap=3
        )
        g.set_titles("{col_name}")
        plt.savefig(f"{path_to_plot}/bar_{tag}.pdf")
    else:
        print("Arsch")
    plt.close()


def compute_best_vs_base(
    df: DataFrame, hparams: List[str], tag: str, path_to_plot: str
):
    """
    Plots the amount of additional accuracy the best agents gains over the baseline
    in the best case vs. the additional accuracy the best agents gets on average.
    """

    # get parameter sets which are 1.0 better than baseline (i.e. all of them)
    df = get_best_params(df, 1.0, hparams)

    # compute difference between best agent
    apply_fn: Callable = lambda x: x.assign(
        max_diff=x[x.Agent != "baseline"].groupby("Agent").mean().Value.max()
        - x[x.Agent == "baseline"].Value
    )
    trans = df.groupby([*hparams, "Epoch"], as_index=False).apply(apply_fn)
    trans = trans[trans.Agent == "baseline"]
    trans = trans[trans.Epoch == EPOCH]
    trans = trans.loc[:, [*hparams, "Epoch", "max_diff"]].sort_values(
        by="max_diff", axis=0, ascending=False
    )

    plt.bar(["Best", "Mean"], [trans.max_diff.iloc[0], trans.max_diff.mean()])
    plt.annotate(str(round(trans.max_diff.iloc[0], 2)), (0, trans.max_diff.iloc[0]))
    plt.annotate(str(round(trans.max_diff.mean(), 2)), (1, trans.max_diff.mean()))
    plt.savefig(f"{path_to_plot}/best_vs_base_{tag}.pdf")
    plt.close()


def compute_plots_rec(df, hparams, path_to_plot):
    _make_plots(df, hparams, "Reconstruction", path_to_plot)


def compute_plots_latent(df, hparams, path_to_plot):
    _make_plots(df, hparams, "Latent", path_to_plot)


def _make_plots(df, hparams, tag, path_to_plot):
    # latent
    df_lat = df[df["Type"] == tag]
    # reg coefs
    compute_and_save_reg_coefs(df_lat, hparams, tag, path_to_plot)
    ## barplots
    compute_barplots(df_lat, hparams, tag, path_to_plot)
    ## diff between best agent and baseline
    compute_best_vs_base(df_lat, hparams, tag, path_to_plot)
    ## pcoords

    df = df[df.Agent != "baseline"]
    df = df[df.Epoch == EPOCH]
    groups = df.groupby([*hparams], as_index=False).mean()
    plot_pcoords(groups, [*hparams, "Value"], tag, path_to_plot)


def _load_aes(path):
    autoencoders = [
        AutoEncoder(30, bnorm=False, affine=False, name=name, lr=0.001)
        for name in string.ascii_uppercase[:2]
    ]
    baseline1 = AutoEncoder(30, bnorm=False, affine=False, name="Base1", lr=0.001)
    baseline2 = AutoEncoder(30, bnorm=False, affine=False, name="Base2", lr=0.001)

    all_agents: List[AutoEncoder] = autoencoders + [baseline1, baseline2]
    [
        agent.load_state_dict(
            torch.load(f"{path}/{agent.name}.pt", map_location=torch.device("cpu"))
        )
        for agent in all_agents
    ]
    return all_agents


def plot_tsne(path, path_to_plot, tag):
    dataset = MNISTDataset()
    ims, labels = dataset.sample_with_label(10000)
    all_agents = _load_aes(path)

    results = []

    for ae in all_agents:
        encoded = ae.encode(ims)
        embedding = TSNE(
            n_components=2, random_state=4444, perplexity=50, method="exact", n_jobs=8
        ).fit_transform(encoded.detach())
        for emb, label in zip(embedding[::5], labels[::5]):
            results.append((emb[0], emb[1], int(label.item()), ae.name))

    _generate_tsne_relplot(results, path_to_plot, tag)


def _generate_tsne_relplot(results, path_to_plot, tag):
    data = pd.DataFrame(results, columns=["x", "y", "cls", "agent"])
    data["cls"] = data.cls.astype("category")
    sns.relplot(
        data=data,
        x="x",
        y="y",
        hue="cls",
        col="agent",
        col_wrap=2,
        facet_kws={"sharex": True, "sharey": True},
    )
    plt.savefig(f"{path_to_plot}/t_sne_{tag}.svg")
    plt.savefig(f"{path_to_plot}/t_sne_{tag}.pdf")
    plt.close()


def plot_img_reconstructions(
    root_path: str,
    name_of_best_exp: str,
    path_to_plot: str,
    baseline: bool = False,
    epoch: int = 49999,
):
    dataset = MNISTDataset()
    ae = AutoEncoder(30, False, False, 0.001, "test")
    ae.load_state_dict(
        torch.load(
            os.path.join(
                root_path,
                name_of_best_exp,
                f"params/step_{epoch}/rank_0/{'A' if not baseline else 'baseline'}.pt",
            ),
            map_location=torch.device("cpu"),
        )
    )
    digits: torch.Tensor = dataset.sample(50)

    _, axes = plt.subplots(
        nrows=10,
        ncols=10,
        figsize=(10, 8),
        gridspec_kw=dict(
            wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845
        ),
    )
    axes = axes.reshape(50, 2)

    for digit, ax_column in zip(digits, axes):
        ax_column[0].imshow(digit.squeeze().detach())
        ax_column[0].set_axis_off()
        rec = ae(digit.reshape(1, 1, 28, 28))
        ax_column[1].imshow(rec.squeeze().detach())
        ax_column[1].set_axis_off()
    plt.show()
    exit(1)
    plt_path = f"plots/{path_to_plot}/reconstructions_baseline_{baseline}"
    plt.savefig(plt_path + ".pdf")
    plt.savefig(plt_path + ".svg")
    plt.close()


def plot_reconstruction_sim_measure(
    root_path: str, name_of_exp: str, path_to_plot: str
):
    dataset = MNISTDataset()
    agents = _load_aes(
        os.path.join(root_path, name_of_exp, "params", f"step_{int(EPOCH)}", "rank_0")
    )

    results: List[Tuple[str, Number]] = []
    for i in range(10):
        batch = dataset.sample_digit(i)
        for agent in agents:
            rec = agent(batch)
            rec = (rec - rec.mean()) / (rec.std() + 0.0001)
            for a, b in itertools.combinations(rec, r=2):
                diff = F.mse_loss(a, b)
                results.append(
                    ("MA" if agent.name != "baseline" else "baseline", diff.item())
                )
    df = pd.DataFrame(results, columns=["Agent", "Difference"])
    sns.barplot(data=df, x="Agent", y="Difference")
    plt_name = f"plots/{path_to_plot}/decoding_space_diff"
    plt.savefig(plt_name + ".svg")
    plt.savefig(plt_name + ".pdf")
    plt.close()


def compute_and_save_cov_matrix(
    df: DataFrame,
    df_acc: DataFrame,
    df_cross_acc: DataFrame,
    hparams: List,
    path: str,
    agent="baseline",
) -> None:
    plt_base_path = f"plots/{'/'.join(path.split('/')[-2:])}"
    if not os.path.exists(plt_base_path):
        os.makedirs(plt_base_path)

    if agent == "baseline":
        df = df[df["Agent_i"] == "baseline"]
        df_acc = df_acc[df_acc["Agent"] == "baseline"]
    elif agent == "ma":
        df = df[df["Agent_i"] != "baseline"]
        df_acc = df_acc[df_acc["Agent"] != "baseline"]
    else:
        raise Exception("Wrong agent type.")

    df = df.groupby(["Type", *hparams], as_index=False).mean()
    df_acc = df_acc.groupby(hparams, as_index=False).mean()
    df_cross_acc = df_cross_acc.groupby([*hparams, "Tag"], as_index=False).mean()
    df_cross_acc = df_cross_acc[df_cross_acc.Tag == "MA"]

    pivot_table = df.pivot(index=[*hparams], columns=["Type"], values="Loss")
    pivot_table = pivot_table.reset_index()

    # need to reset index because we took away elements by preceeding groupby
    df_cross_acc = df_cross_acc.reset_index()
    pivot_table["Cross_acc"] = df_cross_acc["Accuracy"]
    pivot_table["Accuracy"] = df_acc["Value"]
    pivot_table["Swap_diff"] = pivot_table["Accuracy"] - pivot_table["Cross_acc"]

    pivot_table = pivot_table[pivot_table.columns].apply(pd.to_numeric)
    pivot_table.to_csv(f"{plt_base_path}/{agent}.csv")

    # reorder the columns to make display prettier
    pivot_table = pivot_table[
        [
            "eta_dsa",
            "eta_msa",
            "eta_lsa",
            "eta_ae",
            "sigma",
            "Accuracy",
            "Swap_diff",
            "Cross_acc",
            "AE",
            "LSA",
            "DSA",
            "MSA",
            "DECDIFF",
            "MBVAR",
            "LSA-MBVAR",
        ]
    ]
    pivot_table.rename(columns={"LSA-MBVAR": "LSA/MBVAR"}, inplace=True)

    corr = pivot_table.corr()
    corr.to_csv(f"{plt_base_path}/{agent}_corr.csv")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws=dict(shrink=0.5),
        annot=True,
        # annot_kws=dict(rotation=45),
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(plt_base_path + f"/{agent}.pdf")
    plt.savefig(plt_base_path + f"/{agent}.svg")
    plt.savefig(plt_base_path + f"/{agent}.png")


def compute_cross_accuracy(path_to_results: str, hparams: "list[list]", plot_path: str):
    df_cross_acc: DataFrame = load_acc_data(path_to_results)
    df_cross_acc = df_cross_acc.groupby([*hparams, "Tag"], as_index=False).agg(
        [np.mean, np.std]
    )
    df_cross_acc.drop("Rank", axis=1, inplace=True)
    df_cross_acc = df_cross_acc.sort_values(by=("Accuracy", "mean"), ascending=False)
    df_cross_acc.to_csv(f"{plot_path}/cross_acc.csv")


def load_crs_acc_data(path: str) -> DataFrame:
    return load_acc_data(path)


def main(path_to_results: str, hparams: List[str], path_to_plot: str):
    # if not os.path.exists("plots/" + path_to_plot):
    #     os.makedirs("plots/" + path_to_plot)
    path = f"plots/{'/'.join(path_to_results.split('/')[-2:])}"
    if not os.path.exists(path):
        os.makedirs(path)

    # compute_cross_accuracy(path_to_results, hparams, path)

    # df_loss = load_loss_data(path_to_results)
    # df_acc = load_data_raw(path_to_results)
    # df_acc = df_acc[df_acc["Epoch"] == EPOCH]
    # df_acc = df_acc[df_acc["Type"] == "Latent"]
    # # df_cross_acc = load_crs_acc_data(path_to_results)

    # # df_acc = load_data_raw(path_to_results)
    # compute_plots_latent(df_acc, hparams, path)
    # compute_plots_rec(df_acc, hparams, path)
    # name_of_best_exp = (
    #    "sigma:0.001-eta_lsa:0.859-eta_msa:0.017-eta_dsa:0.149-eta_ae:0.653-"
    # )

    # t-sne in latent space
    plot_tsne(
        os.path.join(
            path_to_results,
            "sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.33-eta_dsa:0.0-",
            f"params/step_{int(EPOCH)}/rank_0",
        ),
        path,
        "sigma:0.33-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.33-eta_dsa:0.0-",
    )

    # # reconstr uction from good marl agents vs. baseline agents for some digits
    # plot_img_reconstructions(
    #     path_to_results, name_of_best_exp, path_to_plot, baseline=False
    # )
    # plot_img_reconstructions(
    #     path_to_results, name_of_best_exp, path_to_plot, baseline=True
    # )
    # plot_reconstruction_sim_measure(path_to_results, name_of_best_exp, path_to_plot)

    # # covariance matric between hparams and losses (final?)
    # compute_and_save_cov_matrix(
    #     df_loss, df_acc, df_cross_acc, hparams, path_to_results, agent="ma"
    # )
    # compute_and_save_cov_matrix(
    #    df_loss, df_acc, hparams, path_to_results, agent="baseline"
    # )


if __name__ == "__main__":
    main(
        "results/gridsweep",
        ["eta_ae", "eta_msa", "eta_lsa", "eta_dsa", "sigma"],
        "100-draws-fixed-high-lsa",
    )
