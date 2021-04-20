import os
import random
import string
from itertools import combinations
from pathlib import Path
from typing import AnyStr, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE

from autoencoder import AutoEncoder
from chunked_writer import TidyReader
from mnist import MNISTDataset


def load_df_and_params(posixpath, tag, columns):
    """
    Args:
        posixpath: Posixpath. Path to one specific epxeriment
        tag: String. The name of the metric which we want to retrieve
        columns: List[String]. The column headers of the resulting dataframe
    Returns:
        df: DataFrame
    """
    reader = TidyReader(os.path.join(posixpath, "data"))
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


def stem_to_params(stem) -> dict:
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
    plt.savefig(f"plots/{path_to_plot}/pcoords_{tag}.pdf")
    plt.close()


def series_to_mean(df):
    groups = df.groupby(["Rank", "Metric", "Type", "Agent", "Epoch"], as_index=False)
    return groups.apply(lambda x: x[x["Step"] >= 4000].mean())


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


def get_best_params(df, tolerance=0.00, hparams=[]):
    # compute the mean across ranks
    groups = df.groupby([*hparams, "Agent", "Epoch"], as_index=False).mean()
    # filter groups with lower final performance
    res = groups.groupby([*hparams, "Epoch"]).filter(
        lambda x: x[x["Agent"] == "baseline"].Value
        < x[x["Agent"] != "baseline"].Value.max() + tolerance
    )
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
        df = series_to_mean(df)
        for param, value in params.items():
            df[param] = value
        dfs.append(df)
    return pd.concat(dfs)


def compute_and_save_reg_coefs(df, hparams, tag, path_to_plot):
    # compute impact of hparams on prediction

    ## filter out baseline because most parameters have no influence on it
    ## only look at last epoch
    df = df[(df["Agent"] != "baseline") & (df["Epoch"] == 49999.0)]
    ## compute the mean across ranks and agents to arrive at 1 acc. value per set of hparams
    groups: DataFrame = df.groupby(hparams, as_index=False).mean()

    # write to latex before regressing
    # this contains a table of all parameters
    with open(f"plots/{path_to_plot}/{tag}_table_view.txt", "w") as f:
        f.writelines(
            groups.drop(["Epoch", "Rank", "Step"], axis=1)
            .sort_values(by="Value", ascending=False)
            .to_latex()
        )

    X, y = groups.loc[:, hparams], groups.loc[:, "Value"]
    coefs = compute_reg_coefs(X, y)
    columns = list(map(lambda x: f"beta_{x}", hparams))
    df_coefs = pd.DataFrame((coefs, coefs), columns=columns)
    df_coefs.to_csv(f"plots/{path_to_plot}/{'-'.join(hparams)}_params_{tag}.csv")


def compute_barplots(df, hparams, tag, path_to_plot):
    df = get_best_params(df, 0.0, hparams)
    df = df[df["Epoch"] == 49999.0]
    param_col_name = ""
    for param in hparams:
        param_col_name += param + "=" + df[param] + " "

    df["params"] = param_col_name
    if len(df) > 0:
        g = sns.catplot(
            data=df, kind="bar", col="params", x="Agent", y="Value", col_wrap=3
        )
        g.set_titles("{col_name}")
        plt.savefig(f"plots/{path_to_plot}/bar_{tag}.pdf")
    else:
        print("Arsch")
    plt.close()


def compute_best_vs_base(df, hparams, tag, path_to_plot):
    df = get_best_params(df, 1.0, hparams)
    trans = df.groupby([*hparams, "Epoch"], as_index=False).apply(
        lambda x: x.assign(
            max_diff=x[x.Agent != "baseline"].Value.max()
            - x[x.Agent == "baseline"].Value,
        ),
    )
    trans = trans[trans.Agent == "baseline"]
    trans = trans[trans.Epoch == 49999.0]
    trans = trans.loc[
        :, ["eta_ae", "eta_lsa", "eta_dsa", "eta_msa", "sigma", "Epoch", "max_diff"]
    ].sort_values(by="max_diff", axis=0, ascending=False)

    plt.bar(["Best", "Mean"], [trans.max_diff.iloc[0], trans.max_diff.mean()])
    plt.annotate(str(round(trans.max_diff.iloc[0], 2)), (0, trans.max_diff.iloc[0]))
    plt.annotate(str(round(trans.max_diff.mean(), 2)), (1, trans.max_diff.mean()))
    plt.savefig(f"plots/{path_to_plot}/best_vs_base_{tag}.pdf")
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
    df = df[df.Epoch == 49999.0]
    groups = df.groupby([*hparams], as_index=False).mean()
    plot_pcoords(groups, [*hparams, "Value"], tag, path_to_plot)


def _load_aes(path):
    autoencoders = [
        AutoEncoder(30, bnorm=False, affine=False, name=name, lr=0.001)
        for name in string.ascii_uppercase[:3]
    ]
    baseline = AutoEncoder(30, bnorm=False, affine=False, name="baseline", lr=0.001)

    all_agents: List[AutoEncoder] = autoencoders + [baseline]
    [
        agent.load_state_dict(
            torch.load(f"{path}/{agent.name}.pt", map_location=torch.device("cpu"))
        )
        for agent in all_agents
    ]
    return all_agents


def plot_tsne(path, path_to_plot):
    dataset = MNISTDataset()
    ims, labels = dataset.sample_with_label(5000)
    all_agents = _load_aes(path)
    # load parameters from savefiles
    # for i, ae in enumerate(autoencoders):
    #     ae.load_state_dict(
    #         torch.load(
    #             os.path.join(path, f"{string.ascii_uppercase[i]}.pt"),
    #             map_location=torch.device("cpu"),
    #         )
    #     )
    # baseline.load_state_dict(
    #     torch.load(os.path.join(path, "baseline.pt"), map_location=torch.device("cpu"))
    # )

    results = []

    for ae in all_agents:
        encoded = ae.encode(ims)
        embedding = TSNE(n_components=2, random_state=123).fit_transform(
            encoded.detach()
        )
        for emb, label in zip(embedding, labels):
            results.append((emb[0], emb[1], int(label.item()), ae.name))

    _generate_tsne_relplot(results, path_to_plot)


def _generate_tsne_relplot(results, path_to_plot):
    data = pd.DataFrame(results, columns=["x", "y", "cls", "agent"])
    data["cls"] = data.cls.astype("category")
    sns.relplot(
        data=data,
        x="x",
        y="y",
        hue="cls",
        col="agent",
        facet_kws={"sharex": False, "sharey": False},
    )
    plt.savefig(f"plots/{path_to_plot}/t_sne.svg")
    plt.savefig(f"plots/{path_to_plot}/t_sne.pdf")
    plt.close()


def predict_9s_and_4s(path):
    # torch.manual_seed(42)
    dataset = MNISTDataset()
    autoencoders, baseline = _load_aes(path)
    all_agents: List[AutoEncoder] = autoencoders + [baseline]

    results = []

    mlps: List[MLP] = [MLP(30) for _ in all_agents]
    [mlp.eval() for mlp in mlps]
    bsize = 128
    nsteps = 100

    for mlp, agent in zip(mlps, all_agents):
        for i in range(nsteps):
            digit = random.choice([4, 9])
            ims = dataset.sample_digit(digit, bsize)
            encoding = agent.encode(ims)
            targets = torch.empty(bsize).fill_(digit).long()
            mlp.train_batch(encoding, targets)
            acc = mlp.compute_acc(encoding, targets)
            results.append((agent.name, i, acc))

    df = pd.DataFrame(results, columns=["Agent", "Step", "Accuracy"])
    df = (
        df.groupby(["Agent"], as_index=False)
        .apply(lambda x: x[x.Step >= nsteps - 50])
        .groupby(["Agent"], as_index=False)
        .mean()
    )
    sns.barplot(data=df, x="Agent", y="Accuracy")
    plt.savefig("plots/4s_9s_bar.pdf")
    plt.savefig("plots/4s_9s_bar.svg")
    plt.close()


def plot_img_reconstructions(
    root_path: str, name_of_best_exp: str, path_to_plot: str, baseline: bool = False
):
    dataset = MNISTDataset()
    ae = AutoEncoder(30, False, False, 0.001, "test")
    ae.load_state_dict(
        torch.load(
            os.path.join(
                root_path,
                name_of_best_exp,
                f"params/step_49999/rank_0/{'A' if not baseline else 'baseline'}.pt",
            ),
            map_location=torch.device("cpu"),
        )
    )
    digits = dataset.sample(50)

    fig, axes = plt.subplots(
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

    plt_path = f"plots/{path_to_plot}/reconstructions_baseline_{baseline}"
    plt.savefig(plt_path + ".pdf")
    plt.savefig(plt_path + ".svg")
    plt.close()


def plot_reconstruction_sim_measure(
    root_path: str, name_of_exp: str, path_to_plot: str
):
    dataset = MNISTDataset()
    agents = _load_aes(
        os.path.join(root_path, name_of_exp, "params", "step_49999", "rank_0")
    )

    results = []
    for i in range(10):
        batch = dataset.sample_digit(i)
        for agent in agents:
            rec = agent(batch)
            for a, b in combinations(rec, r=2):
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


def compute_and_save_cov_matrix():
    pass


def make_plots(path_to_results: AnyStr, hparams: List, path_to_plot: AnyStr):
    if not os.path.exists("plots/" + path_to_plot):
        os.makedirs("plots/" + path_to_plot)

    df = load_data_raw(path_to_results)

    compute_plots_latent(df, hparams, path_to_plot)
    compute_plots_rec(df, hparams, path_to_plot)

    # this should be automatically determined?
    #  40 draws run
    # name_of_best_exp = (
    #     "sigma:0.503-eta_lsa:0.015-eta_ae:0.036-eta_dsa:0.936-eta_msa:0.7-"
    # )
    # 100 draws only msa
    # name_of_best_exp = "sigma:0-eta_lsa:0-eta_msa:1-eta_dsa:0-eta_ae:0-"
    # name_of_best_exp = (
    #     "sigma:0.757-eta_lsa:0.004-eta_msa:0.483-eta_dsa:0.623-eta_ae:0.153-"
    # )
    name_of_best_exp = "sigma:0-eta_lsa:1-eta_msa:0-eta_dsa:0-eta_ae:0-"

    # t-sne in latent space
    plot_tsne(
        os.path.join(path_to_results, name_of_best_exp, "params/step_49999/rank_0"),
        path_to_plot,
    )

    # this is a separate experiment, should be done elsewhere
    # predict_9s_and_4s(
    #     "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-12/21-04-14/"
    #     "sigma:0.503-eta_lsa:0.015-eta_ae:0.036-eta_dsa:0.936-eta_msa:0.7-/params/step_49999/rank_0"
    # )

    # reconstruction from good marl agents vs. baseline agents for some digits
    plot_img_reconstructions(
        path_to_results, name_of_best_exp, path_to_plot, baseline=False
    )
    plot_img_reconstructions(
        path_to_results, name_of_best_exp, path_to_plot, baseline=True
    )
    plot_reconstruction_sim_measure(path_to_results, name_of_best_exp, path_to_plot)

    # covariance matric between hparams and losses (final?)
    compute_and_save_cov_matrix()


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.l = nn.Linear(input_size, 10)
        self.opt = optim.Adam(self.parameters())

    def forward(self, x):
        return self.l(x)

    def compute_acc(self, ims, labels):
        pred: torch.Tensor = self.l(ims).argmax(dim=1)
        acc = (pred == labels).float().mean()
        return acc.item()

    def train_batch(self, inputs, targets):
        x = self.l(inputs)
        loss = F.cross_entropy(x, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


if __name__ == "__main__":
    make_plots(
        "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-16/13-15-58",
        ["eta_ae", "eta_lsa", "eta_dsa", "eta_msa", "sigma"],
        "100-draws-lsa",
    )
