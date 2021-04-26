import os
import string
import itertools
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.types import Number

from autoencoder import AutoEncoder
from mnist import MNISTDataset


EPOCH = 49999.0


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


def plot_tsne(path, path_to_plot):
    dataset = MNISTDataset()
    ims, labels = dataset.sample_with_label(5000)
    all_agents = _load_aes(path)

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
                f"params/step_{int(EPOCH)}/rank_0/{'A' if not baseline else 'baseline'}.pt",
            ),
            map_location=torch.device("cpu"),
        )
    )
    digits: torch.Tensor = dataset.sample(50)

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


def compute_and_save_cov_matrix():
    pass


def main(path_to_results: str, exp_name: str, path_to_plot: str):
    # t-sne in latent space
    plot_tsne(
        os.path.join(path_to_results, exp_name, f"params/step_{int(EPOCH)}/rank_0"),
        path_to_plot,
    )

    # reconstruction from good marl agents vs. baseline agents for some digits
    plot_img_reconstructions(path_to_results, exp_name, path_to_plot, baseline=False)
    plot_img_reconstructions(path_to_results, exp_name, path_to_plot, baseline=True)
    plot_reconstruction_sim_measure(path_to_results, exp_name, path_to_plot)

    # covariance matric between hparams and losses (final?)
    compute_and_save_cov_matrix()


if __name__ == "__main__":
    main()