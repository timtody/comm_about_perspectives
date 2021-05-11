import os

import matplotlib.pyplot as plt
import torch
from mnist import MNISTDataset
from autoencoder import AutoEncoder


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


root_path = "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-20/14-58-18/"

exp_path = "sigma:0-eta_lsa:0-eta_msa:0-eta_dsa:1-eta_ae:0-"

plot_img_reconstructions(root_path, exp_path, "")
