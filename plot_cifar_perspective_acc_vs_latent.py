from plotting_functions_cifar import plot_external as plot_acc
from plotting.plotting_helpers import get_size
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig_w, fig_h = get_size("neurips", (1, 2))
    fig, axes = plt.subplots(
        ncols=2, nrows=1, constrained_layout=True, figsize=(fig_w, fig_h)
    )

    fig.savefig("plots/prod/cifar_acc_and_loss_data.pdf", format="pdf", bbox_inches="tight")
