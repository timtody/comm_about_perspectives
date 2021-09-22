from plot_rec_vs_abs import plot_external as plot_reconstruction_vs_abstraction
from eval_with_reprieve import plot_external as plot_loss_data_curve
from plotting_functions import get_size, remove_legend_titles, prepare_plot
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    prepare_plot()
    fig_w, fig_h = get_size(
        "neurips", subplots=(2, 1), height_multiplier=0.5, fraction=0.98
    )
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    plot_reconstruction_vs_abstraction(axes[1], cmap)
    plot_loss_data_curve(axes[0])

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(sm)
    cbar.ax.set_ylabel(r"$\eta_{DTI}$")

    for ax in axes:
        sns.despine(ax=ax)
    remove_legend_titles(axes[0])

    axes[1].get_legend().remove()

    fig.savefig(
        "plots/prod/loss_data_and_rec_abs_stacked.pdf", bbox_inches="tight", format="pdf"
    )
