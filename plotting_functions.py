from plotting_helpers import set_size, set_tex_fonts, set_palette
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

EPOCH = 39999.0


def prepare_plot():
    plt.clf()
    set_tex_fonts()
    set_palette()


def plot_nagents():
    pass


def plot_perspective():
    pass


def plot_perspective_and_nagents(df: DataFrame, plotname: str):
    plt.clf()
    pass


def plot_swap_acc(ax, data):
    pass


def plot_agreement(ax, data):
    pass


def _prepare_data_swapacc(data: DataFrame):
    data = data[(data["Tag"] != "Base") & (data["Epoch"] == EPOCH)]
    df_msa = data[data["eta_msa"] == "1"]
    df_msa["Type"] = "MTI"
    return data


def _prepare_data_agreement(data: DataFrame):
    pass


def plot_swapacc_and_agreement(data: DataFrame, plotname: str):
    prepare_plot()
    fig_w, fig_h = set_size("neurips", subplots=(1, 2))
    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))

    data_swappac = _prepare_data_swapacc(data)
    data_agreement = _prepare_data_agreement(data)

    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axes = [ax1, ax2]
    plot_swap_acc(ax1, data)
    plot_sim_measure(ax2, data)
    sns.despine(axes)

    fig.savefig(
        f"plots/prod/swapacc_and_sim_measure_{plotname}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":

    agreement_data_centralised = ""
    agreement_data_decentralised = ""

    df = DataFrame()
    plot_swapacc_and_agreement(df, "centralised")
    plot_perspective_and_nagents(df, "centralised")

    plot_swapacc_and_agreement(df, "decentralised")
    plot_perspective_and_nagents(df, "decentralised")
