import seaborn as sns
import matplotlib.pyplot as plt


def set_size(width, fraction=0.97, subplots=(1, 1), height_multiplier=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    height_multiplier: float, optional
            Fraction to increase height if desired.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "neurips":
        width_pt = 397.48499
    elif width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * height_multiplier

    return (fig_width_in, fig_height_in)


def set_tex_fonts(fontsize=10, label_fontsize=8, legend_fontsize=8):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": label_fontsize,
        "ytick.labelsize": label_fontsize,
    }
    plt.rcParams.update(tex_fonts)


def set_palette():
    sns.set_palette(sns.color_palette("Set1"))