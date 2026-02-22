import matplotlib.pyplot as plt
import matplotlib as mpl

def set_plot_style():

    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"

    plt.style.use("default")

    mpl.rcParams.update({

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        "axes.spines.top": False,
        "axes.spines.right": False,

        "patch.edgecolor": "black",
        "patch.linewidth": 1.2,
        "patch.force_edgecolor": True,

        "axes.prop_cycle": mpl.cycler(
            color=[BLUE, ORANGE]
        ),

        "lines.linewidth": 2.2,
    })