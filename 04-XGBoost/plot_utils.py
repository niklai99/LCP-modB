import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
from sklearn import metrics

def plot_data(
    x, t, fig, 
    n_samples  = 3,
    subplot_id = 111, 
    title      = "data", 
    fontsize   = 14,
    lw         = 2,
    colors     = ("tab:blue", "tab:orange", "tab:green"),
    labels     = (0, 1, 2),
    legend     = True,
):

    ax = fig.add_subplot(subplot_id)

    for i in range(n_samples):
        ax.plot(t[i], x[i], lw=2, color=colors[i % 3], label=labels[i % 3])

    ax.set_title(title,   fontsize=fontsize+4)
    ax.set_xlabel("time", fontsize=fontsize)
    ax.tick_params(axis="x", which="major", labelsize=fontsize, length=5)



    if legend:
        custom_lines = [
            Line2D([0], [0], color="tab:blue",   lw=lw),
            Line2D([0], [0], color="tab:orange", lw=lw),
            Line2D([0], [0], color="tab:green",  lw=lw)
        ]
        custom_labels = [
            "0", "1", "2"
        ]
                
        ax.legend(custom_lines, custom_labels, fontsize=fontsize-2, title="label", title_fontsize=fontsize)

    return ax

def scatter_results(
    parameter,
    result,
    error, 
    fig, 
    subplot_id   = 111,
    ax           = None,
    label        = None,
    par_label    = None,
    metric_label = None,
    color        = "tab:blue",
    lw           = 1,
    ls           = "-",
    ms           = 12,
    mnorm        = None,
    mstyle       = "o",
    fontsize     = 18,
    legend       = True,
    title        = "plot"
):
    if ax is None:
        ax = fig.add_subplot(subplot_id)
    
    ax.set_title(title,         fontsize=fontsize+4)
    ax.set_xlabel(par_label,     fontsize=fontsize)
    ax.set_ylabel(metric_label, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    sns.scatterplot(
        x         = parameter, 
        y         = result,
        size      = np.array(ms),
        sizes     = (100, 500),
        size_norm = mnorm,
        marker    = mstyle,
        palette   = color,
        color     = color,
        label     = label,
        ax        = ax,
        legend    = False
    )

    ax.errorbar(
        parameter, 
        result,
        error,
        color = color,
        ls    = ls,
        lw    = lw,
        zorder = 0
    )

    if legend:
        ax.legend(fontsize=fontsize-4)

    return ax
