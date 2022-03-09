import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def flatten(t):
    return [item for sublist in t for item in sublist]


def boundaries(ax):    
    ax.plot((-20,-20),(-40,50), color="#ffae66", lw=4)
    ax.plot((-20,50), (-40,-40),color="#ffae66", lw=4)
    ax.plot((-10,50), (50,-10), color="#ffae66", lw=4)


def plot_labeled_data(
    x, 
    labels, 
    fig, 
    subplot_id      = 111, 
    title           = "plot", 
    fontsize        = 18,
    colors          = {0:"#0E0F5A", 1:"#80dfff"},
    legend          = True,
    show_boundaries = False
):

    fig.tight_layout()
    ax = fig.add_subplot(subplot_id)

    ax.set_title(title, fontsize=fontsize+4)
    ax.set_xlabel("x",  fontsize=fontsize)
    ax.set_ylabel("y",  fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)


    sns.scatterplot(
        x         = x[:,0],
        y         = x[:,1],
        hue       = labels,
        palette   = colors, 
        edgecolor = "face",
        ax        = ax,
    )

    ax.legend([],[], frameon=False)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize-4, title="label", title_fontsize=fontsize)

    if show_boundaries:
        boundaries(ax)

    return ax


def plot_loss(
    epochs,
    loss,
    fig, 
    subplot_id = 111,
    ax         = None,
    label      = None,
    color      = "tab:blue",
    fontsize   = 18,
    legend     = True,
    title      = "plot"
):

    if ax is None:
        ax = fig.add_subplot(subplot_id)
    
    ax.set_title(title,     fontsize=fontsize+4)
    ax.set_xlabel("epochs", fontsize=fontsize)
    ax.set_ylabel("loss",   fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    ax.plot(
        np.arange(0, epochs, 1), loss,
        color = color,
        lw    = 3,
        label = label
    )

    if legend:
        ax.legend(fontsize=fontsize-4)

    return ax


