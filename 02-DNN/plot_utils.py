import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def flatten(t):
    return [item for sublist in t for item in sublist]


def boundaries(ax, lw=4, color="#ffae66"):    
    ax.plot((-20,-20),(-40,50), color=color, lw=lw)
    ax.plot((-20,50), (-40,-40),color=color, lw=lw)
    ax.plot((-10,50), (50,-10), color=color, lw=lw)


def plot_labeled_data(
    x, 
    labels, 
    fig, 
    subplot_id      = 111, 
    title           = "plot", 
    fontsize        = 18,
    palette         = {0:"#0E0F5A", 1:"#80dfff"},
    hue_norm        = (0, 1),
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
        palette   = palette,
        hue_norm  = hue_norm,
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
    loss_label = None,
    title      = "plot"
):

    if ax is None:
        ax = fig.add_subplot(subplot_id)
    
    ax.set_title(title,     fontsize=fontsize+4)
    ax.set_xlabel("epochs", fontsize=fontsize)
    ax.set_ylabel(loss_label,   fontsize=fontsize)
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


def plot_metric(
    epochs,
    metric,
    fig, 
    subplot_id   = 111,
    ax           = None,
    label        = None,
    metric_label = None,
    color        = "tab:blue",
    fontsize     = 18,
    legend       = True,
    title        = "plot"
):

    if ax is None:
        ax = fig.add_subplot(subplot_id)
    
    ax.set_title(title,         fontsize=fontsize+4)
    ax.set_xlabel("epochs",     fontsize=fontsize)
    ax.set_ylabel(metric_label, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    ax.plot(
        np.arange(0, epochs, 1), metric,
        color = color,
        lw    = 3,
        label = label
    )

    if legend:
        ax.legend(fontsize=fontsize-4)

    return ax


def plot_weights(
    weights,
    figsize = (14, 18),
    color = "tab:blue",
    fontsize = 18,
):

    fig, ax = plt.subplots(nrows = int(len(weights)/2), ncols = 2, figsize=figsize, constrained_layout=True)

    w_list = []
    b_list = []

    for i, w in enumerate(weights):
        if i % 2 == 0:
            w_list.append(w)
        elif i % 2 != 0:
            b_list.append(w)

    for i, (w, b) in enumerate(zip(w_list, b_list)):

        ax[i][0].set_title("weights", fontsize=fontsize+4)
        ax[i][0].set_xlabel("w", fontsize=fontsize)
        ax[i][0].set_ylabel("counts", fontsize=fontsize)
        ax[i][0].tick_params(axis="both", which="major", labelsize=fontsize, length=5)

        ax[i][0].hist(flatten(w))

        ax[i][1].set_title("biases", fontsize=fontsize+4)
        ax[i][1].set_xlabel("w", fontsize=fontsize)
        ax[i][1].tick_params(axis="both", which="major", labelsize=fontsize, length=5)

        ax[i][1].hist(b)

    return
