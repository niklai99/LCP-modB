import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

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
    figsize  = (14, 18),
    suptitle = None,
    color    = "tab:blue",
    fontsize = 18,
):

    fig, ax = plt.subplots(
        nrows              = int(len(weights)/2), 
        ncols              = 2, 
        figsize            = figsize, 
        constrained_layout = True,
        sharex             = "col"
    )
    fig.suptitle(suptitle, fontsize=fontsize+8)
    
    w_list = []
    b_list = []

    for i, w in enumerate(weights):
        if i % 2 == 0:
            w_list.append(w)
        elif i % 2 != 0:
            b_list.append(w)

    for i, (w, b) in enumerate(zip(w_list, b_list)):

        ax[i][0].set_title("weights", fontsize=fontsize+4)
        ax[i][0].set_xlabel("w",      fontsize=fontsize)
        ax[i][0].set_ylabel("counts", fontsize=fontsize)
        ax[i][0].tick_params(axis="both", which="major", labelsize=fontsize, length=5)
        ax[i][0].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i][0].xaxis.set_tick_params(labelbottom=True)

        
        ax[i][0].hist(
            flatten(w),
            histtype="bar", 
            linewidth=3,
            edgecolor="#009cff", 
            facecolor="#aadeff", 
            alpha=1, 
        )


        ax[i][1].set_title("biases", fontsize=fontsize+4)
        ax[i][1].set_xlabel("b",     fontsize=fontsize)
        ax[i][1].tick_params(axis="both", which="major", labelsize=fontsize, length=5)
        ax[i][1].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i][1].xaxis.set_tick_params(labelbottom=True)

    
        ax[i][1].hist(
            b,
            histtype="bar", 
            linewidth=3,
            edgecolor="#009cff", 
            facecolor="#aadeff", 
            alpha=1, 
        )

    ax = make_xlim_sym(ax)

    return ax


def make_xlim_sym(axes):
    
    lw, uw = [], []
    lb, ub = [], []
    for ax in axes:
        lw.append(ax[0].get_xlim()[0])
        uw.append(ax[0].get_xlim()[1])

        lb.append(ax[1].get_xlim()[0])
        ub.append(ax[1].get_xlim()[1])

    w_bounds = lw + uw
    b_bounds = lb + ub

    w_bounds = [abs(x) for x in w_bounds]
    b_bounds = [abs(x) for x in b_bounds]

    w_bound = max(w_bounds)
    b_bound = max(b_bounds)

    for ax in axes:
        ax[0].set_xlim(-w_bound, w_bound)
        ax[1].set_xlim(-b_bound, b_bound)

    return axes


def plot_gs_results(
    gs_results,
    fig,
    subplot_id = 111,
    fontsize   = 18,
    colors     = None,
    lw         = 0,
    title      = "weigth initializers grid search results",
    labels     = None,
    legend     = True,
):

    ax = fig.add_subplot(subplot_id)

    ax.set_title(title,           fontsize=fontsize+4)
    # ax.set_xlabel("initializers", fontsize=fontsize)
    ax.set_ylabel("accuracy",     fontsize=fontsize)
    
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)


    for i, grid_result in enumerate(gs_results):
        
        means  = grid_result.cv_results_['mean_test_score']
        stds   = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        ax.errorbar(
            x          = np.arange(1+0.08*i, len(means)+0.08*i+1, 1),
            y          = means,
            yerr       = stds,
            linestyle  = "none",
            linewidth  = lw,
            elinewidth = 1, 
            capsize    = 2, 
            marker     = "o",
            color      = colors[labels[i]][0],
            label      = labels[i]
        )

    ax.set_xticks(np.arange(1,len(means)+1))
    ax.set_xticklabels([p["initializer"] for p in params], ha='right', rotation_mode='anchor', rotation=45)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize-4, title="rescaling", title_fontsize=fontsize)
        
    return ax


def plot_confusion_matrix(
    cm,
    ax,
    cmap     = "GnBu_r",
    labels   = [0, 1],
    fontsize = 18,
    title    = None,
):

    mat = ax.matshow(cm, cmap=cmap)

    threshold = mat.norm(cm.max())/2.
    textcolors = ["white", "black"]
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j, 
                i, 
                f"{cm[i, j]*100:.1f}%", 
                ha       = "center", 
                va       = "center", 
                color    = textcolors[int(mat.norm(cm[i, j]) > threshold)],
                fontsize = fontsize
            )

    ax.set_title(title,      fontsize=fontsize+4)
    ax.set_xlabel("pred labels", fontsize=fontsize)
    ax.set_ylabel("true labels", fontsize=fontsize)

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)
    
    return ax