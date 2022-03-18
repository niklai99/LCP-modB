import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, Colormap
from matplotlib.lines  import Line2D

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
    dim = x.shape[1]
    fig.tight_layout()
    ax = fig.add_subplot(subplot_id, projection='3d' if dim==3 else None,
                         alpha=0.5 if dim==3 else 1)

    ax.set_title(title, fontsize=fontsize+4)
    ax.set_xlabel("$x_1$",  fontsize=fontsize)
    ax.set_ylabel("$x_2$",  fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    if dim==3:
        ax.set_zlabel("$x_3$",  fontsize=fontsize)
        ax.scatter3D(
            xs   = x[:,0],
            ys   = x[:,1],
            zs   = x[:,2],
            c    = labels,
            cmap = palette,
            norm = Normalize(hue_norm[0], hue_norm[1])
        )
    else:
        sns.scatterplot(
            x         = x[:,0],
            y         = x[:,1],
            hue       = labels,
            palette   = palette,
            hue_norm  = hue_norm,
            edgecolor = "face",
            ax        = ax
        )

    ax.legend([], [], frameon=False)
    if legend:
        if dim==3:
            cmap      = plt.get_cmap(palette)
            lgn_e     = [Line2D([0], [0], marker='o', lw=0, color=cmap(0.)),
                     Line2D([0], [0], marker='o', lw=0, color=cmap(hue_norm[1]**-1))] if dim==3 else []
            lgn_names = ['0', '1'] if dim==3 else []
            ax.legend(lgn_e, lgn_names, loc="center left", bbox_to_anchor=(1, 0.5), 
                      fontsize=fontsize-4, title="label", title_fontsize=fontsize)
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), 
                      fontsize=fontsize-4, title="label", title_fontsize=fontsize)

    if show_boundaries:
        boundaries(ax)

    return ax

def plot_comparison(
    x, 
    y, 
    network
):
    pred = network.predict(x).reshape((-1,))
    pred_binary = pred.copy()
    pred_binary[pred <= 0.5] = 0
    pred_binary[pred >  0.5] = 1
    y_list = [y, pred, pred_binary]
    titles = ['original data', 'NN prediction', 'NN hard prediction']

    # plots
    fig = plt.figure(figsize=(15, 5))
    for i in range(3):
        plot_labeled_data(
            x, y_list[i],
            fig, 131 + i,
            titles[i],
            palette  = 'GnBu_r',
            hue_norm = (0, 1.5),
            legend   = False if i!=2 else True
        )

    plt.show()

def history_mode(
    history
):
    hist, bins = np.histogram(np.asarray(history), bins=100)
    mode_n = hist.argmax()
    return (bins[mode_n]+bins[mode_n+1])/2

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
    key        = "initializer",
    fill_box   = False,
    legend_title = "rescaling",
    max_comb   = None
):

    ax = fig.add_subplot(subplot_id)

    ax.set_title(title,           fontsize=fontsize+4)
    # ax.set_xlabel("initializers", fontsize=fontsize)
    ax.set_ylabel("accuracy",     fontsize=fontsize)
    
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    if max_comb is None:
        max_comb = len(gs_results[0].cv_results_['mean_test_score'])

    for i, grid_result in enumerate(gs_results):

        # plot only max_comb parameter combinations
        means  = grid_result.cv_results_['mean_test_score']
        idx    = np.argsort(means)[::-1]
        means  = np.array(means)[idx]
        means  = means[:max_comb]
        stds   = np.array(grid_result.cv_results_['std_test_score'])[idx]
        stds   = stds[:max_comb]
        params = np.array(grid_result.cv_results_['params'])[idx]
        params = params[:max_comb]

        # shuffle sorted arrays for a more natural visualization
        idx = np.random.permutation(len(means))
        means  = means[idx]
        stds   = stds[idx]
        params = params[idx]
 
        color = "#009cff" if colors is None else colors[labels[i]][0]
        label = None      if labels is None else labels[i]

        ax.errorbar(
            x          = np.arange(1+0.08*i, len(means)+0.08*i+1, 1),
            y          = means,
            yerr       = stds,
            linestyle  = "none",
            linewidth  = lw,
            elinewidth = 1, 
            capsize    = 2, 
            marker     = "o",
            color      = color,
            label      = label
        )

        if fill_box:
            max_idx = np.argmax(means)
            ax.fill_between(np.arange(1+0.08*i, len(means)+0.08*i+1, 1), 
                            means[max_idx]+stds[max_idx], 
                            means[max_idx]-stds[max_idx], 
                            color="#ff6300", 
                            alpha=0.3, 
                            label=f"best {key} error box")
            ax.axhline(y =means[max_idx], color="#ff6300")

    ax.set_xticks(np.arange(1,len(means)+1))
    if key is None:
        ax.set_xticklabels([p for p in params], ha='right', rotation_mode='anchor', rotation=45)
    elif len(key)>1:
        labels = [str(p[k]) for p in params for k in key ]
        ax.set_xticklabels([", ".join(labels[i:i+len(key)]) for i in range(0, len(labels), len(key))],
                            ha='right', 
                            rotation_mode='anchor', 
                            rotation=45)
    else:
        ax.set_xticklabels([p[key[0]] for p in params], ha='right', rotation_mode='anchor', rotation=45)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize-4, title=legend_title, title_fontsize=fontsize)

   
        
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
                f"{cm[i, j]}",#*100:.1f}%", 
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



def scatter_results(
    parameter,
    result,
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
    mnorm       = None,
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

    ax.plot(
        parameter, 
        result,
        color = color,
        ls    = ls,
        lw    = lw,
        zorder = 0
    )

    if legend:
        ax.legend(fontsize=fontsize-4)

    return ax
