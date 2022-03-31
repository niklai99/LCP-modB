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


def plot_weights(
    model, layer, fig,
    title         = "model",
    fontsize      = 14, 
    lw            = 2,
    ms            = 10,
    legend        = True,
):

    c = ['r','y','c','b','m','g','k','orange']
    m = ['o','s','D','<','>','*','^','v']

        
    w = model.layers[layer].get_weights()[0]
    wT=w.T
    M=len(wT)
    b = model.layers[layer].get_weights()[1]

    axes = []

    ax1 = fig.add_subplot(121)

    ax1.axhline(0, c="k")

    for i in range(M):
        ax1.plot(wT[i][0],"-",c=c[i],marker=m[i],label=str(i),markersize=ms)

    ax1.set_title(title+': filters of layer '+str(layer), fontsize=fontsize+4)
    ax1.set_xlabel('index', fontsize=fontsize)

    if legend:
        ax1.legend(fontsize=fontsize-2)

    ax2 = fig.add_subplot(122)

    ax2.axhline(0, c="k")

    for i in range(M):
        ax2.plot((i),(b[i]),c=c[i],marker=m[i],label="filter "+str(i),markersize=ms)
    ax2.set_title(title+': bias of layer '+str(layer), fontsize=fontsize+4)
    ax2.set_xlabel('filter nr', fontsize=fontsize)
    ax2.set_xticks(np.arange(5), fontsize=fontsize)
    
    if legend:
        ax2.legend(fontsize=fontsize-2)

    return (ax1, ax2)


def plot_history(
    fit, fig,
    title         = "model",
    fontsize      = 14, 
    lw            = 2,
    legend        = True,
    ncols         = 2,
):

    ax1 = fig.add_subplot(int("1"+str(ncols)+"1"))

    ax1.plot(fit.history['accuracy'],     color="tab:blue",   ls="-",  label="train", lw=lw)
    ax1.plot(fit.history['val_accuracy'], color="tab:orange", ls="--", label="valid", lw=lw)

    ax1.set_title("accuracy",  fontsize=fontsize+4)
    ax1.set_xlabel('epoch',    fontsize=fontsize)
    ax1.set_ylabel("accuracy", fontsize=fontsize)
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=fontsize-2)

    ax2 = fig.add_subplot(int("1"+str(ncols)+"2"))

    ax2.plot(fit.history['loss'],     color="tab:blue",   ls="-",  label="train", lw=lw)
    ax2.plot(fit.history['val_loss'], color="tab:orange", ls="--", label="valid", lw=lw)

    ax2.set_title("loss",   fontsize=fontsize+4)
    ax2.set_xlabel('epoch', fontsize=fontsize)
    ax2.set_ylabel("loss",  fontsize=fontsize)
    ax2.legend(fontsize=fontsize-2)

    return (ax1, ax2)


def show_confusion_matrix(true, pred, fig, cmap='GnBu', fontsize=14, subplot_id=111):

    LABELS = ["absent","positive","negative"]

    matrix = metrics.confusion_matrix(true, pred)
    
    ax = fig.add_subplot(subplot_id)

    sns.heatmap(
        matrix,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        fmt='d',
        linecolor='white',
        linewidths=1,
        cmap=cmap,
        ax=ax
    )
    ax.set_title('Confusion Matrix',fontsize=fontsize+4)
    ax.set_ylabel('True Label',fontsize=fontsize)
    ax.set_xlabel('Predicted Label',fontsize=fontsize)

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

    