## IMPORT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.colors import LogNorm
import time

## PLOT SURFACE

def plot_surf_3d(
    x, y, z,
    fig,
    azim = -60,
    elev =  40,
    dist =  10,
    cmap = "RdYlBu_r",
    subplot_id = 111,
    title = "plot",
):
    """
    3D plot of the surface

    Parameters:
    x, y       -> np.meshgrid(array, array) xy grids
    z          -> surf(x, y) surface values
    azim       -> 3d plot azimuth angle
    elev       -> 3d plot elevation
    dist       -> 3d plot distance view
    cmap       -> 3d plot colormap
    subplot_id -> specifies axes location in a multi-axes figure
    title      -> plot title
    """
    
    ax = fig.add_subplot(subplot_id, projection="3d")
    ax.plot_surface(
        x, y, z,
        cmap=cmap,
    )
    ax.view_init(azim=azim, elev=elev)
    ax.dist = dist

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_zlabel("z", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=18, length=5)
    
    ax.set_title(title, fontsize=18)

    return ax


def plot_surf_2d(
    x, y, z,
    fig,
    levels     = 50,
    cmap       = "RdYlBu_r",
    subplot_id = 111,
    title      = "plot",
):
    """
    2D contour plot of the surface

    Parameters:
    x, y       -> np.meshgrid(array, array) xy grids
    z          -> surf(x, y) surface values
    levels     -> number of levels 
    cmap       -> 3d plot colormap
    subplot_id -> specifies axes location in a multi-axes figure
    title      -> plot title
    """

    ax = fig.add_subplot(subplot_id)
    cont = ax.contour(
        x, y, z,
        levels=levels,
        cmap=cmap,
    )

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=18, length=5)
    
    ax.set_title(title, fontsize=18)

    return ax


def plot_surf_2d_fill(
    x, y, z,
    fig,
    levels     = 50,
    cmap       = "RdYlBu_r",
    subplot_id = 111,
    title      = "plot",
):
    """
    2D filled contour plot of the surface

    Parameters:
    x, y       -> np.meshgrid(array, array) xy grids
    z          -> surf(x, y) surface values
    levels     -> number of levels 
    cmap       -> 3d plot colormap
    subplot_id -> specifies axes location in a multi-axes figure
    title      -> plot title
    """

    ax = fig.add_subplot(subplot_id)
    cont = ax.contourf(
        x, y, z,
        levels=levels,
        cmap=cmap,
    )
    cbar = fig.colorbar(cont)

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=18, length=5)
    
    ax.set_title(title, fontsize=18)

    return ax


## OVERLAY TRAJECTORY AND STARS

def overlay_trajectory_3d(
    ax, 
    surface, 
    trajectory, 
    label="", 
    color="black", 
    lw=2,
    ms=18,
    show_i = True,
    show_f = True,
    color_i = "b",
    color_f = "r",
    label_i = "initial point",
    label_f = "convergence point"
    ):
    
    xs = trajectory[:,0]
    ys = trajectory[:,1]
    zs = surface(xs,ys)
    ax.plot(xs, ys, zs, color, lw=lw, label=label, zorder=10)
    if show_i:
        ax.plot(xs[0],  ys[0],  zs[0],  color=color_i, lw=0, marker="*", markersize=ms, markeredgecolor="k", zorder=11, label=label_i)
    if show_f:
        ax.plot(xs[-1], ys[-1], zs[-1], color=color_f, lw=0, marker="*", markersize=ms, markeredgecolor="k", zorder=11, label=label_f)
    return ax

def overlay_trajectory_2d(
    ax, 
    trajectory, 
    label="", 
    color="k", 
    lw=2, 
    ms=18,
    show_i = True,
    show_f = True,
    color_i = "b",
    color_f = "r",
    label_i = "initial point",
    label_f = "convergence point"
):

    xs=trajectory[:,0]
    ys=trajectory[:,1]
    ax.plot(xs, ys, color, lw=lw, label=label, zorder=10)
    if show_i:
        ax.plot(xs[0],  ys[0],  color=color_i, lw=0, marker="*", markersize=ms, markeredgecolor="k", zorder=11, label=label_i)
    if show_f:
        ax.plot(xs[-1], ys[-1], color=color_f, lw=0, marker="*", markersize=ms, markeredgecolor="k", zorder=11, label=label_f)

    return ax


# PLOT LOSS 

def plot_loss(
    loss, 
    fig, 
    n_epochs=None,
    axes=None, 
    subplot_id=111, 
    n_rows=None, 
    n_cols=None, 
    sub_id=None, 
    title = "plot", 
    color="tab:blue", 
    lw=3,
    label=None,
    TIME_FLAG=0,
    times=None
    ):


    if subplot_id is None:
        ax = fig.add_subplot(n_rows, n_cols, sub_id)
    elif axes is not None:
        ax = axes
    else:
        ax = fig.add_subplot(subplot_id)
    
    
    if TIME_FLAG:
        t_grid = times
        xlabel = "time (s)"
    else:
        t_grid = np.arange(0, n_epochs+1, 1)
        xlabel = "epochs"
    

    ax.plot(
        t_grid, loss, 
        color = color,
        lw    = lw,
        label = label
    )

    ax.set_yscale("log")

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("loss", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=18, length=5)
    ax.minorticks_off()

    return ax