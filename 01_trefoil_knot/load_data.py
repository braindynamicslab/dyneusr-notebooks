"""
Trefoil knot data loader
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from pathlib import Path
from itertools import product 

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd 

from sklearn.datasets.base import Bunch

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set("paper", "white")

def gen_trefoil(size=1000, xyz_sd=0.0):
    """Generate synthetic trefoil dataset.

    Params
    -----
    :size = int (default = 1000)
     - the number of data points to use
    """
    # generate trefoil
    phi = np.linspace(0, 2*np.pi, size)
    #phi = np.r_[phi[1:], phi[0]]
    x = np.sin(phi) + 2*np.sin(2*phi)
    y = np.cos(phi) - 2*np.cos(2*phi)
    z = -np.sin(3*phi)
    if xyz_sd is not None and xyz_sd > 0.0:
        xyz_sd = np.ravel(xyz_sd)
        if xyz_sd.shape[0] == 1:
            xyz_sd = xyz_sd.repeat(3)
        x += np.random.normal(0, xyz_sd[0], size)
        y += np.random.normal(0, xyz_sd[1], size)
        z += np.random.normal(0, xyz_sd[2], size)
    # remove tiny numbers
    X = np.c_[x, y, z][:]
    X[np.abs(X) < 1e-6] = 0
    # format data bunch
    dataset = Bunch(
        X=X,
        y=-z,#phi,
        index=np.arange(size)
    )
    return dataset




def load_trefoil(size=[1000], noise=[0.0], y_bins=3, split=True, **kwargs):
    """Generate synthetic trefoil dataset.

    Params
    -----
    :sizes = list of ints (default = [1000])
     - the number of data points to use, per knot
    """
    logger = logging.getLogger(__name__)
    logger.info('load_trefoil(size=%s, noise=%s, y_bins=%s)', size,noise,y_bins)

    # make sure sizes is a list
    sizes = np.ravel(size)
    noises = np.ravel(noise)

    # generate Xs
    X = []
    y = []
    bins = []
    index = []
    cmap = []
    norm = []

    # loop over sizes
    for (size, noise) in product(sizes, noises):

        # generate X
        logging.info("...gen_trefoil(size=%s, noise=%s)", size, noise)

        data_ = gen_trefoil(size, noise)
        X_ = data_.X
        y_ = data_.y
        index_ = data_.index

        # generate meta        
        bins_ = np.linspace(y_.min(), y_.max(), num=y_bins+1)[:-1]
        y_ = np.digitize(y_, bins_)

        # define cmap, norm
        cmap_ = plt.get_cmap(kwargs.get('cmap', "tab10"))
        norm_ = mpl.colors.Normalize(y_.min(), cmap_.N)


        # store
        X.append(X_)
        y.append(y_)
        bins.append(bins_)
        index.append(index_)
        cmap.append(cmap_)
        norm.append(norm_)
   
    # make arrays
    X = np.array(X)
    y = np.array(y)
    bins = np.array(bins)
    index = np.array(index)


    # return dataset as Bunch
    dataset = Bunch(
        X=X,
        y=y,
        bins=bins,
        index=index,
        cmap=cmap,
        norm=norm,
        )
    logger.info('dataset.keys() = {}'.format(dataset.keys()))    

    # return as seperate sets ?
    if split:
        logger.info("Splitting dataset...")
        dataset = [
            Bunch(**{
                k: getattr(dataset, k)[i] for k in dir(dataset)
            })
            for i in range(len(dataset.X))
        ]
        logger.info('dataset = {}'.format([type(_) for _ in dataset]))
    return dataset



   





def scatter3d(X, y=None, c=None, s='z', ax=None, fig=None, view=(90, -90), **kwargs):
    """Plot trefoil knot.
    """   
    if c is None:
    	c = np.copy(y)
    cmap = plt.get_cmap(kwargs.get('cmap', "tab10"))
    norm = mpl.colors.Normalize(c.min(), cmap.N)
    # extract x, y, z
    x,y,z = X.T[:3]
    if s == 'z':
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        s = zbins**2

    # plot data
    fig, axes = plt.subplots(1, 3, figsize=(15,5),subplot_kw=dict(projection='3d'))

    # 3 views
    for ax_i, ax in enumerate(axes):

        if ax_i == 0:
            xcol, ycol, zcol = 0, 1, 2
        elif ax_i == 1:
            xcol, ycol, zcol = 0, 2, 1
        elif ax_i == 2:
            xcol, ycol, zcol = 1, 2, 0
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        ax.scatter(X[:,xcol], X[:,ycol], X[:,zcol], c=c, s=s, alpha=0.8, cmap=cmap, norm=norm)
        ax.set_xlabel(list('xyz')[xcol], fontweight='bold')
        ax.set_ylabel(list('xyz')[ycol], fontweight='bold')
        ax.set_zlabel(list('xyz')[zcol], fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        #if ax_i == 1:
        #    view = (view[0]+0, view[1]+90)
        #if ax_i == 2:
        #    view = (view[0]+90, view[1]+0)

        ax.set_title("view = {}".format(view))
        ax.view_init(*view)

    return axes


def tsplot(X, y=None, c=None, s='z', ax=None, fig=None, **kwargs):
    """Plot trefoil knot.
    """   
    if c is None:
        c = np.copy(y)
    cmap = plt.get_cmap(kwargs.get('cmap', "tab10"))
    norm = mpl.colors.Normalize(c.min(), cmap.N)
    # extract x, y, z
    x,y,z = X.T[:3]
    if s == 'z':
        zbins = np.linspace(z.min(), z.max(), num=10)
        zbins = np.digitize(z, zbins) 
        s = zbins**2


    # plot data
    fig, axes = plt.subplots(3, 1, figsize=(15,5))

    # subplots of each dim as time-series
    for ax_i, (col_name, col_ys) in enumerate(dict(x=x, y=y, z=z).items()):
        ax = axes[ax_i]
        ax.scatter(np.arange(len(col_ys)), col_ys, c=c, s=s, cmap=cmap, norm=norm)
        ax.set_ylabel(col_name, fontweight='bold')
        if ax_i < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('index', fontweight='bold')
    return axes



