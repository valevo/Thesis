# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns


colour_palette = sns.color_palette("bright") # "dark", "deep", "colorblind"


def get_lims(xs, ys, log=False, equal_aspect=False):
    lows = min(xs), min(ys)
    highs = max(xs), max(ys)
    
    if log:
        c_low, c_high = (lambda x: x*0.5), (lambda x: x*1.5)   
    else:
        c_low, c_high = (lambda x: x*0.6), (lambda x: x*1.05)   
    
    if log:
        lows = tuple(max(10**-10, l) for l in lows)
    
    if equal_aspect:
        lims = (c_low(min(lows)),)*2, (c_high(max(highs)),)*2
        return list(zip(*lims))
    
    return (c_low(lows[0]), c_high(highs[0])),\
                (c_low(lows[1]), c_high(highs[1]))


def remove_zeros(x_vals, y_vals):
    return list(zip(*[(x, y) for x, y in zip(x_vals, y_vals) 
                      if x > 0 and y > 0]))


def hexbin_plot(xs, ys, xlbl=None, ylbl=None, log=True,
                ignore_zeros=True, cbar=True,
                set_aspect=False, lims=None, equal_aspect=False, min_y=None,
                **plt_args):
    
    if min_y is not None:
        xs, ys = list(zip(*[(x, y) for x, y in zip(xs, ys) if y >= min_y]))
    
    if not lims:
        lims = get_lims(xs, ys, log=log)
    
    
    if ignore_zeros:
        xs, ys = remove_zeros(xs, ys)

    # cmap='cividis'
    params = dict(cmap='Blues_r', edgecolors="blue", gridsize=75, mincnt=1,
                  linewidths=0.2)
    if log:
        params.update(dict(bins="log", xscale="log", yscale="log"))
    else:
        plt.ticklabel_format(style="sci", axis="both", 
                             scilimits=(0, 0))
    params.update(plt_args)    

    hb = plt.hexbin(xs, ys, **params)
    
    lims_x, lims_y = lims
    plt.xlim(lims_x)
    plt.ylim(lims_y)
    
    if cbar:
        plt.gcf().colorbar(hb)
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
        
    return lims



def plot_preds(model, xs, **plt_args):
    model_params = model.optim_params
    preds = model.predict(model_params, xs)
    
    params = dict(linestyle="--", color="red")
    
    params.update(plt_args)
    
    plt.plot(xs, preds, **params)
    




