# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import numpy.random as rand

def remove_zeros(xs, ys=None):
    if not ys is None:
        return list(zip(*[(x, y) 
                for x, y in zip(xs, ys) 
                if x > 0 and y > 0]))
    else:
        return list(filter(None, xs))
        
    
def get_lims(xs, ys, log=True):
    if log:
        lower = 10**(-0.2)
    else:
        lower = 0.
        
    upper = max([max(xs), max(ys)])
    upper += int(0.1*upper)
    return lower, upper



def hexbin_plot(xs, ys, xlbl=None, ylbl=None, log=True,
                ignore_zeros=True, cbar=True,
                set_aspect=True, lims=None, **plt_args):
        
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)

    params = dict(cmap='cividis', gridsize=50, mincnt=1)
#    params = dict(cmap='winter', gridsize=50, mincnt=1)
    if log:
        params.update(dict(bins="log", xscale="log", yscale="log"))
    
    params.update(plt_args)    
    hb = plt.hexbin(pos_xs, pos_ys, **params)
        
    if cbar:
        plt.gcf().colorbar(hb)
    if set_aspect:
#        pass
        plt.gca().set_aspect('equal', adjustable='datalim', anchor="C")
            
    if lims is None:
        lims = get_lims(xs, ys, log=log) 
    print("LIMS", lims)
    plt.xlim(lims)
    plt.ylim(lims)
    
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
    
def multiple_hexbin_plot(ls_of_xs, ls_of_ys, xlbl=None, ylbl=None,
                         log=True, ignore_zeros=True, **plt_args):
    
    cmaps = ["Blues", "Greens", "Reds", "Oranges"]
    cmaps = ["spring", "summer", "autumn", "winter"]
    
    colours = ["blue", "red", "yellow"]
    
    cbars = [True] + [False]*(len(ls_of_xs)-1)
    
    lims = (min([get_lims(xs, ys)[0] for xs, ys in zip(ls_of_xs, ls_of_ys)]),\
            max([get_lims(xs, ys)[1] for xs, ys in zip(ls_of_xs, ls_of_ys)]))
    
    print("LOWER:", [get_lims(xs, ys)[0] for xs, ys in zip(ls_of_xs, ls_of_ys)])
    print("UPPER:", [get_lims(xs, ys)[1] for xs, ys in zip(ls_of_xs, ls_of_ys)])
    
    for cur_map, col, cb, (xs, ys) in zip(cmaps, colours, cbars, zip(ls_of_xs, ls_of_ys)):
        hexbin_plot(xs, ys, xlbl=xlbl, ylbl=ylbl, log=log, 
                    ignore_zeros=ignore_zeros, cmap="Greys", color=col, 
                    cbar=cb, set_aspect=False, lims=lims, gridsize=75, **plt_args)
                
        
        
def simple_scatterplot(xs, ys, ignore_zeros=False, log=True, 
                       lbl=None, xlbl=None, ylbl=None, **plt_args):
    
    params = dict(marker=".", ms=1.0)
    
    params.update(plt_args)
    
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)
    plot_f = plt.loglog if log else plt.plot
    plot_f(xs, ys, label=lbl, **params)
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
        
        
#%%
        
        
#plt.hexbin(spec_words.domain, spec_words.propens, label="word")
#plt.legend()
#plt.show()