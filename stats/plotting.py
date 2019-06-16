# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import numpy.random as rand

def remove_zeros_numpy(xs, ys=None):
    if ys is not None:
        both_pos = (xs*ys).nonzero()
        return xs[both_pos], ys[both_pos]
    else:
        return xs[xs.nonzero()]
        

def remove_zeros(xs, ys=None):
    if ys is not None:
        return list(zip(*[(x, y) 
                for x, y in zip(xs, ys) 
                if x > 0 and y > 0]))
    else:
        return list(filter(None, xs))
        

def get_lims(xs, ys, log=False, equal_aspect=False):
    lows = min(xs), min(ys)
    highs = max(xs), max(ys)
    
    if log:
        c_low, c_high = (lambda x: x*0.8), (lambda x: x*1.2)   
    else:
        c_low, c_high = (lambda x: x*0.6), (lambda x: x*1.05)   
    
    if log:
        lows = tuple(max(10**-10, l) for l in lows)
    
    if equal_aspect:
        lims = (c_low(min(lows)),)*2, (c_high(max(highs)),)*2
        return list(zip(*lims))
    
#    return (.9*lows[0], .9*lows[1]),\
#                (1.1*highs[0], 1.1*highs[1])
    return (c_low(lows[0]), c_high(highs[0])),\
                (c_low(lows[1]), c_high(highs[1]))                    
                
    
#def get_lims(xs, ys, log=True):
#    if log:
#        lower = 10**(-0.2)
#    else:
#        lower = 0.
#    
##    lower = min([min(xs), min(ys)])
##    lower -= 0.1*lower
#    
#    upper = max([max(xs), max(ys)])
#    upper += 0.1*upper
#    return lower, upper



def hexbin_plot(xs, ys, xlbl=None, ylbl=None, log=True,
                ignore_zeros=True, cbar=True,
                set_aspect=False, lims=None, equal_aspect=False, **plt_args):
        
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)
    else:
        pos_xs, pos_ys = xs, ys

    params = dict(cmap='cividis', gridsize=75, mincnt=1)
    if log:
        params.update(dict(bins="log", xscale="log", yscale="log"))
    
    params.update(plt_args)    
    hb = plt.hexbin(pos_xs, pos_ys, **params)
        
    if cbar:
        plt.gcf().colorbar(hb)
            
    if lims is None:
        lims_x, lims_y = get_lims(pos_xs, pos_ys, log=log, equal_aspect=equal_aspect)
        print("LIMS: ", lims_x, lims_y)
    plt.xlim(lims_x)
    plt.ylim(lims_y)
#    plt.autoscale(tight=False)


    if set_aspect:
        plt.gca().set_aspect('equal', adjustable='datalim', anchor="C")
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
    
def multiple_hexbin_plot(ls_of_xs, ls_of_ys, labels=None, xlbl=None, ylbl=None,
                         log=True, ignore_zeros=True, **plt_args):
    
#    cmaps = ["Blues", "Greens", "Reds", "Oranges"]
    cmaps = ["spring", "summer", "autumn", "winter"]
    colours = ["blue", "red", "yellow", "brown", "orange"]
    
    cbars = [True] + [False]*(len(ls_of_xs)-1)
    
    lims = (min([get_lims(xs, ys)[0] for xs, ys in zip(ls_of_xs, ls_of_ys)]),\
            max([get_lims(xs, ys)[1] for xs, ys in zip(ls_of_xs, ls_of_ys)]))

    if labels is None:
        labels = [None]*len(ls_of_xs)


    for cur_map, col, cb, (xs, ys, l) in zip(cmaps, colours, cbars, zip(ls_of_xs, ls_of_ys, labels)):
        hexbin_plot(xs, ys, label=l, xlbl=xlbl, ylbl=ylbl, log=log, 
                    ignore_zeros=ignore_zeros, cmap="Greys", color=col, 
                    cbar=cb, set_aspect=False, lims=lims, gridsize=75, **plt_args)
                
        
        
def simple_scatterplot(xs, ys, ignore_zeros=False, log=True, xlbl=None, 
                       ylbl=None, **plt_args):
    
    params = dict(marker=".", ms=1.0)
    
    params.update(plt_args)
    
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)
    plot_f = plt.loglog if log else plt.plot
    plot_f(xs, ys, **params)
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
        
        
#%%
        
        
#plt.hexbin(spec_words.domain, spec_words.propens, label="word")
#plt.legend()
#plt.show()
        
        
def f(x=1, y=2):
    return x+y
    