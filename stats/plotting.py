# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import numpy.random as rand

def remove_zeros(xs, ys=None):
    if ys:
        return list(zip(*[(x, y) 
                for x, y in zip(xs, ys) 
                if x > 0 and y > 0]))
    else:
        return list(filter(None, xs))
        
    


def hexbin_plot(xs, ys, xlbl=None, ylbl=None, log=True,
                ignore_zeros=True, cbar=True,
                set_aspect=True, **plt_args):
        
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)

    params = dict(cmap='cividis', gridsize=50, mincnt=1)
    
    if log:
        params.update(dict(bins="log", xscale="log", yscale="log"))
    
    params.update(plt_args)    
    hb = plt.hexbin(pos_xs, pos_ys, **params)
        
    if cbar:
        plt.gcf().colorbar(hb)
   
    if set_aspect:
        plt.gca().set_aspect('equal', 'datalim')
            
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
                
        
        
def simple_scatterplot(xs, ys, ignore_zeros=False, log=True, 
                       lbl=None, xlbl=None, ylbl=None, **plt_args):
    
    if ignore_zeros:
        pos_xs, pos_ys = remove_zeros(xs, ys)
    plot_f = plt.loglog if log else plt.plot
    plot_f(xs, ys, '.', label=lbl, **plt_args)
    
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
        
        
#%%
        
        
#plt.hexbin(spec_words.domain, spec_words.propens, label="word")
#plt.legend()
#plt.show()