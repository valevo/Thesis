# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from collections import Counter

import matplotlib.pyplot as plt

from time import time

from stats.plotting import hexbin_plot, simple_scatterplot, multiple_hexbin_plot

import os
import pickle

from scipy.stats import spearmanr

lg = np.log10

def residuals(preds, true_propens, log=True, rm_0=True):
#    if rm_0:
#        preds, true_propens = [remove_zeros_numpy(p1, p2) 
#                                for p1, p2 in zip(preds, true_propens)]
        
    if log:
        log_propens = lg(true_propens)
        ratios = lg(preds) - log_propens
    else:
        ratios = np.asarray(preds)/np.asarray(true_propens)
    ratios[np.isinf(ratios)] = lg(1e-10) if log else 1e-10
    return ratios




class ImprovedHeap:    
    def __init__(self, corpus, ns, freq=None):
        self.domain = ns
        self.freq = freq if freq else "all"
                
        if freq is None:
            self.counts = list(self.estimate_all(corpus))
        else:
            self.counts = list(self.estimate(corpus, freq))
       
        
    def tokens_from(self, corpus, n):
        cur_n = 0
        for s in corpus:
            if cur_n >= n:
                break
            
            cur_n += len(s)
            yield s
            
    
    def n_tokens_from_randomised(self, corpus, n):
        corp_len = len(corpus)
        rand_inds =  rand.choice(corp_len, size=corp_len, replace=False)
        
        n_toks = 0
        
        for i in rand_inds:
            if n_toks >= n:
                break
            
            s = corpus[i]
            yield from s
            n_toks += len(s)
    
    
    def estimate_all(self, corpus):
        for i in self.domain:
            print("\n heap:", i)
            t0 = time()
            yield len(set(self.n_tokens_from_randomised(corpus, i)))
            print("heap:", time() - t0)
            
            
    def estimate(self, corpus, freq):
        for i in self.domain:
            counts = Counter(self.n_tokens_from_randomised(corpus, i))
            yield sum(map(lambda v: int(v == freq), counts.values()))

            
    def plot(self, log=True, lbl=None, show=False):
        plot_f = plt.loglog if log else plt.plot
        plot_f(self.domain, self.counts, '.', label=lbl)
        plt.legend()
        if show:
            plt.show()
            
            
    def plot(self, plot_type, show=False, preds=None, **plt_args):
        xlbl, ylbl = "number of tokens", "number of types"
        if plot_type == "hexbin":
            params = dict(edgecolors="white", linewidths=0.1, cmap="Blues_r")
            params.update(plt_args)
            
            hexbin_plot(self.domain, self.counts, xlbl=xlbl, ylbl=ylbl,
                        log=False, ignore_zeros=False, **params)
            
        elif plot_type == "residual":
            params = dict(edgecolors="white", cmap="Blues_r", linewidths=0.1)
            params.update(plt_args)
            
            resids = residuals(preds, self.counts, log=False)
            hexbin_plot(self.domain, resids, xlbl=xlbl, ylbl="error", log=False,
                        ignore_zeros=False, **params)
            
            
    @staticmethod
    def pooled_plot(heaps, plot_type, show=False, preds=None, **plt_args):
        xlbl, ylbl = "number of tokens", "number of types"
        concat_domains = np.concatenate([h.domain for h in heaps])
        concat_counts = np.concatenate([h.counts for h in heaps])
        if plot_type == "hexbin":
            params = dict(edgecolors="white", linewidths=0.1, cmap="Blues_r")
            params.update(plt_args)
            
            hexbin_plot(concat_domains, concat_counts, xlbl=xlbl, ylbl=ylbl,
                        log=False, ignore_zeros=False, **params)
            
        elif plot_type == "residual":
            params = dict(edgecolors="white", linewidths=0.1, cmap="Blues_r")
            params.update(plt_args)
            
            concat_preds = np.concatenate(preds)
            resids = residuals(concat_preds, concat_counts, log=False)
            hexbin_plot(concat_domains, resids, xlbl=xlbl, ylbl="error", log=False,
                        ignore_zeros=False, **params)

    
    def __repr__(self):
        return "_".join(["ImprovedHeap", str(len(self.domain)),
                         str(self.freq)])
        
    
    
class ImprovedHeapSuite:
    def __init__(self, corpus, ns, freqs):
        self.domain = ns
        self.freqs = freqs
        
        self.heaps = {(f if f else "all"):ImprovedHeap(corpus, ns, f) for f in freqs}
    
    def plot(self, plot_type, show=False, preds=None, ind=None, **plt_args):
        
        if not ind:
            ind = rand.randin
        
        
        if plot_type == "hexbin":
            pass
    
        
    def plot(self, lbls=None, show=False):
        if not lbls:
            lbls = list(map(str, self.freqs))
        plot_f = plt.plot
        for f, cur_heap in self.heaps.items():
            plot_f(self.domain, cur_heap.counts, '.', label=str(f))
        
        plt.legend()
        if show:
            plt.show()
            
    def __repr__(self):
        return "_".join(["ImprovedHeapSuite", str(len(self.domain)),
                         "-".join(map(str, self.freqs))])

