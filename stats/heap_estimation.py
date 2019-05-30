# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from collections import Counter

import matplotlib.pyplot as plt

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
            
        
    
    def estimate(self, corpus, freq):
        for i in self.domain:
            cur_corp = self.tokens_from(rand.permutation(corpus), i)
            
            counts = Counter(w for s in cur_corp for w in s)
            
            yield sum(map(lambda v: int(v == freq), counts.values()))
            #yield len(list(filter(lambda v: v == freq, counts.values())))
            
            
    def estimate_all(self, corpus):
        for i in self.domain:
#            print(i)
            rand_inds = rand.choice(len(corpus), size=len(corpus), replace=False)
            rand_corp_iter = (corpus[i] for i in rand_inds)
            cur_corp = self.tokens_from(rand_corp_iter, i)
            cur_corp = list(cur_corp)
            yield len(set(w for s in cur_corp for w in s))
            
    def plot(self, log=True, lbl=None, show=False):
        plot_f = plt.loglog if log else plt.plot
        plot_f(self.domain, self.counts, '.', label=lbl)
        plt.legend()
        if show:
            plt.show()
            
    
    def __repr__(self):
        return "_".join(["ImprovedHeap", str(len(self.domain)),
                         str(self.freq)])
        
    
    
class ImprovedHeapSuite:    
    def __init__(self, corpus, ns, freqs):
        self.domain = ns
        self.freqs = freqs
        
        self.heaps = {(f if f else "all"):ImprovedHeap(corpus, ns, f) for f in freqs}
        
        
    def plot(self, lbls=None, show=False):
        if not lbls:
            lbls = list(map(str, self.freqs))
        plot_f = plt.plot
        for cur_heap, l in zip(self.heaps, lbls):
            plot_f(self.domain, cur_heap.counts, '.', label=l)
        
        plt.legend()
        if show:
            plt.show()
            
    def __repr__(self):
        return "_".join(["ImprovedHeapSuite", str(len(self.domain)),
                         "-".join(map(str, self.freqs))])

#%%
            
#hs = [ImprovedHeap(sentences[:50000], ns=list(range(0,50000, 500)), freq=None) 
#            for _ in range(30)]
#
#max_cs = [max([h.counts[i] for h in hs]) for i in range(len(hs[0].domain))]    
#
#min_cs = [min([h.counts[i] for h in hs]) for i in range(len(hs[0].domain))]

#mean_cs = [np.mean([h.counts[i] for h in hs]) for i in range(len(hs[0].domain))]        
##%%            
#
#plt.fill_between(hs[0].domain, min_cs, max_cs, alpha=0.2, facecolor="blue",
#                 linewidth=1.5)
#plt.plot(hs[0].domain, mean_cs, '.', color="blue", alpha=1.0, linewidth=0.)
#
#plt.savefig("stats/plots/heap_band", dpi=200)





            