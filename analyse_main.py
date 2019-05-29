# -*- coding: utf-8 -*-

import os
import pickle

from utils.stat_functions import spectrum

import matplotlib.pyplot as plt

from collections import defaultdict

import numpy.random as rand


#%%

def get_filters(d):
    files = sorted(os.listdir(d))

    filters = []

    for f in files:
        with open(d + f, "rb") as handle:
            cur_filter = pickle.load(handle)
            filters.append(cur_filter)
    return filters
        
#%%
    
if __name__ == "__main__":

    d = "Results/TR/"
    filtered = get_filters(d + "filters/")
    uniform = get_filters(d + "uniform/")
    
    
#%%
    
#    for i, f in enumerate(filtered):
#        spectrum(f.tokens(), ranks=True, freqs=True, log=True, 
#                 lbl=" ".join(["F", str(f.m), str(i)]))
#        print(str((f.n, f.m, i)), end=", ")
#    
#    for i, f in enumerate(uniform):
#        spectrum(f.tokens(), ranks=True, freqs=True, log=True, 
#                 lbl=" ".join(["U", str(i)]))
#        print(str((f.n, i)), end=", ")
#    
#    plt.legend()
#    plt.show()
    
    
    
#%% m GROUPED
 
    n_m_grouped = defaultdict(list)
    
    for f in filtered:
        n_m_grouped[f.m].append(f)
    
    
    for k in sorted(n_m_grouped.keys()):
        print("\n\n=======", str(k), "=======")
        for i, f in list(enumerate(n_m_grouped[k]))[:1]:
            spectrum(f.tokens(), ranks=True, freqs=False, log=True, 
                     lbl=str(f.m) + " " + str(i))
            print(str((f.n, f.m, i)), end=", ")
        print()
        
        
    for i, f in list(enumerate(uniform))[:2]:
        spectrum(f.tokens(), ranks=True, freqs=False, log=True, 
                 lbl=" ".join(["u", str(i)]))
        print(str((f.n, i)), end=", ") 
    print()
        
    plt.legend()
    plt.savefig("TR_1000000_100.png", dpi=300)

#    f.savefig("TR_1000000.pdf", bbox_inches='tight')