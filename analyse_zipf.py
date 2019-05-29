#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:52:23 2019

@author: valentin
"""

import numpy as np
import numpy.random as rand

import os
import pickle
from time import asctime

import seaborn as sns

#from data.DataGenerators import DataGenerator, SentencePieceTranslator, DataGenerator2
#from LMs.models import LanguageModel

import matplotlib.pyplot as plt

from collections import Counter

from data.WikiReader import wiki_from_pickles
from Filters.UniformFilterTokens import UniformFilterTokens
from utils.stat_functions import improved_spectrum

def setup_dir(dir_name, prefix="", error_if_exists=False):
    if prefix and not os.path.isdir(prefix):
        raise FileNotFoundError("DIRECTORY " + prefix + " NOT FOUND!")
    if os.path.isdir(prefix + dir_name):
        if error_if_exists:
            raise FileExistsError("DIRECTORY " + dir_name + 
                                  (" IN " + prefix if prefix else " ") + 
                                  " EXISTS!")
    else:
        os.makedirs(prefix + dir_name)


print_flush = lambda *args: print(*args, flush=True)

if __name__ == "__main__":
    
    langs = ["EO", "FI", "ID", "KO", "NO", "TR", "VI"]
#    langs = ["TR", "VI"]
#    langs = ["ALS"]
    
    for l in langs:
        setup_dir(l + "/stats", prefix="Results/", error_if_exists=True)    
        print_flush("--"*3, " ", l, " ", "--"*3)
        
#%%
        wiki = list(wiki_from_pickles("data/" + l + "_pkl"))
        sentences = [s for title, s_ls in wiki for s in s_ls]
        
        print_flush("\tlen(sents), len(tokens), len(types)")    
        n_tokens = len([w for s in sentences for w in s])
        n_types = len(set(w for s in sentences for w in s))
        print_flush("\t", len(sentences), n_tokens, n_types)
        
#%% ENTIRE CORPUS
        
        freqs = True
        
        d, f = improved_spectrum(sentences, ranks=True, freqs=freqs, log=True, lbl=str(n_tokens))
        
        
#%% UNIFORMLY SUBSAMPLED    
        
        rng = np.linspace(0, n_tokens, num=10, dtype="int")[1:-1]
        
    #    rng = np.exp(np.arange(10, 15, 1)).astype("int")
        
#        rng = []
    
#        rng = np.linspace(0, n_tokens, num=10, dtype="int")[1:-1]
        
        rng = (np.asarray([0.5, 1, 2, 5, 10])*10**6).astype("int")
    
        for n in rng:
            subsample = UniformFilterTokens(sentences, n)
            improved_spectrum(subsample, ranks=True, freqs=freqs, log=True, lbl=str(n))
            #print_flush()
            print_flush("\t", len(subsample), n, len(subsample.types()))
            
        plt.savefig("Results/" + l + "/stats/zipf_diff_sizes_probs.png", dpi=200)
        
        plt.close()
        
        print_flush("\n\n")


##%%
#
#
#wiki = list(wiki_from_pickles("data/" + "ALS" + "_pkl"))
#sentences = [s for title, s_ls in wiki for s in s_ls]     
#
#
##%%
#
#d, f = improved_spectrum(sentences, log=True)
#
##%%
#
#
#
#d, f = list(zip(*[(r, y) for r, y in zip(d, f) if y > 0]))
#
#d, f = np.asarray(d)+1, np.asarray(f)
#
##fig, ax = plt.subplots()
##ax.set(xscale="log", yscale="log")
#
#
#k = 10000
#sns.jointplot(np.log(d[:k]), np.log(f[:k]), kind="reg", label="10000")
#
#k = 50000
#sns.jointplot(np.log(d[:k]), np.log(f[:k]), kind="reg", label="10000")
#
#plt.show()
