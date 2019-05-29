# -*- coding: utf-8 -*-

from stats.CorpusStats import CorpusStats
from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite
from stats.plotting import hexbin_plot, simple_scatterplot




import pickle
import os


def iter_stats(stat_dir):
    for f in os.listdir(stat_dir):
        if not f.endswith(".pkl"):
            raise ValueError("FILE " + f + " DOES NOT END WITH .pkl!")
            
        with open(stat_dir + f, "rb") as handle:
            stat = pickle.load(handle)
            yield stat
            
            
        

if __name__ == "__main__":
    lang = "ALS"
    stat_dir = "Results/" + lang + "/" + "stats/"
    
    stats = list(iter_stats(stat_dir))
    
    get_stats_of_type = lambda t: filter(lambda s: isinstance(s, t), stats)
    
    corpus_stat = next(get_stats_of_type(CorpusStats))
    zipf_stats = get_stats_of_type(ImprovedSpectrum)
    zipf_suites = get_stats_of_type(ImprovedSpectrumSuite) 
        
    
    #%% BASIC STATS
    print("\n" + lang + ": BASIC STATS", flush=True)

    
    print(corpus_stat.n_tokens, corpus_stat.n_types)
    
    print(corpus_stat.basic_stats_as_table())
    
    
    #%% BASIC ZIPF
    print("\n" + lang + ": BASIC ZIPF", flush=True)
    
    
    
    
    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    
    
    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)

    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)

    