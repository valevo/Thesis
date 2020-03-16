# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences

from collections import Counter
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


def number_sents(sents):
    d = dict()
    i = 0
    found = 0
    for s in sents:
        tup_s = tuple(s)
        if tup_s not in d:
            d[tup_s] = i
            i += 1
        else:
            found += 1
    
    print("duplicates: ", found)
    label_func = lambda s: d[tuple(s)] #if tuple(s) in d else -100
    return d, label_func

def number_sents_remove_empty(sents):
    d = dict()
    i = 0
    found = 0
    for s in sents:
        tup_s = tuple(w for w in s if w)
        if tup_s not in d:
            d[tup_s] = i
            i += 1
        else:
            found += 1
    
    print("duplicates: ", found)
    label_func = lambda s: (d[tuple(w for w in s if w)] 
                            if tuple(w for w in s if w) in d else -1)
    return d, label_func

def number_words(words):
    unique_words = set(words)
    d = dict(zip(unique_words, range(len(unique_words))))
    label_func = lambda w: d[w]
    return d, label_func

def jaccard(ls1, ls2, universe=None):    
    cs1, cs2 = Counter(ls1), Counter(ls2)
    
    if not universe:
        universe = cs1.keys() | cs2.keys()
        
    c_vec1, c_vec2 = [cs1[x] for x in sorted(universe)],\
                    [cs2[x] for x in sorted(universe)]
    return (sum(min(one, two) for one, two in zip(c_vec1, c_vec2))/
            sum(max(one, two) for one, two in zip(c_vec1, c_vec2)))



if __name__ == "__main__":
    n = 100000
    d = "results/ALS/"
    
    # GET UNIVERSE    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    sent_d, label_f = number_sents((s for a in wiki for s in a))
    word_d, word_label_f = number_words((w for a in wiki for s in a for w in s))
    
    
    ## LOAD CORPORA
    # SRFs    
    srf_samples = list(corpora_from_pickles(d + "SRF", names=["n", "h", "i"]))
    srf_10 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 10]
    srf_20 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 20]
    srf_30 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 30]
    #TFs
    tf_samples = list(corpora_from_pickles(d + "TF", names=["n", "f", "i"]))
    tf_50 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 50]  
    tf_100 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 100]    
    #UNIs
    uni_samples = corpora_from_pickles(d + "UNI", names=["n", "i"])
    uni = [Sentences(c) for name_d, c in uni_samples if name_d["n"] == n]
    

    # WITHIN POPULATION SIMILARITIES
    for subcorp_set, name in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"]):
        print("\n", name)
        
        labeled_subcorps = [[label_f(s) for s in subcorp.sentences() if "".join(s)] 
                            for subcorp in subcorp_set]
    
        combs = list(combinations(range(len(labeled_subcorps)), 2))
        jaccards = [jaccard(labeled_subcorps[i], labeled_subcorps[j]) for i, j in combs]
        print(np.mean(jaccards), np.var(jaccards)**.5)
        plt.hist(jaccards, bins=5, label=name)

    plt.title("Within Population Jaccard Similarities")
    plt.legend()
    plt.show()
    
    
    # ACROSS POPULATION SIMILARITIES
    cmp_pairs = [(srf_30, uni), (tf_100, uni), (srf_30, tf_100)]
    cmp_names = ["SRF30 - UNI", "TF100 - UNI", "SRF30 - TF100"]
    combs = list(combinations(range(10), 2))
    
    for (corps1, corps2), name in zip(cmp_pairs, cmp_names):

        labeled_corps1 = [[label_f(s) for s in subcorp.sentences() if "".join(s)] 
                                for subcorp in corps1]
        labeled_corps2 = [[label_f(s) for s in subcorp.sentences() if "".join(s)] 
                                for subcorp in corps2]
    
        cross_jccrds = [jaccard(labeled_corps1[i], labeled_corps2[j])
                            for i, j in combs]
            
        print(name, " JCC: ", np.mean(cross_jccrds), np.var(cross_jccrds)**.5)
    
        plt.hist(cross_jccrds, bins=5, label=name)
        
    plt.title("Across Population Jaccard Similarities")
    plt.legend()
    plt.show()
    
    
    
    # WORD LEVEL WITHIN POP. SIMILARITIES
    for subcorp_set, name in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"]):
        print("\n", name)
        
        labeled_subcorps = [[word_label_f(w) for s in subcorp.sentences() for w in s if w] 
                            for subcorp in subcorp_set]
    
        combs = list(combinations(range(len(labeled_subcorps)), 2))
        jaccards = [jaccard(labeled_subcorps[i], labeled_subcorps[j]) for i, j in combs]
        print(np.mean(jaccards), np.var(jaccards)**.5)
        plt.hist(jaccards, bins=5, label=name)

    plt.title("Word Level Within Population Jaccard SImilarities")
    plt.legend()
    plt.show()
    
    
        
    
    



    