# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences

from collections import Counter
from itertools import combinations

import numpy as np
import numpy.random as rand
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
    label_func = lambda s: d[tuple(s)]
    return d, label_func


def jaccard(ls1, ls2, universe=None):    
    cs1, cs2 = Counter(ls1), Counter(ls2)
                        
    if not universe:
        universe = cs1.keys() | cs2.keys()
        
    c_vec1, c_vec2 = [cs1[x] for x in sorted(universe)], [cs2[x] for x in sorted(universe)]
    return sum(min(one, two) for one, two in zip(c_vec1, c_vec2))/sum(max(one, two) for one, two in zip(c_vec1, c_vec2))




if __name__ == "__main__":
    n = 100000
    d = "results/ALS/"
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    print("Total num sents", len([s for a in wiki for s in a]))
    
    
    srf_samples = corpora_from_pickles(d + "SRF", names=["n", "h", "i"])
    srf_30 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 30]
    
    
    tf_samples = corpora_from_pickles(d + "TF", names=["n", "f", "i"])
    tf_100 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 100]    
    
    uni_samples = corpora_from_pickles(d + "UNI", names=["n", "i"])
    uni = [Sentences(c) for name_d, c in uni_samples if name_d["n"] == n]
    
    
    for subcorp_set, name in zip([srf_30, tf_100, uni], ["SRF", "TF", "UNI"]):
        print("\n", name)
        
        shuffled_sents = rand.permutation([s for subcorp in subcorp_set 
                                        for s in subcorp.sentences()])
        
        print("total num sents", len(shuffled_sents))
        
        sent_d, label_f = number_sents(shuffled_sents)

    
        labeled_subcorps = [[label_f(s) for s in subcorp.sentences()] for subcorp in subcorp_set]
    
        combs = list(combinations(range(len(labeled_subcorps)), 2))
        jaccards = [jaccard(labeled_subcorps[i], labeled_subcorps[j]) for i, j in combs]
        print(np.mean(jaccards), np.var(jaccards)**.5)
    
        plt.hist(jaccards, bins=5, label=name)
        
        
        interect_lens = [len(set(labeled_subcorps[i]) & set(labeled_subcorps[j]))
                            for i,j in combs]
        union_lens = [len(set(labeled_subcorps[i]) | set(labeled_subcorps[j]))
                            for i,j in combs]
        
        print("\n\n")
        print(interect_lens)
        print(np.asarray(interect_lens)/np.asarray(union_lens))
        print(np.asarray(interect_lens)/len(sent_d))
        print()

        print("mean len", np.mean([len(subcorp) for subcorp in labeled_subcorps]))

        
        
#        for i, subc in enumerate(labeled_subcorps):
#            plt.plot(subc, [i]*len(subc), "|", label=str(i))
#            
#        plt.title(name)
#        plt.legend()
#        plt.show()
#        
        
        
    plt.legend()
    plt.xlim(0.0, 0.4)
    plt.show()
    
    