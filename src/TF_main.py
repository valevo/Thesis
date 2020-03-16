# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle
from data.corpus import Sentences
from stats.stat_functions import compute_freqs, merge_to_joint
from stats.entropy import typicality
from filtering.typicality import setup_filtering, filter_typicality_incremental

from operator import lt, gt

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    p.add_argument("--factor", type=float,
                   help="The factor to multiply epsilon with; determines"
                   "the degree of atypicality.")
    
    args = p.parse_args()
    return args.lang, args.n_tokens, args.factor

if __name__ == "__main__":
    lang, n, factor = parse_args()
    big_n = lambda wiki: len([w for a in wiki for s in a for w in s])*.49
    setup_m = 100
    m = 10
    
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))
    sents = [s for a in wiki for s in a]

    zipf_model, rank_dict, mean_typ, std_typ, auto_typ = setup_filtering(wiki, 
                                                                         big_n(wiki), 
                                                                         n, 
                                                                         setup_m)
    
    mean_corrected = abs(mean_typ - auto_typ)
    epsilon_f_plus = mean_corrected + std_typ*factor
    epsilon_f_minus = - epsilon_f_plus
    
    print("\nModel and Epsilon established")
    print(auto_typ, mean_typ, std_typ)
    print(epsilon_f_minus, epsilon_f_plus)
    
    
    for m_i in range(m):
        print("started ", m_i)        
        filtered = list(filter_typicality_incremental(sents, zipf_model, 
                        rank_dict, auto_typ, n, epsilon_f_minus, lt))
        filtered_freqs = compute_freqs(Sentences(filtered))
        print("filtered ", m_i, " typicality: ", 
              typicality(zipf_model, merge_to_joint(rank_dict, filtered_freqs)))

        
        name = "_".join((str(n), str(factor), str(m_i)))
        corpus_to_pickle(filtered, "results/" + lang + "/TF", name)