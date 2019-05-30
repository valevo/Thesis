# -*- coding: utf-8 -*-

import numpy as np

from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite,\
                            plt, rand, Counter
                            
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite

from stats.CorpusStats import CorpusStats

from data.WikiReader import wiki_from_pickles


import pickle
import os
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language", type=str,
                   help="The language to use.")
    args = p.parse_args()
    return args.language

def get_length_matched(corpus, k, sents=True):
    corp_list = list(corpus)
    rand_inds = rand.choice(len(corpus), size=len(corpus), replace=False)
    rand_corp_iter = (corp_list[i] for i in rand_inds)
    tok_count_f = (lambda a: len([w for s in a for w in s])) if not sents else len

    n_toks = 0
    for elem in rand_corp_iter:
        if n_toks >= k:
            break
            
        n_toks += tok_count_f(elem)
        yield elem


if __name__ == "__main__":
    lang = parse_args()
    
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
    articles = (a for title, a in wiki)

    n = 5*10**7
    length_matched_articles = list(get_length_matched(articles, n, sents=False))
    length_matched_sents = [s for a in length_matched_articles for s in a]

    n = min(n, len([w for s in length_matched_sents for w in s]))

    print(len(length_matched_sents), 
          len(set(tuple(s) for s in length_matched_sents)),
          len([w for s in length_matched_sents for w in s]), 
          len(set(w for s in length_matched_sents for w in s)), flush=True)
    
    save_dir = "stats/"
    
    top_dir = "Results/" + lang + "/"
    if not os.path.isdir(top_dir+save_dir):
        print(lang + ": MADE DIR ", save_dir, flush=True)
        os.makedirs(top_dir+save_dir) 
    
        
    #%% BASIC STATS
    print("\n" + lang + ": BASIC STATS", flush=True)

    corpus_stats = CorpusStats(length_matched_articles, sent_len_dist=True)
    
    
    #%% BASIC ZIPF
    print("\n" + lang + ": BASIC ZIPF", flush=True)
    
    rng = range(30)
    
    all_sents_ranks_freqs = tuple(ImprovedSpectrum(length_matched_sents,
                                       split_level="sentences", 
                                       ranks=True, freqs=True)
                            for _ in rng)
        
    all_sent_rank_freq_suite = ImprovedSpectrumSuite(all_sents_ranks_freqs,
                                                     names=list(rng))
    
    print(lang + ": Estimated SENTS RANKS FREQS suite", flush=True)
    
    all_words_ranks_freqs = ImprovedSpectrum(length_matched_sents,
                                             split_level="words",
                                             ranks=True, freqs=True)
    all_articles_ranks_freqs = ImprovedSpectrum(length_matched_articles,
                                             split_level="articles",
                                             ranks=True, freqs=True)
    
    print(lang + ": Estimated {WORDS, ARTICLES} RANKS FREQS", flush=True)
    
    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    all_sents_ranks_probs = ImprovedSpectrum(length_matched_sents,
                                             split_level="sentences",
                                             ranks=True, freqs=False)
    
    print(lang + ": Estimated SENTS RANKS PROBS", flush=True)
    
    all_sents_freq_freqs = ImprovedSpectrum(length_matched_sents,
                                             split_level="sentences",
                                             ranks=False, freqs=True)
    
    print(lang + ": Estimated SENTS FREQ FREQS", flush=True)
    
    
    all_sents_freq_probs = ImprovedSpectrum(length_matched_sents,
                                             split_level="sentences",
                                             ranks=False, freqs=False)
    
    print(lang + ": Estimated SENTS FREQ FREQS", flush=True)
    
    
    
    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)
    
    
    rank_specs = []
    rng = (np.linspace(0.02, 1.0, 30)*n).astype("int")
    
    for m in rng:
        print(m)
        cur_sents = get_length_matched(length_matched_sents, m, sents=True)
        cur_spec = ImprovedSpectrum(list(cur_sents),
                                             split_level="sentences",
                                             ranks=True, freqs=False)
        rank_specs.append(cur_spec)        
        
    rank_suite = ImprovedSpectrumSuite(tuple(rank_specs),
                                       names=list(map(str, rng)),
                                       suite_name="convergence_ranks")
    
    
    print(lang + ": Estimated SENT RANKS PROBS convergence", flush=True)
    
    
    freq_specs = []
    rng = (np.linspace(0.02, 1.0, 30)*n).astype("int")
    
    for m in rng:
        print(m)
        cur_sents = get_length_matched(length_matched_sents, m, sents=True)
        cur_spec = ImprovedSpectrum(list(cur_sents),
                                             split_level="sentences",
                                             ranks=False, freqs=False)
        freq_specs.append(cur_spec)        
                
    freq_suite = ImprovedSpectrumSuite(tuple(freq_specs),
                                       names=list(map(str, rng)),
                                       suite_name="convergence_freq")
    
    print(lang + ": Estimated SENT FREQ PROBS convergence", flush=True)
    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)
    
    rng = (np.linspace(0.0, 1.0, 1000)*n).astype("int")
    
    
    heaps = tuple(ImprovedHeap(length_matched_sents, ns=rng, freq=None)
                    for _ in range(20))
    
    print(lang + ": Estimated HEAP suite")
    
    heap_hapaxes = ImprovedHeapSuite(length_matched_sents, ns=rng, 
                                     freqs=(1,2,3,4,5,None))
    
    
    print(lang + ": Estimated HEAP HAPAXES", flush=True)
    
    
    #%% SAVE
    
    to_pkl = [corpus_stats,
              all_words_ranks_freqs,
              all_articles_ranks_freqs,
              all_sents_ranks_probs,
              all_sents_freq_freqs,
              all_sents_freq_probs,
              heap_hapaxes]
    
    for obj in to_pkl:
        with open(top_dir+save_dir + str(obj) + ".pkl", "wb") as handle:
            pickle.dump(obj, handle)
    
    
    
    
    suites_to_pkl = [all_sents_ranks_freqs,
                    rank_specs,
                    freq_specs,]

    for suite in suites_to_pkl:
        suite.to_pickle(dir_prefix=top_dir+save_dir)
    

    
    
    with open(top_dir+save_dir + "heap_ls.pkl", "wb") as handle:
            pickle.dump(obj, handle)
    
    
    print(lang + ": PICKLING DONE", flush=True)
    
    
    
    
    
    