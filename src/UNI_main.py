# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle
from data.corpus import Sentences
from stats.stat_functions import compute_freqs, merge_to_joint


import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    
    args = p.parse_args()
    return args.lang, args.n_tokens


if __name__ == "__main__":
    lang, n = parse_args()
    m = 10
    
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))    
    
    for i in range(m):
        sampled = Sentences.subsample(wiki, n)
        sampled_sents = list(sampled.sentences())
        name = "_".join((str(n), str(i)))
        corpus_to_pickle(sampled_sents, "results/" + lang + "/UNI", name)