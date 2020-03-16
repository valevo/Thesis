# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle
from filtering.speaker_restriction_parallelised import filter_speaker_restrict

import multiprocessing as mp
from multiprocessing import Array
from ctypes import c_wchar_p
import os
import numpy.random as rand

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    p.add_argument("--hist_len", type=int,
                   help="The history length for the sampling constraint.")
    
    args = p.parse_args()
    return args.lang, args.n_tokens, args.hist_len



sep = "■"
def sents_to_mp_array(sents):
    sents_joined = [sep.join(s) for s in sents]
    return Array(c_wchar_p, sents_joined)


if __name__ == "__main__":
    lang, n, hist_len = parse_args()
    m = 10
 
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))
    mp_array = sents_to_mp_array((s for a in wiki for s in a))    
    

    def filter_worker(i):
        print("started ", i)
        cur_seed = int.from_bytes(os.urandom(4), byteorder='little')
        rand.seed(cur_seed)
        filtered = list(filter_speaker_restrict(mp_array, n, hist_len))
        print("filtered ", i)
    
        name = "_".join((str(n), str(hist_len), str(i)))
        corpus_to_pickle(filtered, "results/" + lang + "/SRF", name)
        
    i_ls = list(range(m))
    
    with mp.Pool(6) as p:
        p.map(filter_worker, i_ls)    
    