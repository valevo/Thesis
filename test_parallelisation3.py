# -*- coding: utf-8 -*-

from data.WikiReader import wiki_from_pickles

from Filters.UniformFilterTokens import UniformFilterTokens
from Filters.SpeakerRestrictionFilterRandomised import SpeakerRestrictionFilterRandomised

from collections import Counter
import matplotlib.pyplot as plt
import numpy.random as rand

import os
import pickle
from time import time, asctime

import multiprocessing as mp

from multiprocessing import Pool, Manager

import argparse




def get_length_matched(sents, k):
    sents_list = list(sents)
    perm_sents = rand.permutation(sents_list)

    len_matched_sents = []
    n_toks, cur_i = 0, 0
    
    while n_toks <= k:
        cur_s = perm_sents[cur_i]
        len_matched_sents.append(cur_s)
        n_toks += len(cur_s)
        cur_i += 1
    return len_matched_sents


lang = "NO"
wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
sentences = (s for title, s_ls in wiki for s in s_ls)

k = 5*10**7
length_matched_sents = get_length_matched(sentences, k)

print(len(length_matched_sents), 
      len(set(tuple(s) for s in length_matched_sents)),
      len([w for s in length_matched_sents for w in s]), 
      len(set(w for s in length_matched_sents for w in s)), flush=True)




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filter_type", type=str,
                   help="The type of filter to use. Either `SpeakerRestrictionFilterRandomised`"
                   "or `UniformFilterTokens`.")
    
    args = p.parse_args()
    
    if args.filter_type == "SpeakerRestrictionFilterRandomised":
        return SpeakerRestrictionFilterRandomised, "filters/"
    elif args.filter_type == "UniformFilterTokens":
        return UniformFilterTokens, "uniform/"
    else:
        raise NotImplementedError("No filter with name " + args.filter_type + " is implemented!")


def parallel_filtering(arg_tup):
#    dir_to_save, shared_sents, cur_j, *filter_args = args
    
    print("---- ID =", id(length_matched_sents))
    

    dir_to_save, cur_j, *filter_args = arg_tup
    print("\t", FilterCls.__name__, " with args ",cur_j,  filter_args, "\t(", asctime(), ")",flush=True)
    t0 = time()
    
    cur_seed = int.from_bytes(os.urandom(4), byteorder='little')
    print("\t(Seeded with ", cur_seed, ")", flush=True)
    rand.seed(cur_seed)
    
    cur_filter = FilterCls(length_matched_sents, *filter_args)
    
    
    filename =  dir_to_save + repr(cur_filter) + "_" + str(cur_j) + ".pkl"

    with open(filename, "wb") as handle:
        print("\tRESULT WRITTEN INTO ", filename)
        pickle.dump(cur_filter, handle) 
        
    print("\tDONE WITH", filter_args, cur_j, "(", time() - t0, "secs)", flush=True)
    


if __name__ == "__main__":           
    FilterCls, save_dir = parse_args()
    
    top_dir = "Results/" + lang + "/"
    if not os.path.isdir(top_dir+save_dir):
        print("MADE DIR ", save_dir, flush=True)
        os.makedirs(top_dir+save_dir)  
    
    
    ns = [int(1*10**6)]
    ms = [i**2 for i in range(1, 5)]
    num_samples = 10
    j_start_at = 0
    
    
    if FilterCls is SpeakerRestrictionFilterRandomised:
        params_combined = [(top_dir+save_dir, j, n, m, "count_tokens") 
                            for n in ns for m in ms 
                            for j in range(j_start_at, j_start_at+num_samples)]
    elif FilterCls is UniformFilterTokens:
        params_combined = [(top_dir+save_dir, j, n) for n in ns
                           for j in range(j_start_at, j_start_at+num_samples)]
    
    
    with Pool() as pool:
        pool.map(parallel_filtering, params_combined)
    
    print("\n\nDONE", flush=True)  
            
            

    
    