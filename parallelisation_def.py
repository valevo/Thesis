# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy.random as rand
import numpy as np

import os

from time import time, sleep

from data.WikiReader import wiki_from_pickles

if not os.path.isdir("test"):
    print("MADE DIR:", "test")
    os.makedirs("./test/")


def rand_subset_n_tokens(args):
    corpus, n_sents, i = args
    
    print("FUNCTION; ARGS:", n_sents, i)
    
    rand_inds = rand.choice(len(corpus), size=n_sents, replace=False)
    

    n_toks = sum(map(lambda s_ind: len(corpus[s_ind]), rand_inds))

    with open("test/" + str(i) + "_" + str(n_toks), "w") as handle:
        handle.write("DONE")
        
    sleep(15)

lang = "NO"
wiki = wiki_from_pickles("data/" + lang + "_pkl")
sentences = list(s for title, s_ls in wiki for s in s_ls)

print(len(sentences), 
      len(set(tuple(s) for s in sentences)),
      len([w for s in sentences for w in s]), 
      len(set(w for s in sentences for w in s)), flush=True)

print("OUTSIDE MAIN")

if __name__ == "__main__":
    
    js = list(range(10))
    ns = np.linspace(10**4, len(sentences), num=10, endpoint=False)
    
    args = [(sentences, n, j) for n in ns for j in js]
    
    
    
    with mp.Pool(4) as p:
        p.map(rand_subset_n_tokens, args)
    