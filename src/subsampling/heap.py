# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_vocab_size

from stats.mle import Heap

from jackknife.plotting import hexbin_plot

import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os
        
def heap(corp, rng):
    vocab_sizes = []
    for i, ntoks in enumerate(rng):
        if i % 10 == 0:
            print(i, ntoks)
        subsample = Sentences.subsample(corp, ntoks)
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
        
    return vocab_sizes

def heap_from_file(save_dir, rng_params):
    rng_params = map(str, rng_params)
    required_file_name = "vocab_growth_" + "_".join(rng_params) + ".pkl"
    print(required_file_name)
    if required_file_name in os.listdir(save_dir):
        with open(save_dir + required_file_name, "rb") as handle:
            return pickle.load(handle)
    else:
        raise FileNotFoundError


def do_mles(rng, vocab_sizes, save_dir):
    with open(save_dir + "mle_heap_point_estimates.txt", "w") as handle:
        for vs in vocab_sizes:
            heap = Heap(vs, rng)
            heap_fit = heap.fit(start_params=np.asarray([100000.0, 1.0]), 
                                        method="powell", full_output=True)    
            heap.register_fit(heap_fit)

            handle.write(heap.print_result(string=True))
            handle.write("\n")


def heap_main(wiki, rng_params, m, save_dir="./"):
    rng = list(range(*rng_params))
    
    try:
        vocab_sizes = heap_from_file(save_dir, 
                                     (rng_params[0], rng_params[1], len(rng)))
    except FileNotFoundError:
        vocab_sizes = [heap(wiki, rng) for _ in range(m)]
    
    
    do_mles(rng, vocab_sizes, save_dir)
    
    
    all_sizes = [v_n for size_ls in vocab_sizes for v_n in size_ls]
    
    print(len(all_sizes))
    
    long_rng = np.tile(rng, m)  
    
    print(len(long_rng))
    
    print(len(vocab_sizes))
    hexbin_plot(long_rng, all_sizes, xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, gridsize=100)
    
    mean_vs = np.mean(vocab_sizes, axis=0)
    
    
    hexbin_plot(rng, mean_vs, xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, label="mean",
                color="red", edgecolors="red", cmap="Reds_r", cbar=False,
                gridsize=100, linewidths=0.5)
    
    plt.legend(loc="upper left")
    plt.savefig(save_dir + "vocab_growth_" + 
                str(min(rng)) + "_" + str(max(rng)) + "_" + str(len(rng)) + ".png",
                dpi=300)
    plt.close()
    
    with open(save_dir + "vocab_growth_" + 
              str(rng_params[0]) + "_" + str(rng_params[1]) + "_" + str(len(rng)) + 
              ".pkl", "wb") as handle:
        pickle.dump(vocab_sizes, handle)


if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    save_dir = "results/ALS/jackknife/"
    
    
    m = 7
    
    rng_params = int(0), int(2e4)+1, int(2e2)
    
    heap_main(wiki, rng_params, m=7, save_dir=save_dir)
   