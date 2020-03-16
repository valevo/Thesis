# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles

from subsampling.sampling_levels import sampling_levels_main
from subsampling.variance import variance_main
from subsampling.convergence import convergence_main
from subsampling.heap import heap_main


import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    
    args = p.parse_args()
    return args.lang


if __name__ == "__main__":
    lang = parse_args()
    d = "results/" + lang + "/jackknife/"
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))    
    
    # max data size / 2
    big_n = int(25e6)
    small_n = int(1e6)
    n = int(1e6)
    
#    n = int(25e6)
    small_n = int(1e6)
    m = 20
    
    sampling_levels_main(wiki, small_n, m, save_dir=d)
    print("sampling levels done")
    
    variance_main(wiki, small_n, m, save_dir=d)
    print("variance done")

    rng_conv = list(range(int(5e5), int(2.5e6)+1, int(5e5)))
    convergence_main(wiki, rng_conv, m, save_dir=d)
    print("convergence done")
    
    rng_params = 0, small_n*2+1, (small_n*2)//2000
    heap_main(wiki, rng_params, m, save_dir=d)
    print("heap done")
