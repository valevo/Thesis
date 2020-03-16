# -*- coding: utf-8 -*-

from data.reader import corpora_from_pickles
from data.corpus import Sentences

from evaluation.lex_diversity import lex_div_main
from evaluation.len_dists import len_dists_main
from evaluation.heap import heap_main



import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", type=str)
    p.add_argument("--factors", nargs="*", type=int, default=[])
    p.add_argument("--hist_lens", nargs="*", type=int, default=[])
    
    args = p.parse_args()
    return args.lang, args.factors, args.hist_lens


def get_filters(filter_dir, k, names, param_name, param_ls):
    filters_dict = {}
    
    for param in param_ls:
        all_samples = corpora_from_pickles(filter_dir, names=names)
        cur_param_filters = [Sentences(c) for name_d, c in all_samples if 
                             name_d["k"] == k and name_d[param_name] == param]
        filters_dict[param] = cur_param_filters
        
    return filters_dict

if __name__ == "__main__":
    lang, factors, hist_lens = parse_args()
    print("ARGS: ", lang, factors, hist_lens, "\n")
    d =  "results/" + lang + "/"
    results_d = d + "evaluation/"

    k = 1000000
    
    srfs = get_filters(d + "SRF/", k, ["k", "h", "i"], "h", hist_lens)
    tfs = get_filters(d + "TF/", k, ["k", "f", "i"], "f", factors)
    unis = [Sentences(c) for _, c in corpora_from_pickles(d + "UNI", names=["k", "i"])]
    
    
    # LEN DISTS
    len_dists_main(tfs, srfs, unis, results_d)
    
    print("len dists done", flush=True)
    
    # HEAP
    rng = list(range(0, k, k//100))
    heap_main(tfs, srfs, unis, rng, results_d)
    
    print("heap_main done", flush=True)
    
    # LEX DIV
    lex_div_main(tfs, srfs, unis, results_d)
    
    print("lex div done", flush=True)
    