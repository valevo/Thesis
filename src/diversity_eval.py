# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences, Words

from evaluation.jaccard import number_words, number_sents, number_sents_remove_empty,\
                                jaccard 

from itertools import combinations

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

def subcorps_to_jaccard(subcorp_ls, label_f, get_func, cond):
    labeled_subcorps = [[label_f(elem) for elem in get_func(subcorp)  if cond(elem)] 
                            for subcorp in subcorp_ls]
    combs = list(combinations(range(len(subcorp_ls)), 2))
    return [jaccard(labeled_subcorps[i], labeled_subcorps[j]) for i, j in combs]


def get_within_jaccards(tfs, srfs, unis, label_f, get_func, cond):
    tf_jaccards = {param: subcorps_to_jaccard(sample_ls, label_f, get_func, cond)
                    for param, sample_ls in tfs.items()}
    srf_jaccards = {param: subcorps_to_jaccard(sample_ls, label_f, get_func, cond)
                    for param, sample_ls in srfs.items()}
    uni_jaccards = subcorps_to_jaccard(unis, label_f, get_func, cond)
    
    return tf_jaccards, srf_jaccards, uni_jaccards


def within_jaccard_plots(tfs, srfs, unis, file_name):
    hist_args = dict(alpha=1.0)
    for param, jaccard_vals in tfs.items():
        sns.distplot(jaccard_vals, label="TF " + str(param), hist_kws=hist_args)
    for param, jaccard_vals in srfs.items():
        sns.distplot(jaccard_vals, label="SRF " + str(param), hist_kws=hist_args)    
        
    sns.distplot(unis, label="UNIF", hist_kws=hist_args) 
    
    plt.legend()
    plt.savefig(file_name + ".png", dpi=300)
    plt.close()


def within_jaccard_table(tfs, srfs, unis, file_name):
    with open(file_name + ".txt", "w") as handle:
        
        for param, jaccard_vals in tfs.items():
            m, s = np.mean(jaccard_vals), np.var(jaccard_vals)**.5
            handle.write("TF " + str(param) + "\t")
            handle.write(str(m.round(3)) + "\t" + str(s.round(3)))
            handle.write("\n") 
            
        for param, jaccard_vals in srfs.items():
            m, s = np.mean(jaccard_vals), np.var(jaccard_vals)**.5
            handle.write("SRF " + str(param) + "\t")
            handle.write(str(m.round(3)) + "\t" + str(s.round(3)))
            handle.write("\n") 
            
        m, s = np.mean(unis), np.var(unis)**.5
        handle.write("UNIF \t")
        handle.write(str(m.round(3)) + "\t" + str(s.round(3)))
        handle.write("\n")
    

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

    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
    sent_d, label_f = number_sents_remove_empty((s for a in wiki for s in a))

    k = 1000000    
    srfs = get_filters(d + "SRF/", k, ["k", "h", "i"], "h", hist_lens)
    tfs = get_filters(d + "TF/", k, ["k", "f", "i"], "f", factors)
    unis = [Sentences(c) for _, c in corpora_from_pickles(d + "UNI", names=["k", "i"])]


    get_sents, cond = Sentences.sentences, lambda s: "".join(s)
    tf_jaccards, srf_jaccards, uni_jaccards = get_within_jaccards(tfs, srfs, unis,
                                                                  label_f,
                                                                  get_func=get_sents,
                                                                  cond=cond)
    
    within_jaccard_plots(tf_jaccards, srf_jaccards, uni_jaccards,
                         file_name=results_d + "jaccard_within")
    
    within_jaccard_table(tf_jaccards, srf_jaccards, uni_jaccards,
                         file_name=results_d + "jaccard_within")
    
    
    
    
    word_d, word_label_f = number_words((w for a in wiki for s in a for w in s))
    
    get_tokens, cond = Sentences.tokens, lambda w: w.strip()
    tf_jcc_word, srf_jcc_word, uni_jcc_word = get_within_jaccards(tfs, srfs, unis,
                                                                  word_label_f,
                                                                  get_func=get_tokens,
                                                                  cond=cond)
    
    within_jaccard_plots(tf_jcc_word, srf_jcc_word, uni_jcc_word,
                         file_name=results_d + "jaccard_within_tokens")
    
    within_jaccard_table(tf_jcc_word, srf_jcc_word, uni_jcc_word,
                         file_name=results_d + "jaccard_within_tokens")
    
    
    
