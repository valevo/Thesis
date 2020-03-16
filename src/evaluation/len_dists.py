# -*- coding: utf-8 -*-

from data.reader import corpora_from_pickles
from data.corpus import Sentences

from jackknife.plotting import colour_palette

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.random as rand

from scipy.stats import levene, kruskal

sns.set_palette(colour_palette)    
    
    
def get_lens(sample_ls, level="word", upper_lim=np.inf):
    level_func = Sentences.tokens if level == "word" else Sentences.sentences
    all_lens = [len(elem) for sents in sample_ls for elem in level_func(sents)
                if len(elem) < upper_lim and len(elem) > 0]
    return all_lens


def plot_dist(tfs, srfs, unis, level="word", upper_lim=np.inf, save_dir="./"):
    all_xs, all_ys = [], []
    for param, samples in sorted(tfs.items(), reverse=True):
        lens = get_lens(samples, level=level, upper_lim=upper_lim)
        all_xs.extend(lens)
        all_ys.extend(["TF " + str(param)]*len(lens))
    
    uni_lens = get_lens(unis, level=level, upper_lim=upper_lim)
    all_xs.extend(uni_lens)
    all_ys.extend(["UNIF"]*len(uni_lens))
    
    
    for param, samples in sorted(srfs.items()):
        lens = get_lens(samples, level=level, upper_lim=upper_lim)
        all_xs.extend(lens)
        all_ys.extend(["SRF " + str(param)]*len(lens))            
    
    sns.violinplot(all_xs, all_ys, cut=0, axlabel=level + " length")
    plt.savefig(save_dir + level + "_len_dists.png", dpi=300)
    plt.close()
    
    
def get_mean_std(sample_dict, level="word", upper_lim=np.inf):
    lens_dict = {param: get_lens(sample_ls, level=level, upper_lim=upper_lim)
            for param, sample_ls in sample_dict.items()}
    
    return {param: (np.mean(lens), np.median(lens), np.var(lens)**.5)
            for param, lens in lens_dict.items()}


def mean_std_table(tfs, srfs, unis, level="word", upper_lim=np.inf, save_dir="./"):
    with open(save_dir + level + "_len_means.txt", "w") as handle:
        for param, (m, med, s) in get_mean_std(tfs).items():
            handle.write("TF " + str(param) + "\t")
            handle.write(str(round(m, 3)) + "\t" + 
                         str(round(med, 3)) + "\t" + str(round(s, 3)))
            handle.write("\n")
        for param, (m, med, s) in get_mean_std(srfs).items():
            handle.write("SRF " + str(param) + "\t")
            handle.write(str(round(m, 3)) + "\t" + 
                         str(round(med, 3)) + "\t" + str(round(s, 3)))
            handle.write("\n")
        
        uni_lens = get_lens(unis, level=level, upper_lim=upper_lim)
        m, med, s = np.mean(uni_lens), np.median(uni_lens), np.var(uni_lens)**.5
        handle.write("UNIF \t")
        handle.write(str(round(m, 3)) + "\t" + 
                     str(round(med, 3)) + "\t" + str(round(s, 3)))
        
    
    


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

def len_dists_main(tfs, srfs, unis, results_d):
    word_lim = 25
    sent_lim = 35
    
    factors = sorted(tfs.keys())
    hist_lens = sorted(srfs.keys())
    

    mean_std_table(tfs, srfs, unis, level="word", upper_lim=word_lim,
                   save_dir=results_d)

    mean_std_table(tfs, srfs, unis, level="sentence", upper_lim=sent_lim,
                   save_dir=results_d)
    
    
    highest_two_factors = factors[-2:]
    two_tfs = {k: tfs[k] for k in highest_two_factors}
    highest_two_hist_lens = hist_lens[-2:]
    two_srfs = {k: srfs[k] for k in highest_two_hist_lens}
    
    plot_dist(two_tfs, two_srfs, unis, level="word", upper_lim=word_lim, 
              save_dir=results_d)
    
    plot_dist(two_tfs, two_srfs, unis, level="sentence", upper_lim=sent_lim, 
              save_dir=results_d)
    
    
    
    
#    print("Word len: Kruskal all", 
#          kruskal(get_lens(tfs[max(factors)], level="word", upper_lim=word_lim),
#                  get_lens(srfs[max(hist_lens)], level="word", upper_lim=word_lim),
#                  get_lens(unis, level="word", upper_lim=word_lim)))
#    
#    print("Word len: Levene all", 
#          levene(get_lens(tfs[max(factors)], level="word", upper_lim=word_lim),
#                  get_lens(srfs[max(hist_lens)], level="word", upper_lim=word_lim),
#                  get_lens(unis, level="word", upper_lim=word_lim)))
    