# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                        merge_to_joint, pool_ranks, pool_freqs, reduce_pooled
from stats.mle import Mandelbrot
from jackknife.plotting import hexbin_plot

import numpy as np
import matplotlib.pyplot as plt


def variance_main(wiki, n, m, save_dir="./"):
    subsamples1 = (Sentences.subsample(wiki, n) for _ in range(m))
    subsamples2 = (Sentences.subsample(wiki, n) for _ in range(m))
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    ranks_joined = pool_ranks(ranks)
    
    freqs = [compute_freqs(sub) for sub in subsamples2]
    freqs_joined = pool_freqs(freqs)


    mean_vs_pooled(ranks_joined, freqs_joined, save_dir)
    
    do_mles(ranks, freqs, save_dir)
    
    covariance_across_words(ranks_joined, freqs_joined, save_dir)


def mean_vs_pooled(rank_dist, freq_dist, save_dir):        
    all_joints = merge_to_joint(rank_dist, freq_dist)
    all_xs, all_ys = list(zip(*[(r, f) for r_ls, f_ls in all_joints.values()
                                for r, f in zip(r_ls, f_ls) if f > 0]))
    
    hexbin_plot(all_xs, all_ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$",
                min_y=1)
    
    mean_ranks = reduce_pooled(rank_dist)
    mean_freqs = reduce_pooled(freq_dist)
    
    mean_joints = merge_to_joint(mean_ranks, mean_freqs)
    mean_xs, mean_ys = list(zip(*sorted(mean_joints.values())))
    
    hexbin_plot(mean_xs, mean_ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$", 
                color="red", edgecolors="red", cmap="Reds_r", cbar=False,
                min_y=1, label="mean")
    
    
    plt.legend()
    plt.savefig(save_dir + "rank_freq_mean_vs_var.png", dpi=300)
    plt.close()


def do_mles(ranks, freqs, save_dir):
    with open(save_dir + "mle_mandelbrot_point_estimates.txt", "w") as handle:
        for r_dict, f_dict in zip(ranks, freqs):
            joints = merge_to_joint(r_dict, f_dict)
            xs, ys = list(zip(*sorted(joints.values())))
            
            mandelbrot = Mandelbrot(ys, xs)
            mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                            method="powell", full_output=True)    
            mandelbrot.register_fit(mandelbrot_fit)
#            mandelbrot.print_result()
            
            handle.write(mandelbrot.print_result(string=True))
            handle.write("\n")
            
            
def covariance_across_words(rank_dist, freq_dist, save_dir):
    joints = merge_to_joint(rank_dist, freq_dist)
    mean_ranks = reduce_pooled(rank_dist)
    
    equalize_len = lambda ls1, ls2: (ls1[:min(len(ls1), len(ls2))], 
                                         ls2[:min(len(ls1), len(ls2))])
    
    cov_dict = {w: np.cov(*equalize_len(r_ls, f_ls)) 
                for w, (r_ls, f_ls) in joints.items()}
    
    fano_factor_dict = {w: cov_mat[0][1]/mean_ranks[w]
                for w, cov_mat in cov_dict.items()}
    
    
    words_sorted = [(w, r) for w, r in sorted(mean_ranks.items(), 
                                                 key=lambda tup: tup[1])]
    
    xs, ys = list(zip(*[(r, fano_factor_dict[w]) for w, r in words_sorted 
                        if w in cov_dict]))
    
    hexbin_plot(xs, ys, log=False, xscale="log", bins="log",
                xlbl="$\overline{r}(w)$", ylbl="$D(w)$",
                ignore_zeros=False, gridsize=100)
    
#    plt.legend()
    plt.savefig(save_dir + "dispersion.png", dpi=300)
    plt.close()



def single_word_cov(rank_dist, freq_dist, r_w, save_dir):
    pass