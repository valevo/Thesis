# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                compute_normalised_freqs, merge_to_joint,\
                pool_ranks, pool_freqs, reduce_pooled

from stats.mle import Mandelbrot

from jackknife.plotting import hexbin_plot, colour_palette

import numpy as np
import matplotlib.pyplot as plt

#def format_scientific(n):
#    formatted_s = "%.1e" % n
#    formatted_s = formatted_s.replace("+0", "")
#    formatted_s = formatted_s.replace("+", "")
#    return formatted_s


def format_scientific(n):
    in_millions = n/int(10**6)
    return "$" + str(in_millions) + "\cdot" + "10^6$"


def get_mean_relationship(wiki, n, m, freq_func):
    subsamples1 = (Sentences.subsample(wiki, n) for _ in range(m))
    subsamples2 = (Sentences.subsample(wiki, n) for _ in range(m))
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    ranks_joined = pool_ranks(ranks)
    mean_ranks = reduce_pooled(ranks_joined)
    
    freqs = [freq_func(sub) for sub in subsamples2]
    freqs_joined = pool_freqs(freqs)
    mean_freqs = reduce_pooled(freqs_joined)
    
    return mean_ranks, mean_freqs

def do_mle(xs, ys, n, file_handle):
    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
#    mandelbrot.print_result()
    file_handle.write(str(n))
    file_handle.write("\n")
    file_handle.write(mandelbrot.print_result(string=True))
    file_handle.write("\n\n")


def convergence_main(wiki, rng, m, save_dir="./"):
    handle = open(save_dir + "mle_mandelbrot_convergence_" + 
                  "_".join(map(str, rng)) + ".txt", "w")
    for i, n in enumerate(rng):        
        mean_ranks, mean_freqs = get_mean_relationship(wiki, n, m, compute_freqs)
        joints = merge_to_joint(mean_ranks, mean_freqs)
        xs, ys = list(zip(*joints.values()))

        hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$",
                    edgecolors=colour_palette[i], color=colour_palette[i],
                    label=format_scientific(n), 
                    alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False), min_y=1)
        
        do_mle(xs, ys, n, handle)
            
    handle.close()
    
    plt.legend()
    plt.savefig(save_dir + "convergence_" + "_".join(map(str, rng)) + ".png",
                dpi=300)
    plt.close()
    
    
    for i, n in enumerate(rng):
        mean_ranks, mean_freqs = get_mean_relationship(wiki, n, m,
                                                       compute_normalised_freqs)
        joints = merge_to_joint(mean_ranks, mean_freqs)
        xs, ys = list(zip(*joints.values()))
                        
        hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $P(w)$",
                    edgecolors=colour_palette[i], color=colour_palette[i],
                    label=format_scientific(n), alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False), min_y=1/n)
    
    plt.legend()
    plt.savefig(save_dir + "convergence_probs_" + "_".join(map(str, rng)) + ".png",
                dpi=300)
    plt.close()


if __name__ == "__main__":
    d = "results/ALS/jackknife/"
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))    
    rng = list(range(int(5e5), int(2.5e6)+1, int(5e5)))
    
    m = 5
    
    convergence_main(wiki, rng, m, save_dir=d)


    