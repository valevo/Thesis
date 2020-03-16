# -*- coding: utf-8 -*-

from data.corpus import Sentences
from data.reader import wiki_from_pickles, corpora_from_pickles

from stats.stat_functions import compute_vocab_size
from stats.mle import Heap

from jackknife.plotting import hexbin_plot, colour_palette

import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns


def get_greater_lims(lims1, lims2):
    if lims1 is None:
        return lims2
    (xlo1, xhi1), (xlo2, xhi2) = lims1[0], lims2[0]
    (ylo1, yhi1), (ylo2, yhi2)= lims1[1], lims2[1]
    return ((min(xlo1, xlo2), max(xhi1, xhi2)),
            (min(ylo1, ylo2), max(yhi1, yhi2)))


def sent_subsample(sents, n_toks):
    n_sampled = 0
    
    used = set()
    
    while n_sampled < n_toks:
        cur_ind = rand.randint(len(sents))
        if cur_ind in used:
            continue
        sampled_sent = sents.elements[cur_ind]
        
        yield sampled_sent
        n_sampled += len(sampled_sent)
        used.add(cur_ind)


def heap(corp, rng):
    vocab_sizes = []
    for ntoks in rng:
        subsample = Sentences(sent_subsample(corp, ntoks))
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
    return vocab_sizes
  
    
    
def mean_growth(samples, rng):
    print("mean_growth ", len(rng), flush=True)
    heaps = [heap(s, rng) for s in samples]
    return np.mean(heaps, axis=0)

    
def vocab_growth_plot(tf_means, srf_means, uni_mean, rng, save_dir):
    i = 0
    plot_lims = None
    for name, sample_dict in zip(["TF ", "SRF "], [tf_means, srf_means]):
        for param, mean_vs in sample_dict.items():            
            cur_plot_lims = hexbin_plot(rng, mean_vs, log=False, 
                                        ignore_zeros=False, 
                                        label=name + str(param),
                                        color=colour_palette[i],
                                        edgecolors=colour_palette[i], 
                                        cmap="Blues_r", cbar=False,
                                        gridsize=100, linewidths=1.0)
            
            plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
            i += 1
    
    cur_plot_lims = hexbin_plot(rng, uni_mean, xlbl="$n$", ylbl="$V(n)$",
                                log=False, ignore_zeros=False, label="UNIF",
                                color="black", edgecolors="black", cmap="gray",
                                cbar=True, gridsize=100, linewidths=1.0)
    plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
    
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend(loc="lower right")
    plt.savefig(save_dir + "vocab_growth_comparison.png", dpi=300)
    plt.close()
    
    
def do_mles(tf_means, srf_means, uni_mean, rng, save_dir):
    with open(save_dir + "mle_heap_all.txt", "w") as handle:
        for param, mean_vs in tf_means.items():
            means_floored = np.floor(mean_vs).astype("int")
            heap = Heap(means_floored, rng)
            heap_fit = heap.fit(start_params=np.asarray([100000.0, 0.5]), 
                                method="powell", full_output=True)    
            heap.register_fit(heap_fit)
            handle.write("\n\nTF " + str(param))
            handle.write("\n" + heap.print_result(string=True))
    
        for param, mean_vs in srf_means.items():
            means_floored = np.floor(mean_vs).astype("int")
            heap = Heap(means_floored, rng)
            heap_fit = heap.fit(start_params=np.asarray([100000.0, 0.5]), 
                                method="powell", full_output=True)    
            heap.register_fit(heap_fit)
            handle.write("\n\nSRF " + str(param))
            handle.write("\n" + heap.print_result(string=True))
            
    
        uni_mean_floored = np.floor(uni_mean).astype("int")
        heap = Heap(uni_mean_floored, rng)
        heap_fit = heap.fit(start_params=np.asarray([100000.0, 0.5]), 
                            method="powell", full_output=True)    
        heap.register_fit(heap_fit)
        handle.write("\n\nUNI\n")
        handle.write(heap.print_result(string=True))


def heap_main(tfs, srfs, unis, rng, results_d):
    factors = sorted(tfs.keys())
    hist_lens = sorted(srfs.keys())
    
    tf_means = {param: mean_growth(samples, rng) for param, samples in tfs.items()}
    srf_means = {param: mean_growth(samples, rng) for param, samples in srfs.items()}
    uni_mean = mean_growth(unis, rng)

        
    half_factors = factors[1::2]
    half_tfs = {k: tf_means[k] for k in half_factors}
    half_hist_lens = hist_lens[1::2]
    half_srfs = {k: srf_means[k] for k in half_hist_lens}

    do_mles(half_tfs, half_srfs, uni_mean, rng, save_dir=results_d)
    
    print("Heap MLEs done", flush=True)
    
    highest_two_factors = factors[-2:]
    two_tfs = {k: tf_means[k] for k in highest_two_factors}
    highest_two_hist_lens = hist_lens[-2:]
    two_srfs = {k: srf_means[k] for k in highest_two_hist_lens}
    
    vocab_growth_plot(two_tfs, two_srfs, uni_mean, rng, save_dir=results_d)
    
    print("growth plots done", flush=True)