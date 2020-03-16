# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                            pool_ranks, pool_freqs, reduce_pooled, merge_to_joint

from stats.mle import Mandelbrot

from jackknife.plotting import hexbin_plot

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scistats


def get_mean_relationship(sampling_level, wiki, n, m):
    subsamples1 = (sampling_level.subsample(wiki, n) for _ in range(m))
    subsamples2 = (sampling_level.subsample(wiki, n) for _ in range(m))
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    ranks_joined = pool_ranks(ranks)
    mean_ranks = reduce_pooled(ranks_joined)
    
    freqs = [compute_freqs(sub) for sub in subsamples2]
    freqs_joined = pool_freqs(freqs)
    mean_freqs = reduce_pooled(freqs_joined)
    
    return mean_ranks, mean_freqs


def do_mle(xs, ys, sampling_level, save_dir):
    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    with open(save_dir + "mle_mandelbrot_" + 
              sampling_level.__name__ + ".txt", "w") as handle:
        handle.write(mandelbrot.print_result(string=True))


def sampling_levels_main(wiki, n, m, save_dir="./"):    

    art_mean_ranks, art_mean_freqs = get_mean_relationship(Articles,
                                                 wiki, n, m)
    art_joint = merge_to_joint(art_mean_ranks, art_mean_freqs)
    art_xs, art_ys = list(zip(*sorted(art_joint.values())))
    
    hexbin_plot(art_xs, art_ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$",
                label="texts", min_y=1)
    
    do_mle(art_xs, art_ys, Articles, save_dir)
    
    
    
    sent_mean_ranks, sent_mean_freqs = get_mean_relationship(Sentences,
                                                 wiki, n, m)
    sent_joint = merge_to_joint(sent_mean_ranks, sent_mean_freqs)
    sent_xs, sent_ys = list(zip(*sorted(sent_joint.values())))
    
    do_mle(sent_xs, sent_ys, Sentences, save_dir)
    
    
    
    word_mean_ranks, word_mean_freqs = get_mean_relationship(Words,
                                                 wiki, n, m)
    word_joint = merge_to_joint(word_mean_ranks, word_mean_freqs)
    word_xs, word_ys = list(zip(*sorted(word_joint.values())))
    
    hexbin_plot(word_xs, word_ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$", 
                color="red", edgecolors="red", cmap="Reds_r",
                label="words", cbar=False, min_y=1)
    
    do_mle(word_xs, word_ys, Words, save_dir)



    plt.legend()
    plt.savefig(save_dir + "rank_freq_word_vs_article_" + str(n) + ".png",
                dpi=300)
    plt.close()

    
    
    
    freq_joint = merge_to_joint(art_mean_freqs, word_mean_freqs)
    xs, ys = list(zip(*sorted(freq_joint.values())))
    
    hexbin_plot(xs, ys, 
                xlbl=r"$\log$ $f(w)$ from texts", 
                ylbl=r"$\log$ $f(w)$ from words")    
    plt.savefig(save_dir + "freq_correl_word_vs_article_" + str(n) + ".png",
                dpi=300)
    plt.close()
    
    
    art_word_corr = scistats.spearmanr(xs, ys)
    
    
    freq_joint = merge_to_joint(art_mean_freqs, sent_mean_freqs)
    xs, ys = list(zip(*sorted(freq_joint.values())))
    
    art_sent_corr = scistats.spearmanr(xs, ys)
    
    
    freq_joint = merge_to_joint(sent_mean_freqs, word_mean_freqs)
    xs, ys = list(zip(*sorted(freq_joint.values())))
    
    sent_word_corr = scistats.spearmanr(xs, ys)

    
    with open(save_dir + "freq_sampling_level_correlations.txt", "w") as handle:
        handle.write("\t".join(["Articles-Words:", 
                                str(art_word_corr.correlation),
                                str(art_word_corr.pvalue)]))
        handle.write("\n")
        handle.write("\t".join(["Articles-Sentences:", 
                                str(art_sent_corr.correlation),
                                str(art_sent_corr.pvalue)]))
        handle.write("\n")
        handle.write("\t".join(["Sentences-Words:", 
                                str(sent_word_corr.correlation),
                                str(sent_word_corr.pvalue)]))
        handle.write("\n")        
    
    
    
    
    
    
    rank_joint = merge_to_joint(art_mean_ranks, word_mean_ranks)
    xs, ys = list(zip(*sorted(rank_joint.values())))
    
    hexbin_plot(xs, ys, 
                xlbl=r"$\log$ $r(w)$ from texts", 
                ylbl=r"$\log$ $r(w)$ from words")    
    plt.savefig(save_dir + "rank_correl_word_vs_article_" + str(n) + ".png",
                dpi=300)
    plt.close()


    art_word_corr = scistats.spearmanr(xs, ys)
    
    
    rank_joint = merge_to_joint(art_mean_ranks, sent_mean_ranks)
    xs, ys = list(zip(*sorted(rank_joint.values())))
    
    art_sent_corr = scistats.spearmanr(xs, ys)
    
    
    rank_joint = merge_to_joint(sent_mean_ranks, word_mean_ranks)
    xs, ys = list(zip(*sorted(rank_joint.values())))
    
    sent_word_corr = scistats.spearmanr(xs, ys)

    
    with open(save_dir + "rank_sampling_level_correlations.txt", "w") as handle:
        handle.write("\t".join(["Articles-Words:", 
                                str(art_word_corr.correlation),
                                str(art_word_corr.pvalue)]))
        handle.write("\n")
        handle.write("\t".join(["Articles-Sentences:", 
                                str(art_sent_corr.correlation),
                                str(art_sent_corr.pvalue)]))
        handle.write("\n")
        handle.write("\t".join(["Sentences-Words:", 
                                str(sent_word_corr.correlation),
                                str(sent_word_corr.pvalue)]))
        handle.write("\n")
    
