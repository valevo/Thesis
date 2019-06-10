# -*- coding: utf-8 -*-

from stats.CorpusStats import CorpusStats
from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite
from stats.regressions import Mandelbrot, Heap, LOWESS
from stats.plotting import hexbin_plot, simple_scatterplot, remove_zeros


import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
lg = np.log10

import pickle
import os


def iter_stats(stat_dir):
    for f in os.listdir(stat_dir):
#        if not f.endswith(".pkl"):
#            raise ValueError("FILE " + f + " DOES NOT END WITH .pkl!")
            
        with open(stat_dir + f, "rb") as handle:
            stat = pickle.load(handle)
            yield stat
            


n = 50464   

heap_rng = 100
hapax_fs = [1,2,3,4,5,None]         
stats_names = ["CorpusStats",
               "ImprovedSpectrumSuite_", # sentences_ranks_freq
               f"ImprovedSpectrum_{n}_words_ranks_freq",
               f"ImprovedSpectrum_{n}_articles_ranks_freq",
               f"ImprovedSpectrum_{n}_sentences_ranks_prob",
               f"ImprovedSpectrum_{n}_sentences_freqs_freq",
               f"ImprovedSpectrum_{n}_sentences_freqs_prob",
               "ImprovedSpectrumSuite_convergence_ranks",
               "ImprovedSpectrumSuite_convergence_freq",
               "heap_ls",
               "ImprovedHeapSuite_100_1-2-3-4-5-None"
               ]


def open_pickle(f, dir_prefix="./"):
    with open(dir_prefix + f, "rb") as handle:
        return pickle.load(handle)
    


def get_stat(stats_names_ind, open_f=open_pickle, dir_prefix="./"):
    filename = stats_names[stats_names_ind]
    if open_f is open_pickle:
         filename += ".pkl"
    stat = open_f(filename, dir_prefix=dir_prefix)
    
    return stat


def file_writer(filename):
    handle = open(filename, "w")
    def write(*args):
        for arg in args:
            arg = str(arg)
            handle.write(arg)
            handle.write("\n")
    return write
    


if __name__ == "__main__":
    lang = "ALS"
    stat_dir = "Results/" + lang + "/" + "stats/"
    param_dir = stat_dir + "params/"
    summary_dir = stat_dir + "summary/"
    if not os.path.isdir(summary_dir):
        print("MADE DIR ", summary_dir)
        os.makedirs(summary_dir)
    
    write_summary = file_writer(summary_dir + "stats.txt")

    orig_val = plt.rcParams['agg.path.chunksize']
    plt.rcParams['agg.path.chunksize'] = 10000         
    
    #    plt.style.use("ggplot")

    
    #%% BASIC STATS
    print("\n" + lang + ": BASIC STATS", flush=True)
    
    corpus_stat = get_stat(0, dir_prefix=stat_dir)

    
    print(corpus_stat.n_tokens, corpus_stat.n_types)

    write_summary("\nBASIC STATS")
    write_summary(corpus_stat.basic_stats_as_table())
    
    print("\n" + "_"*30 + "\n")
    
    
    #%% BASIC ZIPF
    print("\n" + lang + ": BASIC ZIPF", flush=True)
    write_summary("\nBASIC ZIPF")
    
    sentence_spec_suite = get_stat(1, open_f=ImprovedSpectrumSuite.from_pickle, 
                                   dir_prefix=stat_dir)
    suite_dir = param_dir + repr(sentence_spec_suite) + "/"
    
    uni_domain, uni_propens = sentence_spec_suite.unify_domains()
    
    mandelbrots = [Mandelbrot.from_pickle(suite_dir+str(name), to_class=True, 
                                          frequencies=spec.propens, ranks=spec.domain)
                    for name, spec in zip(sentence_spec_suite.names, 
                                          sentence_spec_suite.spectra)]
    mandel_preds = np.asarray([m.predict(m.optim_params, ranks=uni_domain, 
                                         n_obs=n)
                                for m, n in zip(mandelbrots, np.sum(uni_propens, axis=-1))])
    
    
    
    
    
    
    
    #%% VARIANCE IN ESTIMATES
    print("\n" + lang + ": VARIANCE IN ESTIMATES", flush=True)
    write_summary("\nVARIANCE IN ESTIMATES")
    
    # VAR IN ESTIMATES
    sentence_spec_suite.plot(plot_type="hexbin_all", cbar=False)
    sentence_spec_suite.plot(plot_type="hexbin")
    plt.savefig(summary_dir + str(sentence_spec_suite) + "_hexbins" + "_hexbin",
                dpi=200)    
    plt.close()

    
    #VAR IN RESIDUALS
#    resids = residuals(mandel_preds, uni_propens, log=False, rm_0=True)
#    concat_resids = np.concatenate(resids[:-1])
#    hexbin_plot(concat_domain, concat_resids, xlbl=None, ylbl=None, log=True, alpha=0.8, 
#                edgecolors=None, cmap="Blues_r", gridsize=75, cbar=0,
#                xscale="log", yscale="log")
#    hexbin_plot(uni_domain, resids[-1], xlbl=None, ylbl=None, log=True, alpha=0.8, 
#                edgecolors="white", cmap="Reds_r", gridsize=75, cbar=True, 
#                linewidths=0.3, xscale="log", yscale="log")
    
    sentence_spec_suite.plot(plot_type="residual_all", preds=mandel_preds, 
                             cbar=False, gridsize=100)
    randi = rand.randint(0, sentence_spec_suite.n_specs)
    sentence_spec_suite.plot(plot_type="residual", preds=mandel_preds[randi],
                             ind=randi, gridsize=100)
    
    plt.plot(uni_domain, np.ones_like(uni_domain), "--", color="red",
             linewidth=0.5)    

    x_lims = min(uni_domain)/2, max(uni_domain)*2
    y_lims = 10**(-2.5), 100
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.savefig(summary_dir + str(sentence_spec_suite) + "_hexbins" + "_hexbin" + "_resid",
                dpi=200)  
#    plt.savefig(summary_dir + str(sentence_spec_suite) + "_band" + "_hexbin" + "_resid",
#                dpi=200)    
    plt.close()
        
    
    #MANDELBROT ESTIMATE DIFFERENCES
    concat_domain = np.tile(uni_domain, sentence_spec_suite.n_specs)
    concat_propens = np.concatenate(uni_propens[:])
    hexbin_plot(concat_domain, concat_propens, xlbl=None, ylbl=None, log=True, alpha=1.0, 
                edgecolors=None, cmap="Blues_r", gridsize=75, cbar=True)
    for pred in mandel_preds:
        plt.loglog(uni_domain, pred, "--")
        
#    plt.xlim((0.9, 50))
#    plt.ylim((100, 1000))
    plt.savefig(summary_dir + str(sentence_spec_suite) + "_band" + "_fits", dpi=200)    
    plt.close()        
    
    
    #%% SPLIT LEVELS
    print("\n" + lang + ": SPLIT LEVELS", flush=True)
    write_summary("\nSPLIT LEVELS")
    all_words_ranks_freqs = get_stat(2, dir_prefix=stat_dir)
    all_articles_ranks_freqs = get_stat(3, dir_prefix=stat_dir)
    split_levels_suite = ImprovedSpectrumSuite(
            [all_articles_ranks_freqs,
            sentence_spec_suite.spectra[-1],
            all_words_ranks_freqs], names=["articles", "sentences", "words"],
             suite_name="split_levels")

    hexbin_plot(all_articles_ranks_freqs.domain, all_articles_ranks_freqs.propens, 
                xlbl=None, ylbl=None, log=True, alpha=1.0, edgecolors="white", 
                cmap="Blues_r", gridsize=75, cbar=True, linewidths=0.1, 
                xscale="log", yscale="log", color="blue", label="articles")
    
#    hexbin_plot(sentence_spec_suite.spectra[-1].domain, 
#                sentence_spec_suite.spectra[-1].propens, 
#                xlbl=None, ylbl=None, log=True, alpha=1.0, edgecolors="white", 
#                cmap="Greens_r", gridsize=75, cbar=False, linewidths=0.1, 
#                xscale="log", yscale="log")


    hexbin_plot(all_words_ranks_freqs.domain, all_words_ranks_freqs.propens, 
                xlbl=None, ylbl=None, log=True, alpha=1.0, edgecolors="white", 
                cmap="Reds_r", gridsize=75, cbar=False, linewidths=0.1, 
                xscale="log", yscale="log", color="red", label="words")
    
    mandel = Mandelbrot.from_pickle(param_dir+str(all_words_ranks_freqs), to_class=True, 
                                          frequencies=all_words_ranks_freqs.propens, 
                                          ranks=all_words_ranks_freqs.domain)
    words_preds = mandel.predict(mandel.optim_params, 
                                 ranks=all_words_ranks_freqs.domain, 
                                 n_obs=all_words_ranks_freqs.n_tokens)
    
    mandel = Mandelbrot.from_pickle(param_dir+str(all_articles_ranks_freqs), to_class=True, 
                                          frequencies=all_articles_ranks_freqs.propens, 
                                          ranks=all_articles_ranks_freqs.domain)
    articles_preds = mandel.predict(mandel.optim_params, 
                                    ranks=all_articles_ranks_freqs.domain, 
                                 n_obs=all_articles_ranks_freqs.n_tokens)
    
    plt.loglog(all_words_ranks_freqs.domain, words_preds, "--", color="blue", label="words")
    plt.loglog(all_articles_ranks_freqs.domain, articles_preds, "--", color="red", label="articles")
    
    plt.legend()
    plt.savefig(summary_dir + split_levels_suite.suite_name + "_hexbin_all", dpi=200)
    plt.close()
    
    print("\n" + lang + "CORRELATIONS")

    
    articles_words_correl = all_articles_ranks_freqs.correlate_with(all_words_ranks_freqs, 
                                                                   compute_correl=True,
                                                                   plot_correl=True,
                                                                   this_name="$\log$ frequency from articles",
                                                                   other_name="$\log$ frequency from words",
                                                                   show=False)
    plt.savefig(summary_dir + "article_word_correlation", dpi=200)
    plt.close()
    
    
    articles_sents_correl = all_articles_ranks_freqs.correlate_with(sentence_spec_suite.spectra[0], 
                                                                    compute_correl=True,
                                                                    plot_correl=False)
    
    
    write_summary("CORRELATIONS:\t ARTICLE--SENTENCE\t ARTICLE--WORD")
    write_summary("\t\t", articles_sents_correl, "\t", articles_words_correl)    
    

    
    #%% RESIDUALS
    
    
    
    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    all_sents_ranks_prob = get_stat(4, dir_prefix=stat_dir)
    all_sents_freqs_freq = get_stat(5, dir_prefix=stat_dir)
    all_sents_freqs_prob = get_stat(6, dir_prefix=stat_dir)
    
    all_sents_freqs_freq.plot(plot_type="hex")
    plt.savefig(summary_dir + str(all_sents_freqs_freq), dpi=200)
    plt.close()

    
    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)

    
    convergence_rank_suite = get_stat(7, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)
#    print(str(convergence_rank_suite))
#    cmaps_cycler = plt.cycler(cmap = ["Purple", "Blue", "Green", "Orange", "Red"])
# 
#    for kwargs, (i, spec) in zip(cmaps_cycler, enumerate(convergence_rank_suite.spectra)):
#        if i % 2 == 0:
#            cur_col = kwargs["cmap"]
#            print(cur_col)
#            a = ((i+1)/len(convergence_rank_suite.spectra))
#            hexbin_plot(spec.domain, spec.propens, 
#                    xlbl=None, ylbl=None, log=True, alpha=a, edgecolors="white", 
#                    cmap=cur_col + "s_r", gridsize=100, cbar=False, linewidths=0.3, 
#                    xscale="log", yscale="log", color=cur_col.lower(), 
#                    label=str(spec.n_tokens))
#    
#    plt.colorbar()
#    plt.legend()
#    plt.ylim((1e-4, 1e-1))
#    plt.savefig(summary_dir + str(convergence_rank_suite), dpi=300)
#    plt.close()
    
    
    from mpltools import color
    
    
    
    fig, ax = plt.subplots(ncols=1)
    
    ntoks = [spec.n_tokens for spec in convergence_rank_suite.spectra]
    map_color = color.color_mapper((min(ntoks), max(ntoks)), cmap='BuPu')


    for i, spec in enumerate(convergence_rank_suite.spectra):
        if i % 3 == 0:
            pos_x, pos_y = remove_zeros(spec.domain, spec.propens)
            a = ((i+1)/convergence_rank_suite.n_specs)
            hb = ax.hexbin(pos_x, pos_y, edgecolors="white", alpha=a,
                           gridsize=75, linewidths=0.3, xscale="log", yscale="log",
                           label=str(spec.n_tokens), mincnt=1, 
                           cmap=map_color(spec.n_tokens))
        
        

    fig.colorbar(hb)
    

    plt.ylim((1e-4, 1e-1))
    plt.savefig(summary_dir + str(convergence_rank_suite), dpi=300)
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    exit(13)
    
    
    convergence_freq_suite = get_stat(8, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)
    
    
    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)

    heaps = get_stat(9, dir_prefix=stat_dir)
    
    
    hapaxes = get_stat(10, dir_prefix=stat_dir)
    

    
    
    
    
    
    plt.rcParams['agg.path.chunksize'] = orig_val
