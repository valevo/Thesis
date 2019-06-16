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
import argparse
            
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language", type=str,
                   help="The language to use.")
    p.add_argument("--n", type=int,
                   help="The number of tokens that appear in the files names of the stats.")
    args = p.parse_args()
    return (args.language, args.n)

#n = 50464   

heap_rng = 100
hapax_fs = hapax_fs = "-".join(map(str, [1,2,3,4,5,None]))
get_stats_names = lambda n : ["CorpusStats",
               "ImprovedSpectrumSuite_", # sentences_ranks_freq
               f"ImprovedSpectrum_{n}_words_ranks_freq",
               f"ImprovedSpectrum_{n}_articles_ranks_freq",
               f"ImprovedSpectrum_{n}_sentences_ranks_prob",
               f"ImprovedSpectrum_{n}_sentences_freqs_freq",
               f"ImprovedSpectrum_{n}_sentences_freqs_prob",
               "ImprovedSpectrumSuite_convergence_ranks",
               "ImprovedSpectrumSuite_convergence_ranks_freq",
               "ImprovedSpectrumSuite_convergence_freq",
               "heap_ls",
               f"ImprovedHeapSuite_{heap_rng}_{hapax_fs}"
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
    lang, n = parse_args()
    stats_names = get_stats_names(n)

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
#    plt.style.use("fivethirtyeight")

    
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
    mandel_preds_uni = np.asarray([m.predict(m.optim_params, ranks=uni_domain, 
                                         n_obs=n)
                                for m, n in zip(mandelbrots, np.sum(uni_propens, axis=-1))])
    mandel_preds = [m.predict(m.optim_params).reshape(-1) for m in mandelbrots]
    
    
    i = 0
    sentence_spec_suite.plot("hexbin", ind=i, cmap="Blues_r", edgecolors="blue")
    plt.plot(sentence_spec_suite.spectra[i].domain, mandel_preds[i], '--',
             color="red")
    
    plt.savefig(summary_dir + "basic_zipf",
                dpi=300)    
    plt.close()
    
    
    sentence_spec_suite.plot("residual", ind=i, preds=mandel_preds[i], 
                             unify_domains=False, cmap="Blues_r", 
                             edgecolor="blue", gridsize=75)
#    x_lims = min(uni_domain)/2, max(uni_domain)*2
#    y_lims = 10**(-2.5), 100
#    plt.xlim(x_lims)
#    plt.ylim(y_lims)
    plt.savefig(summary_dir + "basic_resid",
                dpi=300)    
    plt.close()
    
    
    #%% VARIANCE IN ESTIMATES
    print("\n" + lang + ": VARIANCE IN ESTIMATES", flush=True)
    write_summary("\nVARIANCE IN ESTIMATES")
    
    # VAR IN ESTIMATES
    sentence_spec_suite.plot(plot_type="hexbin_all", cbar=True, edgecolors="white",
                             color="blue", label="pooled")
    sentence_spec_suite.plot(plot_type="hexbin", cbar=False, edgecolors="white",
                             color="red", label="single")
    plt.legend()
    plt.savefig(summary_dir + str(sentence_spec_suite) + "hexbins" + "_hexbin",
                dpi=300)    
    plt.close()

    
    #VAR IN RESIDUALS    
    sentence_spec_suite.plot(plot_type="residual_all", preds=mandel_preds_uni, 
                             cbar=True, gridsize=150,
                             color="blue", label="pooled")
    randi = rand.randint(0, sentence_spec_suite.n_specs)
    sentence_spec_suite.plot(plot_type="residual", preds=mandel_preds_uni[randi],
                             ind=randi, gridsize=175, cbar=False,
                             color="red", label="single")
    
#    plt.plot(uni_domain, np.ones_like(uni_domain), "--", color="red",
#             linewidth=0.5)    

#    x_lims = min(uni_domain)/2, max(uni_domain)*2
#    y_lims = 10**(-2.5), 100
#    plt.xlim(x_lims)
#    plt.ylim(y_lims)
    plt.legend()
    plt.savefig(summary_dir + str(sentence_spec_suite) + "hexbins" + "_hexbin" + "_resid",
                dpi=300)    
    plt.close()
        
    
    #MANDELBROT ESTIMATE DIFFERENCES
    concat_domain = np.tile(uni_domain, sentence_spec_suite.n_specs)
    concat_propens = np.concatenate(uni_propens[:])
    sentence_spec_suite.plot(plot_type="hexbin_all", cbar=True, edgecolors="blue",
                             color="blue", label="pooled")    
    for i, pred in enumerate(mandel_preds_uni):
        plt.loglog(uni_domain, pred, "--", linewidth=0.5)
        
#    plt.xlim((0.9, 50))
#    plt.ylim((100, 1000))
    plt.savefig(summary_dir + str(sentence_spec_suite) + "hexbins" + "_fits", dpi=300)    
    plt.close()        
    
    
    #%% SPLIT LEVELS
    print("\n" + lang + ": SPLIT LEVELS", flush=True)
    write_summary("\nSPLIT LEVELS")
    all_words_ranks_freqs = get_stat(2, dir_prefix=stat_dir)
    all_articles_ranks_freqs = get_stat(3, dir_prefix=stat_dir)
    split_levels_suite = ImprovedSpectrumSuite(
            [all_articles_ranks_freqs,
#            sentence_spec_suite.spectra[-1],
            all_words_ranks_freqs], names=["articles", "sentences", "words"],
             suite_name="split_levels")
    
    split_levels_suite.plot("hexbin", ind=0, cmap="Blues_r", color="blue", label="articles")
    split_levels_suite.plot("hexbin", ind=1, cmap="Reds_r", alpha=1., cbar=False, 
                            color="red", label="words")
    
    mandel = Mandelbrot.from_pickle(param_dir+str(all_words_ranks_freqs), to_class=True, 
                                          frequencies=all_words_ranks_freqs.propens, 
                                          ranks=all_words_ranks_freqs.domain)
    words_preds = mandel.predict(mandel.optim_params)
    
    mandel = Mandelbrot.from_pickle(param_dir+str(all_articles_ranks_freqs), to_class=True, 
                                          frequencies=all_articles_ranks_freqs.propens, 
                                          ranks=all_articles_ranks_freqs.domain)
    articles_preds = mandel.predict(mandel.optim_params)
    
    plt.plot(all_articles_ranks_freqs.domain, articles_preds, "--", color="blue")
    plt.plot(all_words_ranks_freqs.domain, words_preds, "--", color="red")
    
    plt.legend()
    plt.savefig(summary_dir + split_levels_suite.suite_name + "_hexbin_all", dpi=300)
    plt.close()
    
    print("\n" + lang + ": CORRELATIONS")

    
    articles_words_correl = all_articles_ranks_freqs.correlate_with(all_words_ranks_freqs, 
                                                                   compute_correl=True,
                                                                   plot_correl=True,
                                                                   this_name="$\log$ frequency from articles",
                                                                   other_name="$\log$ frequency from words",
                                                                   show=False)
    plt.savefig(summary_dir + "article_word_correlation", dpi=300)
    plt.close()
    
    
    articles_sents_correl = all_articles_ranks_freqs.correlate_with(sentence_spec_suite.spectra[0], 
                                                                    compute_correl=True,
                                                                    plot_correl=False)
    
    
    write_summary("CORRELATIONS:\t ARTICLE--SENTENCE\t ARTICLE--WORD")
    write_summary("\t\t", articles_sents_correl, "\t", articles_words_correl)    
      
    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    all_sents_ranks_prob = get_stat(4, dir_prefix=stat_dir)
    all_sents_freqs_freq = get_stat(5, dir_prefix=stat_dir)
    all_sents_freqs_prob = get_stat(6, dir_prefix=stat_dir)
    
    all_sents_freqs_freq.plot(plot_type="hexbin", gridsize=50)
    plt.savefig(summary_dir + str(all_sents_freqs_freq), dpi=300)
    plt.close()

    
    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)


    
    convergence_rank_suite = get_stat(7, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)
    
    thinned_i = np.linspace(0, convergence_rank_suite.n_specs-1, 5).astype("int")

    
    
    colors = ["purple", "blue", "green", "orange", "red"]
    for c_i, i in enumerate(thinned_i):
            print("\tCONV ", i)
            hexbin_plot(convergence_rank_suite.spectra[i].domain, 
                convergence_rank_suite.spectra[i].propens, xlbl="log rank", ylbl="log frequency", 
                log=True, edgecolors=colors[c_i%len(colors)], cmap="Blues_r", linewidths=0.5, 
                alpha=1-(c_i/convergence_rank_suite.n_specs/2),
                cbar=False, label=str(convergence_rank_suite.spectra[i].n_tokens), color=colors[c_i%len(colors)])
    print()


#    plt.xlim((0.7, 10**4))
#    plt.ylim((1e-5, 1e-1))    
    plt.legend()
    plt.colorbar()    
    plt.savefig(summary_dir + str(convergence_rank_suite), dpi=300)
    plt.close()


    convergence_rank_freq_suite = get_stat(8, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)    

    colors = ["purple", "blue", "green", "orange", "red"]
    for c_i, i in enumerate(thinned_i):
            hexbin_plot(convergence_rank_freq_suite.spectra[i].domain, 
                convergence_rank_freq_suite.spectra[i].propens, xlbl="log rank", ylbl="log frequency", 
                log=True, edgecolors=colors[c_i%len(colors)], cmap="Blues_r", linewidths=0.5, 
                alpha=1-(i/convergence_rank_freq_suite.n_specs/2),
                cbar=False, label=str(convergence_rank_freq_suite.spectra[i].n_tokens), color=colors[c_i%len(colors)])

    plt.legend()
    plt.colorbar()

#    plt.xlim((1, 10**5))
#    plt.ylim((1, 10**5))
    plt.savefig(summary_dir + str(convergence_rank_freq_suite), dpi=300)
    plt.close()    
    
    convergence_freq_suite = get_stat(9, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)

    colors = ["purple", "blue", "green", "orange", "red"]
    for c_i, i in enumerate(thinned_i):            
        hexbin_plot(convergence_freq_suite.spectra[i].domain, 
                convergence_freq_suite.spectra[i].propens, xlbl="log frequency", ylbl="log frequency of frequency", 
                log=True, edgecolors=colors[c_i%len(colors)], cmap="Blues_r", linewidths=0.5, 
                alpha=1-(i/convergence_freq_suite.n_specs/2),
                cbar=False, label=str(convergence_rank_freq_suite.spectra[i].n_tokens), color=colors[c_i%len(colors)])

    plt.colorbar()
    plt.legend()
#    plt.xlim((0.9, 1000))
#    plt.ylim((1e-4, 1))
    plt.savefig(summary_dir + str(convergence_freq_suite), dpi=300)
    plt.close()
    
    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)

    heaps = get_stat(10, dir_prefix=stat_dir)
    
    ImprovedHeap.pooled_plot(heaps, "hexbin", cmap="Blues_r", gridsize=50,
                             color="blue", label="pooled")
    heaps[0].plot("hexbin", cmap="Reds_r", gridsize=100, cbar=False, 
         edgecolors="red", color="red", label="single")
    
    
    plt.legend(loc="upper left")
#    plt.ylim((0, 5*10**5))
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.savefig(summary_dir + "heaps", dpi=300)
    plt.close()
    
    
    
    
    
    hapaxes = get_stat(11, dir_prefix=stat_dir)

    
    
    suite_dir = param_dir + repr(hapaxes) + "/"
    
    estimates_for = {"all", 1, 2}
    
    heap_models = {n:(h, Heap.from_pickle(suite_dir+str(n), to_class=True, 
                                          ns_types=h.counts, ns_tokens=hapaxes.domain))
                    for n, h in hapaxes.heaps.items() if n in estimates_for}
    
    heap_preds = {n:hm.predict(hm.optim_params, ns_tokens=hapaxes.domain) 
                    for n, (h, hm) in heap_models.items()}   
    
    
    colors = ['orange', 'red', 'green', 'blue', 'purple', "black"]
    for i, (n, h) in enumerate(hapaxes.heaps.items()):
        h.plot("hexbin", cbar=False, edgecolors=colors[i%len(colors)],
                label=str(n), color=colors[i%len(colors)], linewidths=0.5,
                gridsize=50)
        if n in estimates_for:
            plt.plot(h.domain, heap_preds[n], '--', color="red")
    
    plt.colorbar()
    plt.legend(loc="upper left")
#    plt.ylim((0, 1.5*10**4))
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.savefig(summary_dir + "hapaxes", dpi=300)
    plt.close()
    
    
#%%  
    
    plt.rcParams['agg.path.chunksize'] = orig_val
    