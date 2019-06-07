# -*- coding: utf-8 -*-

from stats.CorpusStats import CorpusStats
from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite
from stats.regressions import Mandelbrot, Heap, LOWESS
from stats.plotting import hexbin_plot, simple_scatterplot


import numpy as np
import matplotlib.pyplot as plt


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
    def write(s):
        handle.write(s)
        handle.write("\n")
        
    return write
    


if __name__ == "__main__":
    lang = "ALS"
    stat_dir = "Results/" + lang + "/" + "stats/"
    param_dir = stat_dir + "params/"
    summary_dir = stat_dir + "summary/"
    
    write_stat = file_writer(summary_dir)
    
    if not os.path.isdir(summary_dir):
        print("MADE DIR ", summary_dir)
        os.makedirs(summary_dir)
    
#    stats = list(iter_stats(stat_dir))
#    
#    get_stats_of_type = lambda t: filter(lambda s: isinstance(s, t), stats)
#    
#    corpus_stat = next(get_stats_of_type(CorpusStats))
#    zipf_stats = get_stats_of_type(ImprovedSpectrum)
#    zipf_suites = get_stats_of_type(ImprovedSpectrumSuite) 
        
    
    #%% BASIC STATS
    print("\n" + lang + ": BASIC STATS", flush=True)

    corpus_stat = get_stat(0, dir_prefix=stat_dir)

    
    print(corpus_stat.n_tokens, corpus_stat.n_types)
    
    print(corpus_stat.basic_stats_as_table())
    
    print("\n" + "_"*30 + "\n")
    
    
    #%% BASIC ZIPF
    print("\n" + lang + ": BASIC ZIPF", flush=True)
    
    sentence_spec_suite = get_stat(1, open_f=ImprovedSpectrumSuite.from_pickle, 
                                   dir_prefix=stat_dir)    
    
    suite_dir = param_dir + repr(sentence_spec_suite) + "/"
    
    sentence_spec_suite.plot(plot_type="scatter_band", 
                             log=True, show=False)#, alpha=[0.2]*len(sentence_spec_suite.spectra)) # alpha=[0.1]*(len(sentence_spec_suite.spectra)-2) + [1.0] + [0.1])
        
    
    for name, spec in zip(sentence_spec_suite.names, sentence_spec_suite.spectra):
        mandelbrot = Mandelbrot.from_pickle(suite_dir+str(name), to_class=True, 
                                            frequencies=spec.propens, ranks=spec.domain)
        preds_corrected = mandelbrot.predict(mandelbrot.optim_params)
        plt.plot(mandelbrot.exog, preds_corrected, "--", color="red", 
                 label=str(mandelbrot.optim_params))
        
        
        plt.plot(mandelbrot.exog, 10**lowess.predictions, color="grey",
                 label="LWS")        
        




    plt.title("SPECTRUM SUITE")
    plt.savefig(summary_dir + sentence_spec_suite.suite_name, dpi=200)
    
    plt.close()

    
    
    #%% SPLIT LEVELS
    print("\n" + lang + ": SPLIT LEVELS", flush=True)

    all_words_ranks_freqs = get_stat(2, dir_prefix=stat_dir)
    all_articles_ranks_freqs = get_stat(3, dir_prefix=stat_dir)

    
    split_levels_suite = ImprovedSpectrumSuite(
            [all_articles_ranks_freqs,
            sentence_spec_suite.spectra[0],
            all_words_ranks_freqs], names=["articles", "sentences", "words"],
             suite_name="split_levels")
    
    
    
    
    split_levels_suite.plot(plot_type="hexbin", 
                            show=False)#, alpha=[1.0, 0.8, 0.6])
    
    
    plt.savefig(summary_dir + split_levels_suite.suite_name, dpi=200)

    plt.close()
    
    
    print("\n" + lang + "CORRELATIONS")

    
    articles_words_correl = all_articles_ranks_freqs.correlate_with(all_words_ranks_freqs, 
                                                                   compute_correl=True,
                                                                   plot_correl=True,
                                                                   this_name="$\log$ frequency from articles",
                                                                   other_name="$\log$ frequency from words",
                                                                   show=True)
    
    
    articles_sents_correl = all_articles_ranks_freqs.correlate_with(sentence_spec_suite.spectra[0], 
                                                                    compute_correl=True,
                                                                    plot_correl=False)
    
    
    print("CORRELATIONS:\t", "ARTICLE--SENTENCE\t ARTICLE--WORD")
    print("\t\t", articles_sents_correl, "\t", articles_words_correl)    
    
    
    
    
    
    
    
    
    
    all_words_ranks_freqs = get_stat(2, dir_prefix=stat_dir)
    
    mandelbrot = Mandelbrot(all_words_ranks_freqs.propens, all_words_ranks_freqs.domain)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)
    mandelbrot.register_fit(mandelbrot_fit)    
    print(lang, "WORDS", str(all_words_ranks_freqs), flush=True)
    mandelbrot.print_result()
    print(flush=True)
    
    lowess = LOWESS(all_words_ranks_freqs.propens, 
                    all_words_ranks_freqs.domain, log=True)
    lowess.to_pickle(param_dir + str(all_words_ranks_freqs) + "_lowess", remove_data=True)
    
    
    orig_val = plt.rcParams['agg.path.chunksize']
    
    plt.rcParams['agg.path.chunksize'] = 10000
    
    all_words_ranks_freqs.plot(plot_type="scatter")
    
    preds_corrected = mandelbrot.predict(mandelbrot.optim_params)
    plt.plot(mandelbrot.exog, preds_corrected, "--", color="red", 
             label=str(mandelbrot.optim_params))
    plt.plot(mandelbrot.exog, 10**lowess.predictions, color="grey",
             label="LWS")
    plt.legend()
    plt.savefig(param_dir + str(all_words_ranks_freqs) + "_plot", dpi=200)
    plt.close()
    
    mandelbrot.to_pickle(param_dir + str(all_words_ranks_freqs), remove_data=True)
    plt.rcParams['agg.path.chunksize'] = orig_val
    
    
    
    from stats.plotting import remove_zeros
    
    
    all_articles_ranks_freqs = get_stat(3, dir_prefix=stat_dir) 
    
    no_zero_domain, no_zero_propens = remove_zeros(all_articles_ranks_freqs.domain,
                                                   all_articles_ranks_freqs.propens)
    
    all_articles_ranks_freqs.domain = no_zero_domain
    all_articles_ranks_freqs.propens = no_zero_propens
    
    mandelbrot = Mandelbrot(all_articles_ranks_freqs.propens, all_articles_ranks_freqs.domain)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)
    mandelbrot.register_fit(mandelbrot_fit)    
    print(lang, "ARTICLES", str(all_articles_ranks_freqs), flush=True)
    mandelbrot.print_result()
    print()
    
    lowess = LOWESS(all_articles_ranks_freqs.propens, 
                    all_articles_ranks_freqs.domain, log=True)
    lowess.to_pickle(param_dir + str(all_articles_ranks_freqs) + "_lowess", remove_data=True)

    all_articles_ranks_freqs.plot(plot_type="hex")
    preds_corrected = mandelbrot.predict(mandelbrot.optim_params)
    plt.plot(mandelbrot.exog, preds_corrected, "--", color="red")
    plt.savefig(param_dir + str(all_articles_ranks_freqs) + "_plot", dpi=200)
    
    mandelbrot.to_pickle(param_dir + str(all_articles_ranks_freqs), remove_data=True)
    
    
    #%% RESIDUALS
    
    
    
    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    all_sents_ranks_prob = get_stat(4, dir_prefix=stat_dir)
    all_sents_freqs_freq = get_stat(5, dir_prefix=stat_dir)
    all_sents_freqs_prob = get_stat(6, dir_prefix=stat_dir)
    
    
    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)

    
    convergence_rank_suite = get_stat(7, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)
    
    convergence_freq_suite = get_stat(8, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir)
    
    
    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)

    heaps = get_stat(9, dir_prefix=stat_dir)
    
    
    hapaxes = get_stat(10, dir_prefix=stat_dir)
    

    