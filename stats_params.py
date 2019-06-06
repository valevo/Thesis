# -*- coding: utf-8 -*-

import numpy as np

from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite,\
                            plt, rand, Counter     
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite
from stats.regressions import Mandelbrot, Heap, LOWESS


from time import time
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
hapax_fs = "-".join(map(str, [1,2,3,4,5,None]))
get_stats_names = lambda n : ["CorpusStats",
               "ImprovedSpectrumSuite_", # sentences_ranks_freq
               f"ImprovedSpectrum_{n}_words_ranks_freq",
               f"ImprovedSpectrum_{n}_articles_ranks_freq",
               f"ImprovedSpectrum_{n}_sentences_ranks_prob",
               f"ImprovedSpectrum_{n}_sentences_freqs_freq",
               f"ImprovedSpectrum_{n}_sentences_freqs_prob",
               "ImprovedSpectrumSuite_convergence_ranks",
               "ImprovedSpectrumSuite_convergence_freq",
               "heap_ls",
               f"ImprovedHeapSuite_{heap_rng}_{hapax_fs}"]


def open_pickle(f, dir_prefix="./"):
    with open(dir_prefix + f, "rb") as handle:
        return pickle.load(handle)
    


def get_stat(stats_names_ind, open_f=open_pickle, dir_prefix="./", **kwargs):
    filename = stats_names[stats_names_ind]
    if open_f is open_pickle:
         filename += ".pkl"
    stat = open_f(filename, dir_prefix=dir_prefix, **kwargs)
    
    return stat



if __name__ == "__main__":
    lang, n = parse_args()
    stats_names = get_stats_names(n)
    
    print(lang, os.listdir("."))
    print(lang, os.listdir("Results/" + lang + "/" + "stats/"))
    
    stat_dir = "Results/" + lang + "/" + "stats/"
    param_dir = stat_dir + "params/"
    if not os.path.isdir(param_dir):
        print(lang, "MADE DIR ", param_dir, flush=True)
        os.makedirs(param_dir)
    
    #%% ZIPF: SENTENCE RANK FREQ
    print("\n" + lang + ": ZIPF: SENTENCE RANK FREQ", flush=True)
    
    sentence_spec_suite = get_stat(1, open_f=ImprovedSpectrumSuite.from_pickle, 
                                   dir_prefix=stat_dir)    
    
    suite_dir = param_dir + repr(sentence_spec_suite) + "/"
    print(lang, repr(sentence_spec_suite), flush=True)
    os.makedirs(suite_dir)
    
    for name, spec in zip(sentence_spec_suite.names, sentence_spec_suite.spectra):
        mandelbrot = Mandelbrot(spec.propens, spec.domain)
        mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                 method="powell", full_output=True)
        mandelbrot.register_fit(mandelbrot_fit)
        
        print(lang, name, flush=True)
        mandelbrot.print_result()
        print(flush=True)

        mandelbrot.to_pickle(suite_dir + str(name), remove_data=True)
        
        lowess = LOWESS(spec.propens, 
                    spec.domain, log=True)
        lowess.to_pickle(suite_dir + str(name) + "_lowess", remove_data=True)
    
    
    #%% SPLIT LEVELS
    print("\n" + lang + ": SPLIT LEVELS", flush=True)

    all_words_ranks_freqs = get_stat(2, dir_prefix=stat_dir)
    
    mandelbrot = Mandelbrot(all_words_ranks_freqs.propens, all_words_ranks_freqs.domain)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)
    mandelbrot.register_fit(mandelbrot_fit)    
    print(lang, "WORDS", str(all_words_ranks_freqs), flush=True)
    mandelbrot.print_result()
    print(flush=True)
    mandelbrot.to_pickle(param_dir + str(all_words_ranks_freqs), remove_data=True)
    
    lowess = LOWESS(all_words_ranks_freqs.propens, 
                    all_words_ranks_freqs.domain, log=True)
    lowess.to_pickle(param_dir + str(all_words_ranks_freqs) + "_lowess", remove_data=True)
    
    
    
    all_articles_ranks_freqs = get_stat(3, dir_prefix=stat_dir) 
    
    mandelbrot = Mandelbrot(all_articles_ranks_freqs.propens, all_articles_ranks_freqs.domain)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)
    mandelbrot.register_fit(mandelbrot_fit)    
    print(lang, "ARTICLES", str(all_articles_ranks_freqs), flush=True)
    mandelbrot.print_result()
    print()
    mandelbrot.to_pickle(param_dir + str(all_articles_ranks_freqs), remove_data=True)
    
    lowess = LOWESS(all_articles_ranks_freqs.propens, 
                    all_articles_ranks_freqs.domain, log=True)
    lowess.to_pickle(param_dir + str(all_articles_ranks_freqs) + "_lowess", remove_data=True)

    
    #%% REPRESENTATIONS
    print("\n" + lang + ": REPRESENTATIONS", flush=True)
    
    all_sents_freqs_freq = get_stat(5, dir_prefix=stat_dir)
    
    lowess = LOWESS(all_sents_freqs_freq.propens, 
                    all_sents_freqs_freq.domain, log=True)
    
    lowess.to_pickle(param_dir + str(all_sents_freqs_freq) + "_lowess", remove_data=True)

    #%% CONVERGENCE
    print("\n" + lang + ": CONVERGENCE", flush=True)

    
    convergence_rank_suite = get_stat(7, 
                                      open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir,
                                      suite_name=stats_names[7].replace("ImprovedSpectrumSuite_", ""))
    
    suite_dir = param_dir + repr(convergence_rank_suite) + "/"
    print(lang, repr(convergence_rank_suite), flush=True)
    os.makedirs(suite_dir)
    
    for name, spec in zip(convergence_rank_suite.names, convergence_rank_suite.spectra):
        mandelbrot = Mandelbrot(spec.propens, spec.domain)
        mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                 method="powell", full_output=True)
        mandelbrot.register_fit(mandelbrot_fit)        
        print("\t", lang, name, flush=True)
        mandelbrot.print_result()
        print()
        mandelbrot.to_pickle(suite_dir + str(name), remove_data=True)
        
        lowess = LOWESS(spec.propens, 
                    spec.domain, log=True)
        lowess.to_pickle(suite_dir + str(name) + "_lowess", remove_data=True)
    
    
    convergence_freq_suite = get_stat(8, open_f=ImprovedSpectrumSuite.from_pickle,
                                      dir_prefix=stat_dir,
                                      suite_name=stats_names[8].replace("ImprovedSpectrumSuite_", ""))
    
    suite_dir = param_dir + repr(convergence_freq_suite) + "/"
    print(lang, repr(convergence_freq_suite), suite_dir, flush=True)
    os.makedirs(suite_dir)
    
    for name, spec in zip(convergence_freq_suite.names, convergence_freq_suite.spectra):
        print("\t",lang, name, flush=True)
        lowess = LOWESS(spec.propens, 
                    spec.domain, log=True)
        lowess.to_pickle(suite_dir + str(name) + "_lowess", remove_data=True)
    
    #%% HEAP
    print("\n" + lang + ": HEAP", flush=True)

#    heaps = get_stat(9, dir_prefix=stat_dir)
#    
#    
#    print(stats_names[9])
#    print(stats_names[10])
#    print(heaps)
#    
#    heap_ls = []
#    
#    for heap in heaps:
#        print(str(heap))
#        heap_model = Heap(heap.counts, heap.domain)
#        res_heap = heap_model.fit(start_params=(10.0, 0.7), full_output=True)    
#        heap_model.register_fit(res_heap)    
#        heap_model.remove_data()
#        heap_ls.append(heap_model)
#    heap_ls = tuple(heap_ls)
#    
#    print(heap_ls)
#        
#    with open(param_dir + "heap_ls.pkl", "wb") as handle:
#        pickle.dump(heap_ls, handle)
        
    
    hapaxes = get_stat(10, dir_prefix=stat_dir)
    
    suite_dir = param_dir + repr(hapaxes) + "/"
    print(lang, repr(hapaxes))
    os.makedirs(suite_dir)    

    for f, heap in hapaxes.heaps.items():
        print(lang, f, flush=True)
        if f == "all" or f < 3:    
            heap_model = Heap(heap.counts, heap.domain)
            res_heap = heap_model.fit(start_params=(10.0, 0.7), full_output=True)    
            heap_model.register_fit(res_heap)
            heap_model.print_result()
            heap_model.remove_data()
            
            heap_model.to_pickle(suite_dir + str(f), remove_data=True)    
    