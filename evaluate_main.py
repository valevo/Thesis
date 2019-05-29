# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import numpy.random as rand

from data.DataGenerators import DataGenerator, SentencePieceTranslator
#from data.WikiReader import wiki_from_pickles

from LMs.models import LanguageModel1
from LMs.metrics import perplexity

from collections import Counter

import matplotlib.pyplot as plt

#%% DATA HELPERS

def load_filter(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)

def get_filters(filter_dir, bool_func, to_list=False):
    filter_iter = (load_filter(filter_dir + file_n) for file_n in sorted(os.listdir(filter_dir))
                    if not os.path.isdir(filter_dir + file_n))
    
    cur_filters = (cur_f for cur_f in filter_iter if bool_func(cur_f))
    
    if to_list:
        return list(cur_filters)
    return cur_filters
    

def prepare_filter_training(subword_model, filter_obj):
    translator = SentencePieceTranslator(subword_model)
    
    translated = list(translator.translate(filter_obj))
    
    V_this = len(set(w for s in translated for w in s))
    V_spm = translator.sp.GetPieceSize()
    print(V_this, V_spm, max(set(w for s in translated for w in s)))
    
    gen = DataGenerator(translated, vocab_size=V_spm, group=True)
    return gen


#%% MODEL HELPERS

def load_models(weight_dir, prefix, to_list=True):
    model_dirs = filter(lambda s: s.find(prefix) >= 0, os.listdir(weight_dir))
    
    model_iter = (LanguageModel1.from_saved_custom(weight_dir + m_dir)
                    for m_dir in sorted(model_dirs))
    
    if to_list:
        return list(model_iter)
    return model_iter
    

if __name__ == "__main__":
#%% GENERAL
    lang = "TR"    
    top_dir = lambda dir_name: "Results/" + lang + "/" + dir_name + "/"
    
    cur_n = 1000000
    cur_m = 100
    
    print("\n" + "="*20 + " EVALUATION " + "="*20)
    print(top_dir(""), lang, cur_n, cur_m, sep=", ")
    
    #%% LOAD FILTERED EVAL SUBCORPUS
    print("  ... loading filtered subcorpus")
    
    cur_i = 9
    
    filter_dir = top_dir("filters")
    filtered_subcorp = load_filter(filter_dir + 
                        "SpeakerRestrictionFilterRandomised_"+ str(cur_n) + "_" + 
                        str(cur_m) + "_" + str(cur_i) + ".pkl")
     
    gen_filtered = prepare_filter_training(top_dir("subword") + "TR_10000.model",
                                               filtered_subcorp)
    
    
    #%% LOAD UNIFORM EVAL SUBCORPUS
    print("  ... loading uniform subcorpus")
    
    cur_i = 9
    
    uniform_dir = top_dir("uniform")    
    
    uniform_subcorp = load_filter(uniform_dir + 
                        "UniformFilter_"+ str(cur_n) + 
                        "_" + str(cur_i) + ".pkl")
    
    gen_uniform = prepare_filter_training(top_dir("subword") + "TR_10000.model",
                                               uniform_subcorp)
    
    #%% LOAD MODELS
    
    print(" ... loading models")
    
    filter_models = load_models(top_dir("weights"), prefix="SpeakerRestriction")
    uniform_models = load_models(top_dir("weights"), prefix="Uniform")
    
    
    #%% SAVE RESULTS
    
    eval_dir = top_dir("evaluation")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    
    printf = lambda h, obj: h.write(repr(obj) + "\n")
    
    #%% ITERATE & MEASURE
    
    for data_name, cur_eval_gen in zip(["uniform", "filtered"], [gen_uniform, gen_filtered]):
        print("EVALUATING ON", data_name, "DATA")
        print("\n ... iterating filtered models\n\n")
        
        with open(eval_dir + data_name + "_filtered.txt", "w") as handle:     
            for i, cur_model in enumerate(filter_models):
                print(cur_model)
                printf(handle, cur_model)
                printf(handle, i)
                corpus_perplexities = []
                filtered_batches = cur_eval_gen.generate(permute=False)
                for cur_batch, cur_ys in filtered_batches:
                    if i == 1: print(cur_batch.shape)
                    batch_perplexities = perplexity(cur_model, cur_batch, raw=True)
                    corpus_perplexities.extend(batch_perplexities)
                print()
                # distribution over perplexities
                printf(handle, np.mean(corpus_perplexities))
                printf(handle, np.median(corpus_perplexities))
                printf(handle, np.var(corpus_perplexities))
                printf(handle, "\n")
                
                plt.hist(corpus_perplexities, bins=50, histtype="step", label=str(i))
                
                
            plt.title("FILTERED " + str(cur_m))
            plt.legend()
            plt.savefig(eval_dir + data_name + "_filtered.png", dpi=500)
    
                
        
        print("\n\n\n ... iterating uniform models\n\n")
        
        with open(eval_dir + data_name + "_uniform.txt", "w") as handle:     

            for i, cur_model in enumerate(uniform_models):
                printf(handle, cur_model)
                printf(handle, i)
                corpus_perplexities = []
                filtered_batches = cur_eval_gen.generate(permute=False)
                for cur_batch, cur_ys in filtered_batches:
                    batch_perplexities = perplexity(cur_model, cur_batch, raw=True)
                    corpus_perplexities.extend(batch_perplexities)
                # distribution over perplexities
                printf(handle, np.mean(corpus_perplexities))
                printf(handle, np.median(corpus_perplexities))
                printf(handle, np.var(corpus_perplexities))
                printf(handle, "\n")
                
                plt.hist(corpus_perplexities, bins=50, histtype="step", label=str(i))
                
            plt.title("UNIFORM")
            plt.legend()
            plt.savefig(eval_dir + data_name + "_uniform.png", dpi=500)


#%% ITERATE & MEASURE

#cur_eval_gen = gen_uniform
#
#print("\n ... iterating filter models\n\n")
#
#for i, cur_model in enumerate(filter_models):
#    print(cur_model)
#    corpus_perplexities = []
#    filtered_batches = cur_eval_gen.generate(permute=False)
#    for cur_batch, cur_ys in filtered_batches:
#        batch_perplexities = perplexity(cur_model, cur_batch, raw=True)
#        corpus_perplexities.extend(batch_perplexities)
#        
#    # distribution over perplexities
#    print("\t", np.mean(corpus_perplexities), end="")
#    print("\t", np.median(corpus_perplexities), end="")
#    print("\t", np.var(corpus_perplexities))
#    print("\n")
#    
#    plt.hist(corpus_perplexities, bins=50, histtype="step", label=str(i))
#    
#plt.title("FILTERED " + str(cur_m))
#plt.legend()
#plt.show()
#        
#
#print("\n\n\n ... iterating uniform models\n\n")
#
#for i, cur_model in enumerate(uniform_models):
#    print(cur_model)
#    corpus_perplexities = []
#    filtered_batches = cur_eval_gen.generate(permute=False)
#    for cur_batch, cur_ys in filtered_batches:
#        batch_perplexities = perplexity(cur_model, cur_batch, raw=True)
#        corpus_perplexities.extend(batch_perplexities)
#    # distribution over perplexities
#    print("\t", np.mean(corpus_perplexities), end="")
#    print("\t", np.median(corpus_perplexities), end="")
#    print("\t", np.var(corpus_perplexities))
#    print("\n")
#    
#    plt.hist(corpus_perplexities, bins=50, histtype="step", label=str(i))
#    
#plt.title("UNIFORM")
#plt.legend()
#plt.show()
#




#%%
#
#gen 
#
#
#
#test_gen = gen.generate(permute=True)
#
#test_x, test_y = next(test_gen)
#test_x = test_x[0:1]
#test_y = np.argmax(test_y[0:1], axis=-1)
#
#test_pred_dist = lm1.predict(test_x)
#
#test_pred = np.argmax(test_pred_dist, axis=-1)
#
#
#print(test_y)
#print(test_pred)
#
#print(list(translator.invert_translation(test_y)))
#print(list(translator.invert_translation(test_pred)))
#
#
#print("\n")
#print(perplexity(lm1, test_x))
#
#
#
#
##%%
#
#perplexities = []
#
#for _ in range(50):
#
#    test_gen = gen.generate(permute=True)
#
#    test_x, test_y = next(test_gen)
#    test_x = test_x[0:1]
#    test_y = np.argmax(test_y[0:1], axis=-1)
#        
#    
#    test_pred_dist = lm1.predict(test_x)
#    
#    #test_pred = np.argmax(test_pred_dist, axis=-1)
#    sampled = []
#    for cur_p in test_pred_dist[0]:
#        cur_sample = rand.choice(V+1, p=cur_p)
#        sampled.append(cur_sample)
#        
#    sampled = np.asarray([sampled])
#    test_pred = sampled
#    
#    
#    #print(test_y)
#    #print(test_pred)
#    
#    print("".join(list(translator.invert_translation(test_y))[0]))
#    print()
#    print("".join(list(translator.invert_translation(test_pred))[0]))
#    
#    
#    perplex = perplexity(lm1, test_x)
#    perplexities.append(perplex)
#    
#    print()
#    print("Perplexity: ", perplex)
#    print("Relative: ", perplex/V)
#    
#    print("__"*20 + "\n")
#    