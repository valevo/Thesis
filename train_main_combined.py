# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

import os
import pickle
from time import asctime

from data.DataGenerators import DataGenerator, DataGenerator2, SentencePieceTranslator
from LMs.models import LanguageModel

import keras.backend as K
from keras.optimizers import SGD, Adam

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import argparse

#%%
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filter_dir", type=str,
                   help="The folder containing the filtering results.")
    
    p.add_argument("--hist_len", type=int, default=None, 
                   help="The filtering history length to consider."
                   "\nIf not provided, all history lengths are considered.")
    
    args = p.parse_args()
    
    if args.hist_len:
        cur_m = args.hist_len
        cur_get_filters = lambda f: f.m == cur_m
        return args.filter_dir, cur_get_filters
    else:
        cur_get_filters = lambda f: True
        return args.filter_dir, cur_get_filters


#%%

def setup_dir(dir_name, prefix="", error_if_exists=False):
    if prefix and not os.path.isdir(prefix):
        raise FileNotFoundError("DIRECTORY " + prefix + " NOT FOUND!")
    if os.path.isdir(prefix + dir_name):
        if error_if_exists:
            raise FileExistsError("DIRECTORY " + dir_name + 
                                  (" IN " + prefix if prefix else " ") + 
                                  " EXISTS!")
    else:
        os.makedirs(prefix + dir_name)
        
        
def prepare_folders(lang, filter_type, top_dir="Results"):
    dir_prefix = lambda f_name, final_slash=True: "/".join([top_dir, lang, f_name]) + ("/" if final_slash else "")

    subword_file = dir_prefix("subword/ALS_2000.model", final_slash=False)  # TR_10000
    
    filter_dir = dir_prefix(filter_type)
    
    if not os.path.exists(subword_file):
        raise FileNotFoundError("SUBWORD MODEL NOT FOUND AT " + subword_file)
    if not os.path.exists(filter_dir):
        raise FileNotFoundError("FILTERS NOT FOUND IN " + filter_dir)
    
    
    log_dir = dir_prefix("logs")
    setup_dir(log_dir, error_if_exists=False)
    
    weight_dir = dir_prefix("weights")
    setup_dir(weight_dir, error_if_exists=False)
    
    return dir_prefix, subword_file, filter_dir, log_dir, weight_dir





#%%

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


def prepare_filter_training2(subword_model, filter_obj):
    translator = SentencePieceTranslator(subword_model)
    
    translated = list(translator.translate(filter_obj))
    
    V_this = len(set(w for s in translated for w in s))
    V_spm = translator.sp.GetPieceSize()
    print(V_this, V_spm, max(set(w for s in translated for w in s)))
    
    gen = DataGenerator2(translated, vocab_size=V_spm)
    return gen

#%%

if __name__ == "__main__":
    lang = "ALS"

    cur_filter_type, get_filter_f = parse_args()

    print("starting...")

    dir_prefix, subword_file, filter_dir, log_dir, weight_dir = prepare_folders(lang, cur_filter_type)
    
    print("set up directories...")
    
    
    filters = get_filters(filter_dir, get_filter_f, 
                          to_list=True)
    
    print("loaded filters", len(filters))
    
    for i, cur_f in enumerate(filters):     
        
        
        if i > 1:
            break
        
        
        
        cur_name = repr(cur_f) + "_" + str(i)
        print("\n"+"="*50)
        print("Training filter <", repr(cur_f), "> no. ", i)
        
        f_gen = prepare_filter_training2(subword_file, cur_f)
        
        print("  ...DataGenerator setp up; V=", f_gen.vocab_size,
              " batches=", f_gen.n_batches)
        
        
        setup_dir(cur_name, prefix=log_dir, error_if_exists=True)
        K.clear_session()
        tb = TensorBoard(log_dir=log_dir + cur_name,
                         histogram_freq=0,
                         write_grads=False,
                         update_freq=50)
        
        
        forever_gen = f_gen.generate_forever(permute=False) 
        
        lm = LanguageModel(f_gen.vocab_size, rnn_type="gru", compile_now=True)

        
        print("  ...Language model set up; size=", sum(map(np.size, lm.get_weights())))    
        print() 
        
        lm.fit_generator(forever_gen, 
                             steps_per_epoch=f_gen.n_batches, 
                             epochs=3, verbose=1, callbacks=[tb])
        
        setup_dir(cur_name, prefix=weight_dir, error_if_exists=True)
        lm.save_model_custom(weight_dir + cur_name)    