# -*- coding: utf-8 -*-
import os
import pickle
        
def wiki_from_pickles(pkls_dir, drop_titles=True, n_tokens=int(50e6)):
    pkl_files = os.listdir(pkls_dir)
    
    n_loaded = 0
    
    for f in pkl_files:
        with open(pkls_dir + "/" + f, "rb") as handle:
            wiki = pickle.load(handle)
            for title, a in wiki:
                n_loaded += len([w for s in a for w in s])
                yield a if drop_titles else (title, a)
                
                if n_loaded >= n_tokens:
                    break
                
                
def corpus_to_pickle(corpus, pkl_dir, pkl_name):
    with open(pkl_dir + "/" + pkl_name + ".pkl", "wb") as handle:
        pickle.dump(corpus, handle)
        
        
def corpora_from_pickles(pkls_dir, names):
    files = os.listdir(pkls_dir)
    
    if not files:
        raise FileNotFoundError("DIRECTORY " + pkls_dir + " is empty!")
    
    for f in files:
        with open(pkls_dir + "/" + f, "rb") as handle:
            param_vals = list(map(lambda x: int(float(x)), 
                                  f.replace(".pkl", "").split("_")))
            param_dict = dict(zip(names, param_vals))
            cur_corp = pickle.load(handle)
            yield param_dict, cur_corp