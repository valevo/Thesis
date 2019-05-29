# -*- coding: utf-8 -*-

#%%

import pickle
import os

def split_pickles(pkl_dir, into=10):
    path = lambda f_name: pkl_dir + "/" + f_name
    for f in sorted(os.listdir(pkl_dir)):
        contents = None
        with open(path(f), "rb") as handle:
            contents = pickle.load(handle)
            
        cut = int(len(contents)/into)
        
        dir_name = path(f).replace(".pkl", "")
        os.makedirs(dir_name)
        
        for i in range(0, into-1):
            with open(dir_name + "/" + f"{i:02d}_" + f, "wb") as handle:
                pickle.dump(contents[i*cut:(i+1)*cut], handle)
                
        with open(dir_name + "/" + f"{into-1:02d}_" + f, "wb") as handle:
                pickle.dump(contents[(into-1)*cut:], handle)
                
def load_split_pickles(pkl_dir, expected=0):
    files = os.listdir(pkl_dir)
    if expected and len(files) != expected:
        raise ValueError("Expected " + str(expected) + " files, " +
                         " found " + str(len(files)))
        
    path = lambda f_name: pkl_dir + "/" + f_name

    contents = []
    for f in sorted(files):
        with open(path(f), "rb") as handle:
            contents.extend(pickle.load(handle))
            
    return contents
            
          
        
##%%
#                
#ls = list(range(102))
#
#with open("test_dir/test.pkl", "wb") as handle:
#    pickle.dump(ls, handle)
#        
##%%
#    
#split_pickles("test_dir")
#
#
##%%
#
#ls2 = load_split_pickles("test_dir/test", expected=10)