# -*- coding: utf-8 -*-

#%%

import numpy as np
import numpy.random as rand

from collections import Counter




def from_categorical(categorical_mat, axis=-1):
    if len(np.shape(categorical_mat)) < 2:
        raise ValueError("Conversion on one-dimensional object undefined!")
    return np.argmax(categorical_mat, axis=-1)


#def perplexity(probs):
#    neg_avg_log_p = -np.mean(np.log2(probs))
#    return 2**neg_avg_log_p



def perplexity(lm, test_batch, raw=True):
    preds = lm.predict(test_batch)
    
    avgs = np.zeros(len(test_batch))
    
    for i, s in enumerate(test_batch):
        sent_probs = [preds[i, j, w_ind] for j, w_ind in enumerate(s)]
        
#        print(s)
#        print(sent_probs)
#        print("\n")
        
        neg_sent_log_p = -np.sum(np.log2(sent_probs))
        avgs[i] = neg_sent_log_p
    
    if raw:    
        return avgs
    return 2**np.mean(avgs)
            

# 
def baseline_perplexity(V, n):
    unif_p = np.ones((V, ))/V
    
    
    
    
    
