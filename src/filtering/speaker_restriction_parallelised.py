# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from functools import reduce


sep = "■"


def filter_speaker_restrict(sents, n, history_len):
    cur_sample = rand.randint(len(sents))
    sampled_s = sents[cur_sample].split(sep)
    yield sampled_s
    
    cur_hist = [sampled_s]
    used = {cur_sample}
    sampled = len(sampled_s)
    
    num_not_found = 0
    num_iter = 0
    
    while sampled < n:
        num_iter += 1
        
        cur_sample = rand.randint(len(sents))
        sampled_s = sents[cur_sample].strip().split(sep)
        
        if not sampled_s:
            continue
        
        if cur_sample in used:
            continue
        
        cur_disallowed = reduce(np.union1d, cur_hist)
        
        if np.intersect1d(sampled_s, cur_disallowed).size > 0:
            num_not_found += 1
#            if num_not_found >= history_len*n:
#                print("NUM ITER: ", num_iter)                
#                raise RuntimeError("number of samples has outgrown n! aborting")
            continue
        
        if len(cur_hist) >= history_len:
            cur_hist.pop(0)
        cur_hist.append(sampled_s)
        
        used.add(cur_sample)
        sampled += len(sampled_s)
        
        if sampled % 1e5 == 0:
            print("sampled/1e5: ", sampled/1e5, flush=True)
        
        yield sampled_s
    
    
    print("NUM ITER: ", num_iter, flush=True)
    print("NUM NOT FOUND: ", num_not_found, flush=True)
    
    