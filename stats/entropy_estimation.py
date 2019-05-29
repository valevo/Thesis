# -*- coding: utf-8 -*-

from stats.stat_functions import get_freqs, get_probs

from scipy.special import digamma

import numpy as np

from collections import Counter

lg = np.log2

class ImprovedMutualInfo:
    def __init__(self, corpus, max_len):
        self.max_len = max_len
        self.vocab = sorted(set(self.chars_from(corpus, keep_sents=False)))
        
        self.joints
        
                
    def iter_joints(self, corpus, chars=True):
        iter_items = self.chars_from if chars else iter
        items = iter_items(corpus)
        
        for sent in items:
            m = min(len(sent), self.max_len)
            for i in range(m):
                for j in range(i+1, m):
                    yield (i, j), (sent[i], sent[j])

                    
    
    def group_joints(self, corpus, chars=True):
        joint_dict = {}
        
        joint_iter = self.iter_joints(corpus, chars=chars)
        
        for ind_tup, joint_tup in joint_iter:
            if not ind_tup in joint_dict:
                joint_dict[ind_tup] = []
            joint_dict[ind_tup].append(joint_tup)
            
        return joint_dict
            
    
    def group_joints(self, corpus, chars=True):
        iter_items = self.chars_from if chars else iter
        items = iter_items(corpus)
        joint_dict = {}
        for sent in items:
            m = min(len(sent), self.max_len)
            for i in range(m):
                for j in range(i+1, m):
                    if not (i, j) in joint_dict:
                        joint_dict[(i, j)] = []
                    joint_dict[(i, j)].append((sent[i], sent[j]))
        
        return joint_dict

    def joint_probs(self, corpus):
        vocab_cross = [(v1, v2) for v1 in self.vocab for v2 in self.vocab]
        joint_dict = self.group_joints(corpus)
        
        joint_freqs = ((ind_tup, Counter(get_freqs(char_tup_sample)))
                        for ind_tup, char_tup_sample in joint_dict.items())
        
        joint_freqs = {ind_tup: list(map(c.__getitem__, vocab_cross))
                        for ind_tup, c in joint_freqs}
        
        return vocab_cross, joint_freqs
        
        
    

    def entropy_estimate(self, freqs):
        n = sum(freqs)
        
        return lg(n) - (1/n)*sum([f*digamma(f) for f in freqs])
        
#        return lg(n) - (1/n)*np.sum(freqs*digamma(freqs))
        
        
    
    def naive_entropy_estimate(self, freqs):
        freqs = np.asarray(freqs)
        emp_probs = freqs/freqs.sum()
        return -sum((p*lg(p) for p in emp_probs if p > 0))
#        return - np.sum(emp_probs*lg(emp_probs))
            
    
    def chars_from(self, corpus, keep_sents=True, to_list=False):
        if keep_sents:
            chars = ([c for w in s for c in w] for s in corpus)
        else:
            chars = (c for s in corpus for w in s for c in w)
            
        if to_list:
            return list(chars)
        
        return chars
    
    
    def tokens_from(self, corpus, to_list=False):
        toks = (w for s in corpus for w in s)
        if to_list:
            return list(toks)
        return toks

#%%
        
from stats.entropy_estimation import ImprovedMutualInfo
