# -*- coding: utf-8 -*-

from Filters.Filter import Filter, rand

#%%

class UniformFilterTokens(Filter):
    def __init__(self, corpus, num_tokens):
        self.n = num_tokens
        super().__init__(corpus)
        
    def sample(self, corpus, num_tokens=None):
        if not num_tokens:
            num_tokens = self.n
        
        sampled_tokens = 0
        sampled = []
        used = set()
        corpus_len = len(corpus)
        while sampled_tokens < num_tokens:
            cur_ind = rand.randint(corpus_len)
            
            if cur_ind in used:
                continue
            
            if not corpus[cur_ind]:
                continue
            
            sampled.append(cur_ind)
            used.add(cur_ind)
            sampled_tokens += len(corpus[cur_ind])
        
        
        return (corpus[i] for i in sampled)
    
    def __repr__(self):
        return "UniformFilter_" + str(self.n)
