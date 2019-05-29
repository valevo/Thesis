# -*- coding: utf-8 -*-

from Filters.Filter import Filter, np, rand

from collections import Counter


class DropAllFilter(Filter):
    def __init__(self, corpus, n, k, recalc):
        self.n = n
        self.k = k
        
        self.sample = self.sample_recalc if recalc else self.sample_no_recalc
        
        super().__init__(corpus)
        
        
    def sample_no_recalc(self, corpus, n=None, k=None):
        if not n:
            n = self.n
        if not k:
            k = self.k

        top_w, top_w_freqs = list(zip(*self.most_common(k)))
        
        n_tokens_corp = len(list(self.tokens(corpus)))
        
        if (n_tokens_corp - sum(top_w_freqs)) < 
        
        cur_corp = rand.permutation(corpus)
        
            filtered_corp = list(filter(lambda s: not any(w in top_w for w in s), 
                                self.corpus))
            
        
            
        
        
    def most_common(self, corpus, k_most=None):
        counter = Counter(self.tokens(corpus))
        top_ranks = counter.most_common(k_most)
        return top_tanks
        
    
    
#"""
#This filter drops all sentences that contain the most common words.
#
#:param k: how many of the most frequent words to eliminate
#:param do_recalc: whether to re-calculate rank-ordering after each elimination
#
#
#"""
#
##%%
#from Filters.Filter import *
#from collections import Counter
#
#class DropAllFilter(Filter):
#    def __init__(self, corpus, k, do_recalc, initial_sample=True):
#        self.k = k
#        self.do_recalc = do_recalc
#        
#        super().__init__(corpus, initial_sample)
#        
#    def resample(self, k=None, do_recalc=None, reset=False):       
#        if not k:
#            k = self.k
#        if not do_recalc:
#            do_recalc = self.do_recalc
#            
#        if reset:
#            self.k = k
#            self.do_recalc = do_recalc
#            self._reset()
#        
#        self.eliminated_words = set()
#        return self.resample_recalc(k) if do_recalc \
#                else self.resample_no_recalc(k)
#            
#    def resample_recalc(self, k):
#        filtered_sents = self.corpus
#        for i in range(k):
#            top_w = self.most_common(filtered_sents, return_set=False)[0]
#            self.eliminated_words.add(top_w)
#            #print("MOST COMMON WORD", top_w)
#            filtered_sents = list(filter(lambda s: not any(w == top_w for w in s), 
#                                filtered_sents))
#        return filtered_sents
#    
#    def resample_no_recalc(self, k):
#        top_w = self.most_common(self.corpus, n=k)
#        self.eliminated_words.update(top_w)
#        filtered_sents = list(filter(lambda s: not any(w in top_w for w in s), 
#                                self.corpus))
#        return filtered_sents
#    
#    def most_common(self, sents, n=1, return_set=True):
#        tokens = self.tokens(sents)
#        most_common_w = [w for w, c in
#                Counter(tokens).most_common(n)]
#        if return_set:
#            most_common_w = set(most_common_w)
#        return most_common_w
#    
#        

        