# -*- coding: utf-8 -*-

from Filters.Filter import Filter, np, rand

from scipy.special import zeta
from scipy.optimize import minimize

from collections import Counter
from data.DataGenerators import RankTranslator

#class TypicalityFilter:#(Filter):
#    def __init__(self, corpus, n_samples, epsilon, cmp=float.__lt__):
#        self.n = n_samples
#        self.epsilon = epsilon
#        self.cmp = cmp
#        
##        super().__init__(corpus)
#    
#    
#    def sample(self, corpus, n_samples=None, epsilon=None):
#        if not n_samples:
#            n_samples = self.n_samples
#        if not epsilon:
#            epsilon = self.epsilon
#            
#        corpus_len = len(corpus)
#        
#        sampled, used_inds = [], {}
#        cur_typ, n_sampled = 0., 0
#            
#        while n_sampled < n_samples:
#            cur_ind = rand.randint(len(corpus))
#            cur_sample = corpus[cur_ind]
#            if cur_ind in used_inds:
#                continue    
#            if not cur_sample:
#                continue
#            
#            
#            
#    def is_next_sample(self, cur_sample, cur_typ, max_typ):
#        pass
#        
#        
#        
#    def zipf_seq_prob(self, token_ranks, alpha, log=False):
#        if log:
#            norm = np.log2(zeta(alpha))
#            numer = -alpha*np.log2(token_ranks)
#            return np.sum(numer-norm)
#        norm = zeta(alpha)
#        indiv_probs = (np.asarray(token_ranks)**(-alpha))/norm
#        return np.prod(indiv_probs)
#
#
#    def empirical_entropy(self, token_ranks, alpha=1.15):
#        return -(1/len(token_ranks))*self.zipf_seq_prob(token_ranks, alpha, log=True)
#        
#            
#    def increment_emprical_entropy(self, alpha, cur_ent, cur_len, new_seq):
#        new_ent = self.empirical_entropy(new_seq, alpha)
#        new_len = len(new_seq)
#        
#        return (cur_len/(cur_len + new_len))*cur_ent +\
#                (new_len/(cur_len + new_len))*new_ent
            
            
class TypicalityFilter(Filter):
    def __init__(self, corpus, n_samples, epsilon, cmp=float.__lt__):
        self.n = n_samples
        self.epsilon = epsilon
        self.cmp = cmp
        self.rank_trans = RankTranslator()
        self.rank_trans._init_ranks(corpus)
        super().__init__(corpus)
    
    
    def estimate_source(self, corpus):
        cur_words = [w for s in corpus for w in s]
        ent_a = lambda a: self.empirical_entropy(cur_words, alpha=a)
        
        optim_result = minimize(ent_a, 1.01)
        return optim_result["x"], optim_result["fun"]
        
        
    
    def zipf_seq_prob(self, token_ranks, alpha, log=False):
        if log:
            norm = np.log2(zeta(alpha))
            numer = -alpha*np.log2(token_ranks)
            return np.sum(numer-norm)
        norm = zeta(alpha)
        indiv_probs = (np.asarray(token_ranks)**(-alpha))/norm
        return np.prod(indiv_probs)


    def empirical_entropy(self, token_ranks, alpha):
        return -(1/len(token_ranks))*self.zipf_seq_prob(token_ranks, alpha, log=True)
        
    
    def increment_emprical_entropy(self, cur_ent, cur_len, new_ent, new_len):        
        return (cur_len/(cur_len + new_len))*cur_ent +\
                (new_len/(cur_len + new_len))*new_ent
                
    def in_typical_set(self, sample_ent, entropy, epsilon=None):
        if not epsilon:
            epsilon = self.epsilon
            
        return abs(entropy - sample_ent) <= epsilon
    
    
    def sample(self, corpus, n_samples=None, epsilon=None):
        if not n_samples:
            n_samples = self.n
        if not epsilon:
            epsilon = self.epsilon
            
        corpus_len = len(corpus)
        corpus_trans = list(self.rank_trans.translate(corpus))
        
        zipf_param, zipf_ent = self.estimate_source(corpus_trans)
        
        n_sampled, cur_ent, cur_len = 0, .0, 0
        used_inds, sampled_inds = set(), []
        
        while n_sampled < n_samples:
            cur_ind = rand.randint(corpus_len)
            if cur_ind in used_inds:
                continue
            used_inds.add(cur_ind)

            cur_sent = corpus_trans[cur_ind]
            print(corpus[cur_ind])
            if not cur_sent:
                continue
            sent_len = len(cur_sent)
        
            sent_ent = self.empirical_entropy(cur_sent, zipf_param)        
            
            new_ent = self.increment_emprical_entropy(cur_ent, cur_len,
                                                      sent_ent, sent_len)
            
            if self.cmp(zipf_ent, new_ent) and \
                not self.in_typical_set(new_ent, zipf_ent):
                    sampled_inds.append(cur_ind)
                    cur_ent = new_ent
                    cur_len += sent_len
                    n_sampled += 1
        
        sampled_sents = (corpus[i] for i in sampled_inds)
        return list(sampled_sents), zipf_param, \
                zipf_ent, cur_ent,\
                self.empirical_entropy([w for i in sampled_inds for w in corpus_trans[i]], zipf_param)
        

#%%
        
#import numpy as np
#import numpy.random as rand
#    
#w = ["the", "old", "young", "tree", "house"]
#p_w = 1/np.arange(1, len(w)+1)
#p_w = p_w/np.sum(p_w)
#
#ls = list(filter(lambda l: 1 < l < 6, rand.poisson(lam=2, size=30)))
#print(len(ls))
#c = [list(rand.choice(w, size=l, p=p_w)) for l in ls]
#
#
#
##%%
#
#tf = TypicalityFilter(c, n_samples=5, epsilon=0.1)
    