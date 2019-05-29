# -*- coding: utf-8 -*-

"""
This filter 


"""

#%%

from Filters.Filter import Filter, np, rand, reduce

class SpeakerRestrictionFilterRandomised(Filter):
    def __init__(self, corpus, n_samples, history_len, sample_count_f=None):
        self.n = n_samples
        self.m = history_len
        if not sample_count_f or sample_count_f == "count_sentences":
            self.sample_count_f = self.count_sentences
        elif sample_count_f == "count_tokens":
            self.sample_count_f = self.count_tokens
        else:
            raise NotImplementedError("count function keyword " + str(sample_count_f) +\
                                      " was given but isn't implemented!")
    
    
        super().__init__(corpus)
        
    def count_sentences(self, s):
        return 1
    
    def count_tokens(self, s):
        return len(s)
    
    def sample(self, corpus, n_samples=None, history_len=None):
        if not n_samples:
            n_samples = self.n
        if not history_len:
            history_len = self.m
        
        corpus_len = len(corpus)
        sampled, used, cur_hist = self.initial_sample(corpus)
        n_sampled = 0
        
        while n_sampled < n_samples:
            cur_ind = rand.randint(corpus_len)
            
            if self.is_next_sample(cur_ind, used, corpus, cur_hist, history_len):
                sampled.append(cur_ind)
                used.add(cur_ind)
                n_sampled += self.sample_count_f(corpus[cur_ind])
            
        return (corpus[i] for i in sampled)
    
    
    def initial_sample(self, corpus):
        s1 = rand.randint(len(corpus))
        print("FIRST SAMPLE: ", s1)
        return [s1], {s1}, [corpus[s1]]
    
    
    def is_next_sample(self, ind, used_inds, corpus, hist, hist_len):
        sampled_sent = corpus[ind]
        
        if not sampled_sent:
            return False
        
        if ind in used_inds:
            return False
            
        cur_disallowed = reduce(np.union1d, hist)
        
        if np.intersect1d(sampled_sent, cur_disallowed).size > 0:
            return False
            
        if len(hist) >= hist_len:
            hist.pop(0)
        hist.append(sampled_sent)
        
        return True
        
    
    
    def __repr__(self):
        return "SpeakerRestrictionFilterRandomised_" +\
                str(self.n) + "_" + str(self.m)
    



#    def sample_old(self, corpus, n_samples=None, history_len=None):
#        if not n_samples:
#            n_samples = self.n
#        if not history_len:
#            history_len = self.m
#        
#        corpus_len = len(corpus)
#        s1 = rand.randint(corpus_len)
#        sampled = [s1]
#        used = {s1}
#        
#        cur_hist = [corpus[s1]]
#        
#        while len(sampled) < n_samples:
#            cur_ind = rand.randint(corpus_len)
#            cur_sent = corpus[cur_ind]
#            
#            if cur_ind in used:
#                continue
#            
#            cur_disallowed = reduce(np.union1d, cur_hist)
#            if np.intersect1d(cur_sent, cur_disallowed).size > 0:
#                continue
#            
#            if len(cur_hist) >= history_len:
#                cur_hist.pop(0)
#            cur_hist.append(cur_sent)
#            
#            sampled.append(cur_ind)
#            used.add(cur_ind)
#            
#        return (corpus[i] for i in sampled)