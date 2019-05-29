# -*- coding: utf-8 -*-

#from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite
#
#from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite

from stats.stat_functions import get_freqs, get_probs

import numpy as np

#from collections import Counter

class CorpusStats:
    def __init__(self, corpus, sent_len_dist=False):
        sents = list(self.sentences_from(corpus))
        self.n_tokens = self.token_stats(sents)
        self.n_types = self.type_stats(sents)
        self.n_articles, self.article_len = self.article_stats(corpus,
                                                               return_dist=False)
        self.n_sentences, self.sent_len = self.sentence_stats(sents,
                                                              return_dist=sent_len_dist)

        
        
    def token_stats(self, corpus):
        return len(self.tokens_from(corpus, to_list=True))
    
    def type_stats(self, corpus):
        return len(self.types_from(corpus))
    
    
    def ttr(self, n_tokens, n_types):
        return n_types/n_tokens
    
    
    def avg_from_dist(self, dist, stddev=True):
        p = 0.5 if stddev else 1.
        
        sum_of_numbers = sum(x*count for x, count in dist.items())
        n = sum(dist.values())
        mean = sum_of_numbers / n

        total_squares = sum(x*x*c for x, c in dist.items())
        mean_of_squares = total_squares / n
        var = mean_of_squares - mean * mean
        
        
        return mean, var**p
    
    def article_stats(self, corpus, return_dist=False):
        art_lens = list(map(len, corpus))
        avg, var = np.mean(art_lens), np.var(art_lens)
        
        if return_dist:
            return len(corpus), get_probs(art_lens)
        
        return len(corpus), (avg, var)
    
    
    def sentence_stats(self, corpus, return_dist=False):
        s_lens = list(map(len, corpus))
        avg, var = np.mean(s_lens), np.var(s_lens)
        
        if return_dist:
            return len(corpus), get_probs(s_lens)
        
        return len(corpus), (avg, var)
    
    
    def sentences_from(self, corpus, to_list=False):
        if to_list:
            return [s for a in corpus for s in a]
        return (s for a in corpus for s in a)
    
    def tokens_from(self, corpus, to_list=False):
        if to_list:
            return [w for s in corpus for w in s]
        return (w for s in corpus for w in s)
    
    
    def types_from(self, corpus):
        return set(self.tokens_from(corpus))
        
        
    def __repr__(self):
        return "CorpusStats"
    
    
    def basic_stats_as_table(self, SI=True):
        def format_x(x, exponent=6):
            m = x/(10**exponent)
            # {0:.3g} formats numbers to be 3 digits 
            return '{0:.3g}'.format(m) + ("e" + str(exponent) if exponent > 0 else "")
        
        def tup_to_str(tup):
            return "(" + ",".join(map(str, tup)) + ")"
    
        article_len = tup_to_str((format_x(self.article_len[0], 0), 
                                  format_x(self.article_len[1], 0)))
        sent_len = self.avg_from_dist(self.sent_len)
        sent_len = tup_to_str((format_x(sent_len[0], 0), 
                               format_x(sent_len[1], 0)))
        
        header = " & ".join(["n_articles", 
                           "(avg article_len, std dev article_len)",
                           "n_sentences",
                           "(avg sentence_len, std dev sentence_len)",
                           "n_tokens",
                           "n_types",
                           "TTR"])
        
        return header + "\n" + " & ".join([format_x(self.n_articles, 5),
                            article_len,
                            format_x(self.n_sentences, 6),
                            sent_len,
                            format_x(self.n_tokens),
                            format_x(self.n_types),
                            format_x(self.ttr(self.n_tokens, self.n_types), exponent=0)])
        
        
        
        
            
            