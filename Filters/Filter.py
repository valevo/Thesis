# -*- coding: utf-8 -*-


import numpy as np
import numpy.random as rand
from functools import reduce


class Filter(list):
    def __init__(self, corpus, **kwargs):
        super().__init__(self.sample(corpus))
    
    
    def sample(self, corpus):
        return iter(corpus)
    

    def tokens(self, corpus=None):
        if not corpus:
            corpus = self
        for s in corpus:
            for w in s:
                yield w
                
                
    def types(self, corpus=None):
        if not corpus:
            corpus = self
        return set(self.tokens(corpus))
