# -*- coding: utf-8 -*-

import numpy.random as rand

class Corpus:
    def __init__(self, corpus_list):
        self.elements = corpus_list

    def elements(self):
        yield from self.elements
        
    def sentences(self):
        pass

    def tokens(self):
        pass
        
    def types(self):
        return set(self.tokens())
    
    def __len__(self):
        return len(self.elements)

class Articles(Corpus):
    def sentences(self):
        for art in self.elements:
            yield from art
    
    def tokens(self):
        for art in self.elements:
            for sent in art:
                yield from sent
   
    @classmethod
    def subsample(cls, corpus_list, n_tokens):
        def sample(articles, n_tokens):
            n_sampled = 0
            used = set()
            
            while n_sampled < n_tokens:
                cur_ind = rand.randint(len(articles))
                if cur_ind in used:
                    continue

                sampled_art = articles[cur_ind]
                if not sampled_art:
                    continue
                
                yield sampled_art
                n_sampled += sum([len(s) for s in sampled_art])
                used.add(cur_ind)
        
        subcorpus = sample(corpus_list, n_tokens)
        return cls(list(subcorpus))


class Sentences(Corpus):
    def sentences(self):
        yield from self.elements
    
    def tokens(self):
        for sent in self.elements:
            yield from sent
    
    @classmethod
    def subsample(cls, corpus_list, n_tokens):
        def sample(sents, n_tokens):
            n_sampled = 0
            used = set()
            
            while n_sampled < n_tokens:
                cur_ind = rand.randint(len(sents))
                if cur_ind in used:
                    continue
                
                sampled_sent = sents[cur_ind]
                if not sampled_sent:
                    continue
                
                yield sampled_sent
                n_sampled += len(sampled_sent)
                used.add(cur_ind)
        
        subcorpus = sample([s for a in corpus_list for s in a], 
                           n_tokens)
        return cls(list(subcorpus))

class Words(Corpus):    
    def sentences(self):
        raise NotImplementedError("Only tokens available in class Words!")
    
    def tokens(self):
        yield from self.elements
        
    @classmethod
    def subsample(cls, corpus_list, n_tokens):
        def sample(words, n_tokens):
            n_sampled = 0
            used = set()
            
            while n_sampled < n_tokens:
                cur_ind = rand.randint(len(words))
                if cur_ind in used:
                    continue
                
                sampled_word = words[cur_ind]
                if not sampled_word:
                    continue
                
                yield sampled_word
                n_sampled += 1
                used.add(cur_ind)
        
        subcorpus = sample([w for a in corpus_list for s in a for w in s], 
                           n_tokens)
        return cls(list(subcorpus))
        