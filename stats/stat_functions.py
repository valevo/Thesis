# -*- coding: utf-8 -*-

from collections import Counter, defaultdict

import matplotlib.pyplot as plt

import numpy as np
import numpy.random as rand
lg = np.log2


def normalise(v):
    v = np.asarray(v)
    return v/np.sum(v)


def get_ranks(words):
    rank_dict = {w: r for r, (w, c) in enumerate(Counter(words).most_common(), 1)}
    return rank_dict


def get_freqs(sample, log=False, vector=False):
    domain, counts = list(zip(*sorted(Counter(sample).items())))
    if log:
        freqs = lg(counts)
    else:
        freqs = np.asarray(counts)

    return freqs if vector else dict(zip(domain, freqs))


def get_probs(sample, log=False, vector=False):
    # n = len(sample)
    domain, counts = list(zip(*sorted(Counter(sample).items())))
    n = sum(counts)
    if log:
        probs = lg(counts) - lg(n)
    else:
        probs = np.asarray(counts) / n

    return probs if vector else dict(zip(domain, probs))
    

def interpolated_probs(sent_sample, log=False, vector=False, base_dist=None):
    if not base_dist:
        base_dist = get_probs((w for s in sent_sample for w in s), log=log)
        
    reduce_func = np.sum if log else np.prod
    if vector:
        return np.asarray([reduce_func([base_dist[w] for w in s]) 
                           for s in sent_sample])
    else:
        return {s: reduce_func([base_dist[w] for w in s]) 
                for s in set(sent_sample)}


def spectrum(source, ranks=True, freqs=True, log=False, lbl=None):
    plot_f = plt.loglog if log else plt.plot

    propensity_f = get_freqs if freqs else get_probs

    if ranks:
        domain, propens = list(zip(*sorted(propensity_f(source).items(), 
                                         reverse=True, 
                                         key=lambda tup: tup[1])))
    else:
        item_propens = propensity_f(source).values()
        domain, propens = list(zip(*Counter(item_propens).most_common()))

    plot_f(np.arange(1, len(domain)+1), propens, '.', label=lbl)
    plt.legend()
    # plt.show()
    return domain, propens



def improved_spectrum(corpus, ranks=True, freqs=True, log=False, lbl=None):
    plot_f = plt.loglog if log else plt.plot
    propensity_f = get_freqs if freqs else get_probs
    
    sub1, sub2 = split_corpus(corpus)#,to_list=True)
    sub1_tokens = (w for s in sub1 for w in s)
    sub2_tokens = (w for s in sub2 for w in s)    
    propens_dict1 = propensity_f(sub1_tokens)
    propens_dict2 = propensity_f(sub2_tokens)

    if ranks:
        merged_propens = [(i, propens_dict2.get(w, 0.)) for i, (w, p) in
                          enumerate(sorted(propens_dict1.items(),
                                           key=lambda t: t[1],
                                           reverse=True))]
                          
        domain, propens = list(zip(*merged_propens))
    else:
        if not freqs:
            raise NotImplementedError("Freq of freq with probabilities not implemented yet!")
        propens1 = set(propens_dict1.values())
        propens_of_propens2 = Counter(propens_dict2.values())
        
        merged = sorted([(p, propens_of_propens2[p]) for p in propens1])        
        domain, propens = list(zip(*merged))
        
    
    plot_f(domain, propens, '.', label=lbl)
    plt.legend()
    # plt.show()
    return domain, propens


def split_corpus(corpus, to_list=False):
    rand_indicators = rand.choice(2, size=len(corpus))
    rand_iter = iter(rand_indicators)
    sub_corp1 = filter(lambda s: next(rand_iter), corpus)
    rand_iter2 = iter(rand_indicators)
    sub_corp2 = filter(lambda s: not next(rand_iter2), corpus)
    
    if to_list:
        return list(sub_corp1), list(sub_corp2)
    return sub_corp1, sub_corp2


def match_len(*multiple_ls):
    min_len = min(map(len, multiple_ls))
    print(min_len)
    for ls in multiple_ls:
        yield ls[:min_len]

def entropy(seq):
    probs = np.asarray(list(map(lambda c: c/len(seq), Counter(seq).values())))
    return -np.sum(probs*np.log2(probs))


        