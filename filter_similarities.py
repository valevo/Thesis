#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:47:08 2019

@author: valentin
"""

import os
import pickle

from collections import Counter

import matplotlib.pyplot as plt
#%%

def load_filters(filter_dir, select_f=lambda f: True):
    files = os.listdir(filter_dir)
    for f in filter(select_f, files):
        with open(filter_dir + "/" + f, "rb") as handle:
            cur_filter = pickle.load(handle)
            cleaned_filter = list(filter(None, cur_filter))
            yield cleaned_filter

def jaccard_dist(counter1, counter2):
    intersect_counter = counter1 & counter2
    return sum(intersect_counter.values())/(sum(counter1.values()) +
               sum(counter2.values()) - sum(intersect_counter.values()))
            
            
#%%
            
select_f = lambda f: f.find("_100_") >= 0

fs = list(load_filters("Results/TR/filters", select_f=select_f))

counters = [Counter(tuple(s) for s in f) for f in fs]

word_counters = [Counter(w for s in f for w in s) for f in fs]


#%%

dists = []

for c1 in word_counters:
    for c2 in word_counters:
        if not c1 == c2:
            d = jaccard_dist(c1, c2)
            dists.append(d)
            print(d)


#%%
            
plt.hist(dists, bins=10)
