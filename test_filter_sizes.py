#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:40:47 2019

@author: valentin
"""

import os
import pickle
import time
import numpy as np

lang = "ALS"

d = lambda f, slash=False: "Results/"+lang+"/" + f + ("/" if slash else "")

fs = sorted(os.listdir(d("filters")))

#%%
        
from data.DataGenerators import DataGenerator2, SentencePieceTranslator

def prepare_filter_training2(subword_model, filter_obj):
    translator = SentencePieceTranslator(subword_model)
    
    translated = list(translator.translate(filter_obj))
    
    V_this = len(set(w for s in translated for w in s))
    V_spm = translator.sp.GetPieceSize()
    print(V_this, V_spm, max(set(w for s in translated for w in s)))
    
    gen = DataGenerator2(translated, vocab_size=V_spm, max_batch_size=32)
    return translated, gen

#%%

filters = []
f_trans = []
f_gens = []
cur_m = 4

for f in fs:
    if f.find("_" + str(cur_m) + "_") >= 0:
        print(d("filters/" + f))
        with open(d("filters/" + f), "rb") as handle:
            cur_filter = pickle.load(handle)
            filters.append(cur_filter)
            tr, g = prepare_filter_training2(d("subword/ALS_2000.model"), cur_filter)
            f_trans.append(tr)
            f_gens.append(g)
            
            
    
#%%
us = sorted(os.listdir(d("uniform")))
uniforms = []
u_trans = []
u_gens = []
for f in us:
    print(d("uniform/" + f))
    with open(d("uniform/" + f), "rb") as handle:
        cur_filter = pickle.load(handle)
        uniforms.append(cur_filter)
        tr, g = prepare_filter_training2(d("subword/ALS_2000.model"), cur_filter)
        u_trans.append(tr)
        u_gens.append(g)
        
        



#%% SRF m=1
#
#list(map(lambda f: (len(f), len(list(f.tokens()))), filters))
#Out[6]: 
#[(79773, 1000015),
# (79349, 1000026),
# (79413, 1000021),
# (79570, 1000025),
# (80071, 1000017),
# (79589, 1000018),
# (79553, 1000025),
# (79682, 1000032),
# (79968, 1000034),
# (79602, 1000012)]
#list(map(lambda g: (g.n, g.n_groups, g.n_batches), f_gens))
#Out[5]: 
#[(79773, 871, 54246),
# (79349, 727, 54058),
# (79413, 1541, 54781),
# (79570, 808, 53936),
# (80071, 437, 53614),
# (79589, 638, 53845),
# (79553, 516, 53803),
# (79682, 610, 53939),
# (79968, 323, 53499),
# (79602, 793, 54031)]


#%% SRF m=100

#list(map(lambda g: (g.n, g.n_groups, g.n_batches), f_gens))
#Out[4]: 
#[(174724, 935, 61360),
# (175149, 758, 61115),
# (175210, 1601, 62034),
# (174809, 935, 61326),
# (175199, 1601, 61952),
# (175217, 1815, 62139),
# (175309, 819, 61189),
# (175164, 690, 61063),
# (175264, 1123, 61439),
# (174731, 540, 60946)]
#
#list(map(lambda f: (len(f), len(list(f.tokens()))), filters))
#Out[5]: 
#[(174724, 1000023),
# (175149, 1000011),
# (175210, 1000011),
# (174809, 1000008),
# (175199, 1000011),
# (175217, 1000008),
# (175309, 1000022),
# (175164, 1000014),
# (175264, 1000002),
# (174731, 1000005)]


#%% SRF m=1...100 where j=3
        
#list(map(lambda f: (len(f), len(list(f.tokens()))), filters))
#Out[18]: 
#[(79570, 1000025),
# (94522, 1000012),
# (108401, 1000024),
# (119941, 1000015),
# (131043, 1000012),
# (140887, 1000025),
# (150409, 1000028),
# (158981, 1000007),
# (167317, 1000013),
# (174809, 1000008)]
#
#list(map(lambda g: (g.n, g.n_groups, g.n_batches), f_gens2))
#Out[19]: 
#[(79570, 808, 53936),
# (94522, 545, 54383),
# (108401, 427, 54950),
# (119941, 818, 56090),
# (131043, 540, 56667),
# (140887, 935, 57968),
# (150409, 793, 58550),
# (158981, 758, 59528),
# (167317, 397, 60063),
# (174809, 935, 61326)]


#%% UNIFORM
#list(map(lambda g: (g.n, g.n_groups, g.n_batches), u_gens))
#Out[5]: 
#[(71765, 396, 53455),
# (71762, 471, 53502),
# (71437, 1815, 54901),
# (71411, 473, 53473),
# (71459, 590, 53564),
# (71625, 365, 53497),
# (71698, 603, 53650),
# (71746, 844, 53916),
# (71693, 470, 53548),
# (72028, 437, 53472)]
#
#list(map(lambda f: (len(f), len(list(f.tokens()))), uniforms))
#Out[6]: 
#[(71765, 1000002),
# (71762, 1000019),
# (71437, 1000033),
# (71411, 1000002),
# (71459, 1000011),
# (71625, 1000022),
# (71698, 1000036),
# (71746, 1000003),
# (71693, 1000020),
# (72028, 1000005)]



#%% SRF m=100, mac_batch_size=1
        
#list(map(lambda g: (g.n, g.n_groups, g.n_batches), f_gens))
#
#Out[12]: 
#[(174724, 935, 1941014),
# (175149, 758, 1937830),
# (175210, 1601, 1941525),
# (174809, 935, 1940903),
# (175199, 1601, 1939740),
# (175217, 1815, 1938444),
# (175309, 819, 1937683),
# (175164, 690, 1936964),
# (175264, 1123, 1937057),
# (174731, 540, 1938440)]
        
        

#%%
        
plt.plot(*list(zip(*map(np.shape, u_gens[0].grouped_data))), '.', label="u")
plt.plot(*list(zip(*map(np.shape, f_gens[0].grouped_data))), '.', label="f")
plt.legend()
plt.show()


#%%

plt.hist(list(map(len, filters[0])), histtype="step", label="f",
         bins=50)
plt.hist(list(map(len, uniforms[0])), histtype="step", label="u",
         bins=50)
plt.legend()
plt.show()
