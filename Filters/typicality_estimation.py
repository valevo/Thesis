# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from scipy.special import zeta
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm, tqdm_notebook

from Filters.UniformFilterTokens import UniformFilterTokens

from data.WikiReader import wiki_from_pickles


#%%
def zipf_prob(token_ranks, alpha):
    token_ranks =  np.asarray(token_ranks)
    return token_ranks**(-alpha)/zeta(alpha)


def zipf_seq_prob(token_ranks, alpha, log=False):
    if log:
        norm = np.log2(zeta(alpha))
        numer = -alpha*np.log2(token_ranks)
        return np.sum(numer-norm)
    norm = zeta(alpha)
    indiv_probs = (np.asarray(token_ranks)**(-alpha))/norm
    return np.prod(indiv_probs)


def empirical_entropy(token_ranks, alpha=1.15):
    return -(1/len(token_ranks))*zipf_seq_prob(token_ranks, alpha, log=True)




def words_to_ranks(corpus):
    word_counts = Counter((w for s in corpus for w in s))
    wr_tuples = [(w,r) for r, (w, c) in 
                  enumerate(word_counts.most_common(), 1)]
    
    word_rank = dict(wr_tuples)
    rank_word = dict(((t[1], t[0]) for t in wr_tuples))
    
        
    return [[word_rank[w] for w in s] for s in corpus], rank_word


#%%
lang = "ALS"
wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
sentences = [s for title, s_ls in wiki for s in s_ls]    

#%%
    
cur_corp = rand.permutation(sentences)

cur_corp = list(filter(None, cur_corp))


cur_corp_ranks, r_dict = words_to_ranks(cur_corp)

#%% estimate alpha for entire corpus

cur_words = [w for s in cur_corp_ranks for w in s]
ent_a = lambda a: empirical_entropy(cur_words, alpha=a)


rng = np.arange(1.01, 2.3, 0.1)

plt.plot(rng, list(map(ent_a, rng)), '.')
plt.xlabel("$\\alpha$")
plt.ylabel("$H_\\alpha(D)$")
plt.legend()

#optim_a = minimize(ent_a, 1.01)



#%% estimate alpha across different subsets of the corpus

rng = np.linspace(100, len(cur_corp_ranks), num=20,  dtype="int32")

optim_results = []

for n in tqdm(rng):
    cur_words = [w for s in cur_corp_ranks[:n] for w in s]
    ent_a = lambda a: empirical_entropy(cur_words, alpha=a)
    
    optim_a = minimize(ent_a, 1.01)
    optim_results.append(optim_a)


#%%
    
Hs = [r.fun for r in optim_results]

plt.plot(rng, Hs, '.')
plt.show()

#%%
alphs = [r.x[0] for r in optim_results]

plt.plot(rng, alphs, '.')
plt.xlabel("$n$")
plt.ylabel("$\\hat{\\alpha}$")
plt.show()


#%%

H = optim_a.fun

min_ent = lambda token_ranks: empirical_entropy(token_ranks, optim_a.x[0])

sampled_ents = []

n = 50000

m = 1000
rng = np.arange(m)

for _ in tqdm(rng):    
    u = UniformFilterTokens(cur_corp_ranks, n)
    sampled_ents.append(min_ent(list(u.tokens())))
#%%
sampled_ents = []

m = 2000
rng = np.arange(m)

for n in [50000, 100000, 150000]:
    cur = []    
    for _ in tqdm(rng):
        u = UniformFilterTokens(cur_corp_ranks, n)
        cur.append(min_ent(list(u.tokens())))
    sampled_ents.append(cur)
    

    
#%%
for n, cur in zip([50000, 100000, 150000], sampled_ents):    
    plt.hist(cur, bins=20, histtype="step", label=str(n))
    plt.axvline(x=H, ymax=m/10)
    plt.vlines(np.quantile(cur, [0.25, 0.5, 0.75]),
               ymin=0, ymax=m/10)
plt.legend()
plt.show()




