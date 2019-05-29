# -*- coding: utf-8 -*-

"""
This filter samples atypical subsets of the data directly. Typicality is 
measured wrt. Zipf's Law.

:param n: number of sentences to sample


1. â = MLE(data)
2. H = H(P_â) \approx -\sum_{i \in [10^6]} P_â(i)*log(P_â(i))
3. p_{C} = \prod_{i \in C} P_â(i)
4. H-e < -1/n log(p_{C}) < H+e


"""

#%%

from scipy.special import zeta
import numpy as np
from collections import Counter


# a/z(a)*sum_{x \in X}(log(x)/(x^a)) + log(z(a))
def zipf_entropy(alpha, x=10**5):
    term1 = alpha/zeta(alpha)
    rng = np.arange(1, x+1)
    term2 = np.sum(np.log2(rng)/(rng**alpha))
    term3 = np.log2(zeta(alpha))
    return term1*term2+term3
    

#%%

from Filters.Filter import Filter


class TypicalityFilter(Filter):
    def __init__(self, corpus, n, init_sample=True):
        self.n = n
        self.rank_dict = None
        super().__init__(corpus, init_sample)
        
    def resample(self, n=None, reset=False):
        if not n:
            n = self.n
            
        if reset:
            self.n = n
            self._reset()
            
        # translate 
        ranks = self.words_to_ranks()
        corpus_inds = set(range(0, len(self.corpus)))
        
        sampled_sent = rand.choice(corpus_inds, size=1, replace=False)[0]
        corpus_inds.difference_update(sampled_sent)
        
        for j in range(self.n):
            sampled_sent = rand.choice()
        
        
        
        
        
        
        
    def words_to_ranks(self):
        word_counts = Counter(self.tokens(self.corpus))
        word_ranks = {w: r for r, (w, c) in 
                      enumerate(word_counts.most_common(), 1)}
        
        self.rank_dict = word_ranks
        
        return [[word_ranks[w] for w in s] for s in self.corpus]
        


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

def prop_to_empirical_entropy(token_ranks, alpha=None):
    if alpha:
        return np.sum(np.asarray(token_ranks)**alpha)
    return np.sum(np.asarray(token_ranks))
    

def words_to_ranks(corpus):
    word_counts = Counter((w for s in corpus for w in s))
    wr_tuples = [(w,r) for r, (w, c) in 
                  enumerate(word_counts.most_common(), 1)]
    
    word_rank = dict(wr_tuples)
    rank_word = dict(((t[1], t[0]) for t in wr_tuples))
    
        
    return [[word_rank[w] for w in s] for s in corpus], rank_word


def inds_to_ent(corpus, inds, alpha):
    #    print(inds)
    ss = (corpus[i] for i in inds)
    return empirical_entropy([w for s in ss for w in s], alpha=alpha)
    

#%%

rng = np.arange(1, 50)

plt.plot(rng, [zipf_seq_prob([1]*l, 1.3)**(1/l) for l in rng], '--', label="p")
plt.show()
    

#%%

rng = np.arange(1,30)

plt.plot(rng, [empirical_entropy([1]*l, 1.3) for l in rng], '--')
plt.show()



#%%

cur_c = sents[:100000]

cur_c = list(filter(None, cur_c))


cur_c_ranks, r_dict = words_to_ranks(cur_c)

#%%

#lens = map(len, cur_c)
#
#len_grouped = {l: }

len_sorted = sorted(cur_c_ranks, key=len)

len_grouped = groupby(len_sorted, len)

#grouped_ent_sorted = {l: sorted(ss, key=empirical_entropy) 
#                    for l, ss in len_grouped.items()}


for l, ss in len_grouped.items():
    if l < 200:
        rng = np.arange(len(ss))
        plt.plot(rng, sorted(map(empirical_entropy, ss)), '--', label=str(l))
    
#        plt.hist(list(map(empirical_entropy, ss)), bins=min(50, len(ss)))
#plt.axhline(zipf_entropy(1.01))
#plt.legend()
plt.show()



#%%

ent_sorted = sorted(cur_c_ranks, key=empirical_entropy, reverse=True)

rng = np.arange(len(ent_sorted))
plt.plot(rng, list(map(empirical_entropy, ent_sorted)))
plt.vlines(np.arange(1000, 20000, 3000), ymin=7, ymax=25)
plt.axhline(zipf_entropy(1.15))
plt.show()

#%%

rng = np.linspace(1.01, 2.5)
ents = [empirical_entropy([w for s in cur_c_ranks for w in s], alpha=a) for a in rng]
plt.plot(rng, ents, '.')
plt.show()

#%%

ent_sorted = sorted(cur_c_ranks, key=empirical_entropy, reverse=True)


print("sorted")

from Filters.UniformFilter import UniformFilter


for m in reversed(np.arange(1000, 20000, 3000)):

    less_sents = [tuple(map(r_dict.__getitem__, s_rs)) 
                    for s_rs in ent_sorted[:m]]
    #words_from_less_sents = [w_ind for s_rs in ent_sorted[:m] for w_ind in s_rs]
    spectrum(words_from_sents(less_sents), log=True, 
             lbl=inds_to_ent(ent_sorted, range(0, m), alpha=1.15))

    uf = UniformFilter(cur_c, m)
    spectrum(list(uf.tokens()),  log=True, lbl="unif")
    
    plt.title(str(m))
    plt.show()



#%%
    
rng = np.arange(1000, 90000, 1000)

    
emp_ents_sorted = [inds_to_ent(ent_sorted, range(0, j), 1.15) 
                        for j in rng]

plt.plot(rng, emp_ents_sorted, '.')
plt.show()
#%%

ents = []
rng = np.arange(5000, 30000, 1000)
p = np.arange(len(ent_sorted), 1, -1)
p = p/np.sum(p)

emp_ents_sorted = list(map(empirical_entropy, ent_sorted))
p = np.asarray(emp_ents_sorted)**10
p = p/np.sum(p)

#%%

sampled_inds = []

for _ in range(5):
    ents = []
    for j in rng:
        cur_inds = rand.choice(len(ent_sorted), size=j, 
                              replace=False, p=p)
        
        sampled_inds.append(cur_inds)
        
        ents.append(inds_to_ent(ent_sorted, cur_inds, 1.15))
        
    
    plt.plot(rng, ents, '--', label="sampled")
    print(".")
plt.show()


#%%
    
population = set(range(len(cur_c_ranks)))
    
num_seed = 100
    
# this is a vector
first_samples = rand.choice(list(population), size=num_seed, replace=False)
    
sampled_inds = list(first_samples)
    
population.difference_update(sampled_inds)
    
    
cur_ent = inds_to_ent(sampled_inds, alpha=1.15)


#%%

ents = []

n = 1000

nws = []

for j in range(n):
    if j % 100 == 0:
        print("---", j, "---")
        print(cur_ent, "\t")
        
    next_sample = rand.choice(list(population), size=1, replace=False)
    next_ent = inds_to_ent(sampled_inds + [int(next_sample[0])])
    num_while = 0
    while next_ent <= cur_ent:
        num_while += 1
        next_sample = rand.choice(list(population), size=1, replace=False)
        next_ent = inds_to_ent(sampled_inds + [int(next_sample[0])])
    
    nws.append(num_while)
    cur_ent = next_ent
    ents.append(cur_ent)
    population.difference_update(next_sample)
    sampled_inds.append(next_sample[0])    
    


"""
inside while:
    sample multiple sentences at once
    sort by entropy
    take max/take one that increases entropy
"""

#%%
    
rng = np.arange(len(ents))

plt.plot(rng, ents, '.')
plt.plot(rng, np.sqrt(rng)+15, '--')
plt.show()


#%%

from Filters.UniformFilter import UniformFilter

uf = UniformFilter(cur_c, len(sampled_inds))

spectrum(list(uf.tokens()),  log=True, lbl="unif")


sampled_sents = [cur_c[i] for i in sampled_inds]

spectrum(words_from_sents(sampled_sents), log=True)

plt.show()






#%% loop v2

ents = []

n = 3

k = 20

nws = []

for j in range(n):
    if j % 100 == 0:
        print("---", j, "---")
        print(cur_ent, "\t")
        
    next_sample = rand.choice(list(population), size=1, replace=False)
    next_ent = inds_to_ent(sampled_inds + [int(next_sample[0])])
    
    num_while = 0
    while next_ent <= cur_ent:
#        print("inside while")
        num_while += 1
        
        next_sample = rand.choice(list(population), size=k, replace=False)
        
        
        cand_next_ents = sorted([(i, inds_to_ent(sampled_inds + [i])) 
                                for i in next_sample], key=lambda tup: tup[1])
        print(cand_next_ents)
        max_ent = max(cand_next_ents, key=lambda tup: tup[1])
        
        
        
        next_ent = max_ent[1]
    
        
        
##        print(type(next_sample), next_sample)
#        
#        candidate_ents = [(i, inds_to_ent([i])) for i in next_sample]
#        highest_ent_cand = max(candidate_ents, key=lambda tup: tup[1])
#        
#        print(sorted([inds_to_ent([i]) for i in next_sample]))
        
#        print(candidate_ents)
#        print(highest_ent_cand)
        
#        next_ent = inds_to_ent(sampled_inds + [highest_ent_cand[0]])
        
        
        print(cur_ent, next_ent)
    print(num_while)
    cur_ent = next_ent
    ents.append(cur_ent)
    population.difference_update(next_sample)
    sampled_inds.append(next_sample[0])
    print("\n\n")
    


"""
inside while:
    sample multiple sentences at once
    sort by entropy
    take max/take one that increases entropy
"""



#%%


def inds_to_ent(inds, alpha=1.1):
    ss = [cur_c_ranks[i] for i in inds]
    return empirical_entropy([w for s in ss for w in s], alpha=alpha)
 

m = 5000
k = 10

ms = []

n_seed = 10
seed_samples = list(rand.choice(len(cur_c_ranks), size=n_seed, replace=False))
seed_ent = inds_to_ent(seed_samples)

max_ent = seed_ent

cur_ent = seed_ent
inds_greater = []

for j in range(m):
#    if j % 100 == 0:
#        print("---", j, "---")
#        print(max_ent)
#        print()
    
    samples = rand.choice(len(cur_c_ranks), size=k, replace=False)
    
    ents = [inds_to_ent(seed_samples + [i]) for i in samples]
    
    sorted_ents = sorted(samples)
    
    max_ent = max(ents)
    ms.append(max_ent)
    
    if max_ent > cur_ent:
        inds_greater.append(j)
        cur_ent = max_ent
        
    

    
    
    
#%% 

plt.hist(ms, bins=50)
#plt.xlim((13.05, 13.07))
plt.show()