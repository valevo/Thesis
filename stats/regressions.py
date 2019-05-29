# -*- coding: utf-8 -*-

import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel, LikelihoodModel

from scipy.special import zeta
from scipy.stats import binom


import numpy as np
lg = np.log2

class Mandelbrot(GenericLikelihoodModel):
    def __init__(self, frequencies, ranks, **kwargs):
        if not len(frequencies) == len(ranks):
            raise ValueError("NOT THE SAME NUMBER OF RANKS AND FREQS!")
        
        frequencies = np.asarray(frequencies)
        ranks = np.asarray(ranks)
        
        self.n_obs = np.sum(frequencies)
        
        super().__init__(endog=frequencies, exog=ranks, **kwargs)
    

    def prob(self, params, ranks=None, log=False):
        if ranks is None:
            ranks = self.exog
        
        alpha, beta = params
        
        if log:
            return -alpha*lg(beta+ranks) - lg(zeta(alpha, q=beta+1.))
        else:
            return ((beta + ranks)**(-alpha))/zeta(alpha, q=beta+1.)
        
    
    
    def loglike(self, params):
        rs = self.exog
        fs = self.endog
        alpha, beta = params
        
        if alpha > 10 or beta > 20:
            return -np.inf
        
        if alpha < 1.0 or beta < 0.0:
            return -np.inf
        
        log_probs = -alpha*lg(beta+rs) - lg(zeta(alpha, q=beta+1.))
        # no need to calculate P(r) when observed f(r) was zero
        
        log_probs = log_probs.reshape(-1, )

        
        
        return np.sum(fs * log_probs)
    
    
    def predict(self, params, ranks=None, freqs=True, n_obs=None):
        if not ranks:
            ranks = self.exog
        ranks = np.asarray(ranks)
        
        if not n_obs:
            n_obs = self.n_obs
            
        alpha, beta = params
        pred_probs = self.prob(params, ranks=ranks, log=False)
        
        if freqs:
            return n_obs*pred_probs
        return pred_probs
       
        
    def get_diagnostics(self, fit_result):
        alpha_st_error, beta_std_error = fit_result.bse
        bic = fit_result.bic
        
        confidence_interval = fit_result.conf_int()
        
        
        
    
    
    
#%%
        
class Heap(GenericLikelihoodModel):
    def __init__(self, ns_types, ns_tokens, **kwargs):
        if not len(ns_types) ==  len(ns_tokens):
            raise ValueError("N TYPES AND N TOKENS OF DIFFERENT LENGTH!")
            
        ns_types = np.asarray(ns_types)
        ns_tokens = np.asarray(ns_tokens)
        
#        self.ttrs = ns_types/ns_tokens
        
        
#        self.log_ttrs = np.log(types)/np.log(tokens)
        
        super().__init__(endog=ns_types, exog=ns_tokens, **kwargs)
        
    def loglike(self, params):        
        K, beta = params
        
        print(K, beta)
        
        if beta > 1. or K < 1:
            return -np.inf
        
        types, tokens = self.endog, self.exog
                
        # V(n) = K*n**beta
        projected_n_types = K*tokens**beta
        
#        print(projected_n_types)
                
#        ps = np.ones_like(types)
#        ps = ps/2.
        
        p = .5
        
#        print(ps)

        # binom mode = floor((n+1)*p),
        # so binom_n = floor(1/p*n)
#        binom_ns = np.floor((1/ps)*projected_n_types)
#        binom_ns = np.asarray([np.floor((1/p)*proj) 
#                        for p, proj in zip(projected_n_types)])
    
        binom_ns =  np.floor((1/p)*projected_n_types)
        
#        print(binom_ns)
        
#        logprobs = binom.logpmf(types, binom_ns, ps)
        logprobs = list(binom.logpmf(t, bn, p)[0] 
                    for t, bn in zip(types, binom_ns))
        
        logprobs_clipped = np.clip(logprobs, -10**6, 0)
        
#        print(logprobs)
#        print(logprobs_clipped)
#        print("----------------")
        
        return sum(logprobs_clipped)

    
    def fit(self, start_params=None, method="powell", **kwargs):
        if start_params is None:
            start_params = (1, 0.8) # np.mean(np.log(self.endog)/np.log(self.exog)))
            print("LOG TRR: ", start_params)
        return super().fit(start_params=start_params, method=method, **kwargs)
    
    
    def predict(self, params, ns_tokens=None):
        if not ns_tokens:
            ns_tokens = self.exog
        ns_tokens = np.asarray(ns_tokens)
        
        K, beta = params

        return K*ns_tokens**beta
        
#%% HEAP

K, beta = 1, .5
types, tokens = np.ceil(K*np.arange(5)**beta), np.arange(5)





#%%

types, tokens = heap.counts, heap.domain

heap_model = Heap(types, tokens)        
        
#%%       
#res_heap = heap_model.fit(start_params=np.asarray([1, 0.1]), 
#                 method="powell", full_output=True)    

res_heap = heap_model.fit(start_params=(2, 0.5), skip_hessian=True,)

opt_K, opt_beta = res_heap.params

#%%        
plt.plot(heap.domain, heap.counts, '.')
plt.plot(heap.domain, sheap_model.predict(res_heap.params), '--', color="red")
#plt.savefig("stats/plots/heap_fitted", dpi=150)



#%%

ys = np.arange(0, 100)
obs_y = 50

p = 0.1
for beta in np.arange(0.3, 0.7, 0.1):

    proj_y = obs_y**beta
    print(beta, proj_y)
    binom_n = np.floor((1/p)*proj_y)
    ps = binom.pmf(ys, binom_n, p)

    plt.plot(ys, ps, '.', label=str(beta))
#    plt.axvline(x=proj_y, ymin=0., ymax=1.)
    
plt.legend()
#plt.plot(xs, binom.pmf(xs, ), '.')



#%%

spec = spec_sents
        
i = 0
j = 1000

mandel = Mandelbrot(spec.propens, spec.domain)

res = mandel.fit(start_params=np.asarray([1.0, 1.0]), 
                 method="powell", full_output=True)

opt_alpha, opt_beta = res.params

#%%

preds = mandel.predict((opt_alpha, opt_beta), freqs=True)

preds_correct = preds*(mandel.n_obs/np.sum(preds))

emp_probs = np.asarray(spec.propens)/(mandel.n_obs)#*zeta(opt_alpha, opt_beta+1.))
emp_freqs = np.asarray(spec.propens)

#plt.loglog(spec_arts.domain[i:j], preds, '--', color="green")

plt.loglog(spec.domain, emp_freqs, '.')

plt.loglog(spec.domain, preds_correct, '--', color="green")


#%%

from statsmodels.nonparametric.smoothers_lowess import lowess


#lmodel = lowess(spec_words.propens, spec_words.domain, frac=0.5, it=1,
#                delta=1000,is_sorted=True, return_sorted=False)


x, y = np.asarray(spec_arts.domain[:1000]),\
            np.asarray(spec_arts.propens[:1000])


lmodel = lowess(np.log(y), np.log(x), frac=0.02, it=3,
                delta=1,is_sorted=True, return_sorted=False)


#%%


#plt.loglog(spec_arts.domain[i:j], emp_freqs, '.')

plt.loglog(spec_arts.domain[i:j], preds_correct, '--', color="green")

plt.loglog(x, y, '.')
plt.plot(x, np.exp(lmodel), color="red")



plt.savefig("stats/plots/estimates", dpi=100)

plt.close()




#%%

import mle


