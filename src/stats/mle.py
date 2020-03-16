# -*- coding: utf-8 -*-
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel,\
        GenericLikelihoodModelResults

from statsmodels.nonparametric.smoothers_lowess import lowess


from scipy.special import zeta
from scipy.stats import binom

import pickle

import numpy as np
lg = np.log10


class Mandelbrot(GenericLikelihoodModel):
    def to_pickle(self, filename, remove_data=True):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        if not self.fit_result:
            raise ValueError("No fit result registered yet; pickling pointless!")
        
        if remove_data:
            self.fit_result.model = None
            self.fit_result.exog = None
            self.fit_result.endog = None
            
            
        with open(filename, "wb") as handle:
            pickle.dump(self.fit_result, handle)   
            
    @classmethod
    def from_pickle(cls, filename, to_class=False, frequencies=None, 
                    ranks=None, **kwargs):
        
        if not filename.endswith(".pkl"):
            filename += ".pkl"        
        with open(filename, "rb") as handle:
            fit_res = pickle.load(handle)
            
        if not to_class:
            return fit_res
        
        if (frequencies is None) or (ranks is None):
            raise ValueError("Mandelbrot class can only be instatiated with" 
                              "frequencies and ranks given!")
            
        mandel = cls(frequencies, ranks, **kwargs)
        fit_res.model = mandel
        mandel.register_fit(fit_res)
        return mandel
            
        
    
    def __init__(self, frequencies, ranks, **kwargs):
        if not len(frequencies) == len(ranks):
            raise ValueError("NOT THE SAME NUMBER OF RANKS AND FREQS!")
        
        frequencies = np.asarray(frequencies)
        ranks = np.asarray(ranks)
        
        self.n_obs = np.sum(frequencies)
        
        super().__init__(endog=frequencies, exog=ranks, **kwargs)
        self.fit_result = None
    

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
        
#        if alpha > 10 or beta > 20:
#            return -np.inf
        
        if alpha < 1.0 or beta < 0.0:
            return -np.inf
        
        # no need to calculate P(r) when observed f(r) was zero
        log_probs = -alpha*lg(beta+rs) - lg(zeta(alpha, q=beta+1.))
        log_probs = log_probs.reshape(-1, )
        return np.sum(fs * log_probs) - beta**5
    
    
    def register_fit(self, fit_result, overwrite=False):
        if not self.fit_result is None and not overwrite:
            raise ValueError("A fit result is already registered and overwrite=False!")
            
        self.fit_result = fit_result
        self.optim_params = fit_result.params
        self.pseudo_r_squared = self.pseudo_r_squared(self.optim_params)
        self.SE, self.SE_relative = fit_result.bse, fit_result.bse/self.optim_params
        self.BIC, self.BIC_relative = fit_result.bic,\
                            (-2*self.null_loglike())/fit_result.bic
    
    def print_result(self, string=False):
        if self.fit_result is None:
            raise ValueError("Register a fitting result first!")

        def format_x(x):
            return float('{0:.3g}'.format(x))


        s = "="*50
        s += "\n" + "MANDELBROT"
        s += "\n" + "  Optimal Parameters " + str(tuple(map(format_x, self.optim_params)))
        
        s += "\n" + "  Standard Error [relative]: " + str(tuple(map(format_x, self.SE))) +\
              ", [" + str(tuple(map(format_x, self.SE_relative))) + "]"
        
        s += "\n" + "  Pseudo R^2: " + str(format_x(self.pseudo_r_squared))
        
        s += "\n" + "  BIC [relative]: " + str(format_x(self.BIC)) +\
              ", [" + str(format_x(self.BIC_relative)) + "]"
        s += "\n" + "="*50
        
        if string:
            return s
        
        print(s)
    
    
    def null_loglike(self, epsilon=1e-10):
        return self.loglike((1.+epsilon, 0.0))
    
    def pseudo_r_squared(self, params):
        return 1-self.loglike(params)/self.null_loglike()
    
    
    def predict(self, params, ranks=None, freqs=True, n_obs=None, 
                correct_for_finite_domain=True):
        if ranks is None:
            ranks = self.exog
        ranks = np.asarray(ranks)
        
        if n_obs is None:
            n_obs = self.n_obs
            
        alpha, beta = params
        pred_probs = self.prob(params, ranks=ranks, log=False)
        
        if correct_for_finite_domain:
            if not freqs:
                raise NotImplementedError("Correction for "\
                                          "finite domain not implemented with probabilities!")
            return pred_probs*(n_obs/np.sum(pred_probs))
        
        if freqs:
            return n_obs*pred_probs
        
        return pred_probs









class Mandelbrot2(GenericLikelihoodModel):
    def to_pickle(self, filename, remove_data=True):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        if not self.fit_result:
            raise ValueError("No fit result registered yet; pickling pointless!")
        
        if remove_data:
            self.fit_result.model = None
            self.fit_result.exog = None
            self.fit_result.endog = None
            
            
        with open(filename, "wb") as handle:
            pickle.dump(self.fit_result, handle)   
            
    @classmethod
    def from_pickle(cls, filename, to_class=False, frequencies=None, 
                    ranks=None, **kwargs):
        
        if not filename.endswith(".pkl"):
            filename += ".pkl"        
        with open(filename, "rb") as handle:
            fit_res = pickle.load(handle)
            
        if not to_class:
            return fit_res
        
        if (frequencies is None) or (ranks is None):
            raise ValueError("Mandelbrot class can only be instatiated with" 
                              "frequencies and ranks given!")
            
        mandel = cls(frequencies, ranks, **kwargs)
        fit_res.model = mandel
        mandel.register_fit(fit_res)
        return mandel
            
        
    
    def __init__(self, frequencies, ranks, regulariser, **kwargs):
        if not len(frequencies) == len(ranks):
            raise ValueError("NOT THE SAME NUMBER OF RANKS AND FREQS!")
        
        frequencies = np.asarray(frequencies)
        ranks = np.asarray(ranks)
        
        self.n_obs = np.sum(frequencies)
        
        self.regulariser = regulariser
        
        super().__init__(endog=frequencies, exog=ranks, **kwargs)
        self.fit_result = None
    

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
        
#        if alpha > 10 or beta > 20:
#            return -np.inf
        
        if alpha < 1.0 or beta < 0.0:
            return -np.inf
        
        # no need to calculate P(r) when observed f(r) was zero
        log_probs = -alpha*lg(beta+rs) - lg(zeta(alpha, q=beta+1.))
        log_probs = log_probs.reshape(-1, )
        
        return self.regulariser(np.sum(fs*log_probs), alpha, beta)
#        return np.sum(fs * log_probs) - beta**5
    
    
    def register_fit(self, fit_result, overwrite=False):
        if not self.fit_result is None and not overwrite:
            raise ValueError("A fit result is already registered and overwrite=False!")
            
        self.fit_result = fit_result
        self.optim_params = fit_result.params
        self.pseudo_r_squared = self.pseudo_r_squared(self.optim_params)
        self.SE, self.SE_relative = fit_result.bse, fit_result.bse/self.optim_params
        self.BIC, self.BIC_relative = fit_result.bic,\
                            (-2*self.null_loglike())/fit_result.bic
    
    def print_result(self, string=False):
        if self.fit_result is None:
            raise ValueError("Register a fitting result first!")

        def format_x(x):
            return float('{0:.3g}'.format(x))


        s = "="*50
        s += "\n" + "MANDELBROT"
        s += "\n" + "  Optimal Parameters " + str(tuple(map(format_x, self.optim_params)))
        
        s += "\n" + "  Standard Error [relative]: " + str(tuple(map(format_x, self.SE))) +\
              ", [" + str(tuple(map(format_x, self.SE_relative))) + "]"
        
        s += "\n" + "  Pseudo R^2: " + str(format_x(self.pseudo_r_squared))
        
        s += "\n" + "  BIC [relative]: " + str(format_x(self.BIC)) +\
              ", [" + str(format_x(self.BIC_relative)) + "]"
        s += "\n" + "="*50
        
        if string:
            return s
        
        print(s)
    
    
    def null_loglike(self, epsilon=1e-10):
        return self.loglike((1.+epsilon, 0.0))
    
    def pseudo_r_squared(self, params):
        return 1-self.loglike(params)/self.null_loglike()
    
    
    def predict(self, params, ranks=None, freqs=True, n_obs=None, 
                correct_for_finite_domain=True):
        if ranks is None:
            ranks = self.exog
        ranks = np.asarray(ranks)
        
        if n_obs is None:
            n_obs = self.n_obs
            
        alpha, beta = params
        pred_probs = self.prob(params, ranks=ranks, log=False)
        
        if correct_for_finite_domain:
            if not freqs:
                raise NotImplementedError("Correction for "\
                                          "finite domain not implemented with probabilities!")
            return pred_probs*(n_obs/np.sum(pred_probs))
        
        if freqs:
            return n_obs*pred_probs
        
        return pred_probs
























































class Heap(GenericLikelihoodModel):
    def to_pickle(self, filename, remove_data=True):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        if not self.fit_result:
            raise ValueError("No fit result registered yet; pickling pointless!")
        
        if remove_data:
            self.remove_data()

        with open(filename, "wb") as handle:
            pickle.dump(self.fit_result, handle)
            
    def remove_data(self):
        self.fit_result.model = None
        self.fit_result.exog = None
        self.fit_result.endog = None        
            
    @classmethod
    def from_pickle(cls, filename, to_class=False, ns_types=None, 
                    ns_tokens=None, **kwargs):
        if not filename.endswith(".pkl"):
            filename += ".pkl"        
        with open(filename, "rb") as handle:
            fit_res = pickle.load(handle)
            
        if not to_class:
            return fit_res
        
        if (ns_types is None) or (ns_tokens is None):
            raise ValueError("Heap class can only be instatiated with" 
                              "frequencies and ranks given!")
            
        heap = cls(ns_types, ns_tokens, **kwargs)
        fit_res.model = heap
        heap.register_fit(fit_res)
        return heap
    
    def __init__(self, ns_types, ns_tokens, **kwargs):
        if not len(ns_types) ==  len(ns_tokens):
            raise ValueError("N TYPES AND N TOKENS OF DIFFERENT LENGTH!")
            
        self.n_obs = len(ns_types)
        ns_types = np.asarray(ns_types)
        ns_tokens = np.asarray(ns_tokens)
        
        if ns_tokens[0] == 0:
            ns_types[0] = 1
            ns_tokens[0] = 1
        
        self.ttrs = ns_types/ns_tokens
#        self.log_ttrs = lg(ns_types)/lg(ns_tokens)
        
        super().__init__(endog=ns_types, exog=ns_tokens, **kwargs)
        self.fit_result = None
        
    def loglike(self, params):        
        K, beta = params
        
        if beta > 1. or K < 1:
            return -np.inf
        
        types, tokens = self.endog, self.exog
                
        # V(n) = K*n**beta
        projected_n_types = K*tokens**beta 
        p = .5

        # binom mode = floor((n+1)*p),
        # so binom_n = floor(1/p*n)    
        binom_ns =  np.floor((1/p)*projected_n_types)

        logprobs = list(binom.logpmf(t, bn, p)[0] 
                    for t, bn in zip(types, binom_ns))        
        logprobs_clipped = np.clip(logprobs, -10**6, 0)
        return sum(logprobs_clipped)# - beta*1000

    def null_loglike(self):
        types, tokens = self.endog, self.exog
        projected_n_types = np.median(self.ttrs)*tokens.reshape((-1, ))
        p = .5
        binom_ns =  np.floor((1/p)*projected_n_types)
        logprobs = list(binom.logpmf(t, bn, p)
                    for t, bn in zip(types, binom_ns))
        logprobs_clipped = np.clip(logprobs, -10**6, 0)        
        return sum(logprobs_clipped)
    
    def fit(self, start_params=None, method="powell", **kwargs):
        if start_params is None:
            start_params = (10, 0.75) 
        return super().fit(start_params=start_params, method=method, **kwargs)
    
    def predict(self, params, ns_tokens=None):
        if ns_tokens is None:
            ns_tokens = self.exog
        ns_tokens = np.asarray(ns_tokens)
        
        K, beta = params
        return K*ns_tokens**beta
    
    def register_fit(self, fit_result, overwrite=False):
        if not self.fit_result is None and not overwrite:
            raise ValueError("A fit result is already registered and overwrite=False!")
            
        self.fit_result = fit_result
        self.optim_params = fit_result.params
        self.pseudo_r_squared = self.pseudo_r_squared(self.optim_params)
        self.SE, self.SE_relative = fit_result.bse, fit_result.bse/self.optim_params
        self.BIC, self.BIC_relative = fit_result.bic,\
                            (-2*self.null_loglike())/fit_result.bic
    
    def print_result(self, string=False):
        if self.fit_result is None:
            raise ValueError("Register a fitting result first!")

        def format_x(x):
            return float('{0:.3g}'.format(x))


        s = "="*50
        s += "\n" + "HEAP"
        s += "\n" + "  Optimal Parameters " + str(tuple(map(format_x, self.optim_params)))
        
        s += "\n" + "  Standard Error [relative]: " + str(tuple(map(format_x, self.SE))) +\
              ", [" + str(tuple(map(format_x, self.SE_relative))) + "]"
        
        s += "\n" + "  Pseudo R^2: " + str(format_x(self.pseudo_r_squared))
        
        s += "\n" + "  BIC [relative]: " + str(format_x(self.BIC)) +\
              ", [" + str(format_x(self.BIC_relative)) + "]"
        s += "\n" + "="*50
        
        if string:
            return s
        
        print(s)
        
    def pseudo_r_squared(self, params):
        return 1-self.loglike(params)/self.null_loglike()