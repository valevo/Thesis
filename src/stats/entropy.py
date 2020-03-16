# -*- coding: utf-8 -*-
from scipy.special import zeta
from scipy.misc import derivative

import numpy as np

lg = np.log10

def zipf_entropy(alpha, dx=1e-10):
    if alpha <= 1.0:
        raise ValueError("Entropy undefined for the given parameter:\n" + 
                         str(alpha))
    return alpha*(-derivative(zeta, alpha, dx=dx))/zeta(alpha) + lg(zeta(alpha))

def mandelbrot_entropy(alpha, beta, dx=1e-10):
    if alpha <= 1.0 or beta <= 1.0:
        raise ValueError("Entropy undefined for the given parameters:\n" + 
                         str(alpha) + " and " + str(beta))
    zeta_b = lambda a: zeta(a, beta+1)
    return alpha*(-derivative(zeta_b, alpha, dx=dx))/zeta_b(alpha) + lg(zeta_b(alpha))


def neg_log_likelihood(zipf_model, ranks, freqs):
    mle_params = zipf_model.optim_params
    log_rank_probs = zipf_model.prob(params=mle_params, ranks=ranks, log=True)    
    return -freqs*log_rank_probs
    
    
def empirical_entropy(zipf_model, joint_rank_freqs):
    rs, fs = list(zip(*joint_rank_freqs.values()))
    ranks = np.asarray(rs)
    freqs = np.asarray(fs)
    n = np.sum(freqs)
    return (1/n)*np.sum(neg_log_likelihood(zipf_model, ranks, freqs))


def typicality(zipf_model, joint_rank_freqs):
    mle_params = zipf_model.optim_params
    return mandelbrot_entropy(*mle_params) - empirical_entropy(zipf_model, joint_rank_freqs)



#def construct_typical_set(zipf_model, corpus, m, k):
#    rs, fs = compute_ranks(corpus), compute_freqs(corpus)
#    joints = merge_to_joint(rs, fs)
#    auto_typicality = typicality(zipf_model, joints)
#    
#    typicalities = []
#    
#    for i in range(k):
#        sub1 = Sentences.subsample(corpus, k)
#        sub2 = Sentences.subsample(corpus, k)
#            
#        sub_ranks = compute_ranks(sub1)
#        sub_freqs = compute_freqs(sub2)
#        sub_joints = merge_to_joint(sub_ranks, sub_freqs)
#
#        sub_typicality = typicality(zipf_model, sub_joints)
#        corrected_typicality = sub_typicality - auto_typicality
#        
#        typicalities.append(corrected_typicality)
#        
    
    




    