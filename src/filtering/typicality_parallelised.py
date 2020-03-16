# -*- coding: utf-8 -*-

from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs, merge_to_joint

from stats.mle import Mandelbrot

from stats.entropy import mandelbrot_entropy, typicality

import numpy as np
import numpy.random as rand


def get_model(corpus, n):
    big_ranks = compute_ranks(Sentences.subsample(corpus, n))
    freqs = compute_freqs(Sentences.subsample(corpus, n))
    
    joint = merge_to_joint(big_ranks, freqs)

    xs, ys = list(zip(*sorted(joint.values())))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    auto_typ = typicality(mandelbrot, joint)
    
    return big_ranks, mandelbrot, auto_typ


def establish_typical_set(corpus, rank_dict, zipf_model, n, m):
    typicalities = []
    
    for i in range(m):
        sub = Sentences.subsample(corpus, n)
            
        sub_freqs = compute_freqs(sub)
        sub_joints = merge_to_joint(rank_dict, sub_freqs)

        sub_typicality = typicality(zipf_model, sub_joints)        
        typicalities.append(sub_typicality)
            
    mean_typ, std_typ = np.mean(typicalities), np.var(typicalities)**.5    
    return mean_typ, std_typ


def setup_filtering(corpus, big_n, k, m):
    rank_dict, zipf_model, auto_typ = get_model(corpus, big_n)
    mean_typ, std_typ = establish_typical_set(corpus, rank_dict, zipf_model, k, m)
    return zipf_model, rank_dict, mean_typ, std_typ, auto_typ
    

def sent_neg_log_prob(sent, zipf_model, rank_dict):
    ranks = [rank_dict[w] if w in rank_dict else len(rank_dict)+1
            for w in sent]
    log_probs = zipf_model.prob(params=zipf_model.optim_params, 
                                ranks=ranks, log=True)    
    return - np.sum(log_probs)


sep = "■"

# add safety measure against non-halting
def filter_typicality_incremental(sents, zipf_model, rank_dict, auto_typ, n, 
                                  epsilon, direction):
    
    if epsilon > 0 and direction(0, 1):
        raise ValueError("use EITHER epsilon < 0 and direction == < "
                         "OR epsilon > 0 and direction == >")
    
    sampled = 0
    used = set()
    
    theoretical_entropy = mandelbrot_entropy(*zipf_model.optim_params)
    
    cur_nll = 0
    
    num_not_found = 0
    num_iter = 0
    
    while sampled < n:
        num_iter += 1
        
        cur_sample = rand.randint(len(sents))
        if cur_sample in used:
            continue
        
        cur_sent = sents[cur_sample].strip().split(sep)
        
        if not cur_sent:
            continue
        
        coeff = 1/(sampled + len(cur_sent))
        sent_nll = sent_neg_log_prob(cur_sent, zipf_model, rank_dict)
        
        cur_typ = theoretical_entropy - coeff*(cur_nll + sent_nll)
        
        if direction(cur_typ - auto_typ, epsilon):
            used.add(cur_sample)
            sampled += len(cur_sent)
            cur_nll += sent_nll
            
            yield cur_sent
        else:
            num_not_found += 1
#            if num_not_found >= n:
#                print("NUM ITER: ", num_iter)                
#                raise RuntimeError("number of samples has outgrown n! aborting")        
                
                
    print("NUM ITER: ", num_iter)
    print("NUM NOT FOUND: ", num_not_found)