# -*- coding: utf-8 -*-

from stats.stat_functions import get_ranks, get_freqs, get_probs, plt,\
                             Counter, rand, np

import seaborn as sb

from stats.plotting import hexbin_plot, simple_scatterplot, multiple_hexbin_plot

import os
import pickle

from scipy.stats import spearmanr

import numpy.random as rand

class ImprovedSpectrum:
    def __init__(self, corpus, split_level="words", ranks=True, freqs=True):        
        self.split_lvl = split_level
        self.ranks = ranks
        self.freqs = freqs
        
        self.rand_split = None

        if split_level not in {"words", "sentences", "articles"}:
            raise NotImplementedError("Only 'words' and 'sentences' implemented!")
        
        
        self.subpropens1 = self.subpropens2 = None
        
        
        if split_level == "articles":
            if not type(corpus[0][0]) == list:
                raise ValueError("CORPUS SEEMS TO HAVE WRONG DEPTH!" + 
                                 str(type(corpus[0][0])))
            
            articles_merged = [[w for s in a for w in s] for a in corpus]
            self.n_tokens = len(self.tokens_from(articles_merged, to_list=True))
            self.domain, self.propens = self.estimate(articles_merged)
        else:
            self.n_tokens = len(self.tokens_from(corpus, to_list=True))
            self.domain, self.propens = self.estimate(corpus)
        
        self.n_zero_counts = sum(1 for c in self.propens if c == 0)
    
    def estimate(self, corpus):
        if self.split_lvl == "words":
            words1, words2 = self.split(self.tokens_from(corpus, to_list=True))
        else:
            sents1, sents2 = self.split(corpus)
            words1, words2 = self.tokens_from(sents1, to_list=True),\
                        self.tokens_from(sents2, to_list=True)
            
        propensity_f = get_freqs if self.freqs else get_probs

        
        if self.ranks:
            propens_dict1 = propensity_f(words1)
            propens_dict2 = propensity_f(words2)
            
            merged_propens = [(i, propens_dict2.get(w, 0.)) for i, (w, p) in
                              enumerate(sorted(propens_dict1.items(),
                                               key=lambda t: t[1],
                                               reverse=True), 1)]
                              
            domain, propens = list(zip(*merged_propens))
        else:
#            if not self.freqs:
#                raise NotImplementedError("Freq of freq with probabilities not implemented yet!")
            
            propens_dict1 = get_freqs(words1)
            propens_dict2 = get_freqs(words2)
            
            
            propens1 = set(propens_dict1.values())
            propens_of_propens2 = propensity_f(propens_dict2.values())
                        
            merged = sorted([(p, propens_of_propens2[p] if p in propens_of_propens2 else 0.0) 
                             for p in propens1])        
            domain, propens = list(zip(*merged))
        
        
        self.subpropens1 = propens_dict1
        self.subpropens2 = propens_dict2

        
        return domain, propens            
            
        
    def split(self, corpus, to_list=False):
        rand_indicators = rand.choice(2, size=len(corpus))
        self.rand_split = rand_indicators
        rand_iter = iter(rand_indicators)
        sub_corp1 = filter(lambda s: next(rand_iter), corpus)
        rand_iter2 = iter(rand_indicators)
        sub_corp2 = filter(lambda s: not next(rand_iter2), corpus)
            
        if to_list:
            return list(sub_corp1), list(sub_corp2)
        return sub_corp1, sub_corp2
        
    
    def tokens_from(self, corpus, to_list=False):
        toks = (w for s in corpus for w in s)
        if to_list:
            return list(toks)
        return toks
    
    
    def plot(self, plot_type="hex", log=True, lbl=None, show=False, **plt_args):
        lbl_str = "$\log$ " if log else ""
        xlbl = lbl_str + ("rank" if self.ranks else "frequency")
        ylbl = lbl_str + ("frequency" if self.freqs else "normalised frequency")
        if plot_type == "hex":
            hexbin_plot(self.domain, self.propens,
                        xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
        elif plot_type == "scatter":
            simple_scatterplot(self.domain, self.propens, log=log,
                               lbl=lbl, xlbl=xlbl, ylbl=ylbl, **plt_args)
            
        if show:
            plt.legend()
            plt.show()
    
    
    def correlate_with(self, other_spectrum, compute_correl=False, plot_correl=False,
                       this_name=None, other_name=None, log=True, show=False, **plt_args):
        min_max_r = min(map(max, [self.domain, other_spectrum.domain]))
        self_propens, other_propens = self.propens[:min_max_r], other_spectrum.propens[:min_max_r]
        
        if plot_correl:
            hexbin_plot(self_propens, other_propens, xlbl=this_name, 
                        ylbl=this_name, log=log, **plt_args)
        if show:
            plt.legend()
            plt.show()
            
        if compute_correl:
            correl = spearmanr(self_propens, other_propens)
            return tuple(map(lambda x: np.round(x, 2),\
                             (correl.correlation, correl.pvalue)))
            
    
            
    def cumulative_mass(self, rank_interval=None, freq_interval=None,
                        as_prob=False):
        if rank_interval and freq_interval:
            raise ValueError("BOTH RANK AND FREQ GIVEN FOR INTERVAL!")
            
        if not rank_interval and not freq_interval:
            return self.n_tokens if self.freqs else 1
    
        lower, upper = rank_interval if rank_interval else freq_interval
        
        x_arr = np.asarray(self.domain) if rank_interval else np.asarray(self.propens)
        relevant_inds = np.argwhere((lower <= x_arr) & 
                                        (x_arr < upper)).reshape(-1)
        
        norm_factor = self.n_tokens if as_prob else 1        
        return sum([self.propens[i] for i in relevant_inds])/norm_factor
        
#        if rank_interval:
#            lower, upper = rank_interval
#            domain_arr = np.asarray(self.domain)
#            relevant_inds = np.argwhere((lower <= domain_arr) & 
#                                        (domain_arr < upper)).reshape(-1)
#            return sum([self.propens[i] for i in relevant_inds])
#        if freq_interval:
#            lower, upper = freq_interval
#            propens_arr = np.asarray(self.propens)
#            relevant_inds = np.argwhere((lower <= propens_arr) &
#                                        (propens_arr < upper)).reshape(-1)    
#            return sum([self.propens[i] for i in relevant_inds])
        
        
    def __repr__(self):
        return "_".join(["ImprovedSpectrum", str(self.n_tokens),
                        self.split_lvl,
                        ("ranks" if self.ranks else "freqs"),
                        ("freq" if self.freqs else "prob")])
                        
        

    
    
    


class ImprovedSpectrumSuite:
    def __init__(self, spectra, names, suite_name=None):
        self.spectra = spectra
        self.names = names
        self.suite_name = "" if suite_name is None else str(suite_name)
    
    def get_domains(self, as_dict=False):
        domains = [spec.domain for spec in self.spectra]
        if as_dict:
            return dict(zip(self.names, domains))
        return domains
    
    def get_propens(self, as_dict=False):
        propens = [spec.propens for spec in self.spectra]
        if as_dict:
            return dict(zip(self.names, propens))
        return propens
    
    def unify_domains(self):
        min_max_r = min(list(map(max, self.get_domains())))
        
        uni_domain = np.arange(1, min_max_r+1)

        uni_propens = np.asarray([ps[:min_max_r] for ps in self.get_propens()])

        return uni_domain, uni_propens              
        
    
        
    def plot(self, plot_type="scatter", log=True, show=False, 
             unify_domains=True, **plt_args):
        
        if plot_type == "scatter":
            for nm, spec in zip(self.names, self.spectra):
                spec.plot(plot_type="scatter", log=log, lbl=nm, show=False, **plt_args)
            
        elif plot_type == "scatter_band":
            uni_domain, uni_propens = self.unify_domains()
            
            means, mins, maxs = np.median(uni_propens, axis=0),\
                        np.min(uni_propens, axis=0),\
                        np.max(uni_propens, axis=0)
            
            xlbl, ylbl = "$\log$ rank", "$\log$ frequency"
            
            
            
            rand_ind = rand.randint(len(self.spectra), dtype="int")

            print("SCATTER_BAND RAND_IND:", rand_ind)
            simple_scatterplot(uni_domain, uni_propens[rand_ind], log=log, color="orange",
                               lbl=None, xlbl=xlbl, ylbl=ylbl, linewidth=0.)             
            
#            simple_scatterplot(uni_domain, means, log=log, color="blue",
#                               lbl=None, xlbl=xlbl, ylbl=ylbl, alpha=0.5, linewidth=0.) 
#          
            
            plt.fill_between(uni_domain, mins, maxs, 
                             alpha=0.2, facecolor="blue", linewidth=1.5,
                             interpolate=False)
        
        elif plot_type == "scatter_all":
            if "alpha" in plt_args:
                alphas = plt_args["alpha"]
                
            
            
            rand_ind = rand.randint(len(self.spectra), dtype="int")

            print("RAND IND", rand_ind)
            print("NAMES", self.names)

            uni_domain, uni_propens = self.unify_domains()
            
            means = np.median(uni_propens, axis=0)
            
            xlbl, ylbl = "$\log$ rank", "$\log$ frequency"

            
            for i, (nm, ps, a) in enumerate(zip(self.names, uni_propens, alphas)):
                if i == rand_ind:
                    continue
                simple_scatterplot(uni_domain, ps, log=log, ignore_zeros=True,
                                   lbl=nm, xlbl=xlbl, ylbl=ylbl, alpha=0.5, linewidth=0.)
                
                
            simple_scatterplot(uni_domain, uni_propens[rand_ind], log=log, ignore_zeros=True,
                               lbl=str(self.names[rand_ind]) + " CHOSEN", 
                               xlbl=xlbl, ylbl=ylbl, alpha=1.0, linewidth=0.,
                               color="black")
            
            
            if "plot_median" in plt_args and plt_args["plot_median"]:
                simple_scatterplot(uni_domain, means, log=log, ignore_zeros=True,
                                   lbl=None, xlbl=xlbl, ylbl=ylbl, alpha=1.0, linewidth=0.,
                                   color="black") 
                
        elif plot_type == "hexbin":
            xs = [spec.domain for spec in self.spectra]
            ys = [spec.propens for spec in self.spectra]
            
            print("LEN xs:", len(xs), len(ys))
            xlbl, ylbl = "$\log$ rank", "$\log$ frequency"
            multiple_hexbin_plot(xs, ys, xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)


        if show:
#            plt.legend()
            plt.show()
                
        
            
    def __repr__(self):
        return "_".join(["ImprovedSpectrumSuite", self.suite_name])
    
    
    def to_pickle(self, dir_prefix="./", error_if_exists=False):
        
        folder = dir_prefix + str(self) + "/"
        
        if os.path.exists(folder):
            raise ValueError(folder + " ALREADY EXISTS!")
        
        os.makedirs(folder)
        
        names = map(str, self.names)
        
        for name, spec in zip(names, self.spectra):
            with open(folder + name + ".pkl", "wb") as handle:
                pickle.dump(spec, handle)
    
    @classmethod
    def from_pickle(cls, dir_name, dir_prefix="./", suite_name=None):
        my_name = None
        if suite_name:
            my_name = suite_name
        
        files = os.listdir(dir_prefix + dir_name)
        try:
            names = sorted([int(f.replace(".pkl", "")) for f in files])
        except ValueError:
            names = sorted([f.replace(".pkl", "") for f in files])
        
        specs = []
        for f in names:
            with open(dir_prefix + dir_name + "/" + str(f) + ".pkl", "rb") as handle:
                cur_spec = pickle.load(handle)
                specs.append(cur_spec)        
        return cls(specs, names, suite_name=my_name)