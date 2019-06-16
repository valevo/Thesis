# -*- coding: utf-8 -*-

from stats.stat_functions import get_ranks, get_freqs, get_probs, plt,\
                             Counter, rand, np

import seaborn as sb

from stats.plotting import hexbin_plot, simple_scatterplot, multiple_hexbin_plot

import os
import pickle

from scipy.stats import spearmanr

import numpy.random as rand

lg = np.log10



def residuals(preds, true_propens, log=True, rm_0=True):
#    if rm_0:
#        preds, true_propens = [remove_zeros_numpy(p1, p2) 
#                                for p1, p2 in zip(preds, true_propens)]
        
    if log:
        log_propens = lg(true_propens)
        ratios = lg(preds) - log_propens
    else:
        ratios = np.asarray(preds)/np.asarray(true_propens)
    ratios[np.isinf(ratios)] = 0 # lg(1e-10) if log else 1e-10
    return ratios








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
        xlbl = lbl_str + "$r(w)$" # ("rank" if self.ranks else "frequency")
        ylbl = lbl_str + "$f(w)$" #("frequency" if self.freqs else "normalised frequency")
        if plot_type == "hexbin":
            params = dict(edgecolors="blue", linewidths=0.2, cmap="Blues_r")
            params.update(plt_args)
            hexbin_plot(self.domain, self.propens,
                        xlbl=xlbl, ylbl=ylbl, log=log, **params)
        elif plot_type == "scatter":
            simple_scatterplot(self.domain, self.propens, log=log,
                               lbl=lbl, xlbl=xlbl, ylbl=ylbl, **plt_args)
            
        if show:
#            plt.legend()
            plt.show()
            
            
    
    
    def correlate_with(self, other_spectrum, compute_correl=False, plot_correl=False,
                       this_name=None, other_name=None, log=True, show=False, **plt_args):
        min_max_r = min(map(max, [self.domain, other_spectrum.domain]))
        self_propens, other_propens = self.propens[:min_max_r], other_spectrum.propens[:min_max_r]
        
        if plot_correl:
            params = dict(edgecolors="blue", linewidths=0.2, cmap="Blues_r")
            params.update(plt_args)
                        
            hexbin_plot(self_propens, other_propens,
                        xlbl=this_name, ylbl=other_name, log=log, **params)
        if show:
#            plt.legend()
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
        self.n_specs = len(spectra)
    
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

        uni_propens = [list(ps[:min_max_r]) for ps in self.get_propens()]

        uni_propens = np.asarray(uni_propens)

        return uni_domain, uni_propens              
        
    
        
    def plot(self, plot_type, log=True, show=False, 
             unify_domains=False, preds=None, ind=None, **plt_args):
        
        if plot_type.startswith("residual") and preds is None:
            raise ValueError("Need preds to calculate residuals!")
    
        if unify_domains:
            domain, propens = self.unify_domains()
        else:
            domain, propens = self.get_domains(), self.get_propens()
        
        
#        means, mins, maxs = np.median(propens, axis=0),\
#                            np.min(propens, axis=0),\
#                            np.max(propens, axis=0)
        
        
        
        use_ranks, use_freqs = self.spectra[0].ranks, self.spectra[0].freqs
        lbl_str = "$\log$ " if log else ""
        xlbl = lbl_str + "$r(w)$" if use_ranks else "$f(w)$"
        ylbl = ("" if use_freqs else "normalised ") + lbl_str +\
                    ("$f(w)$" if use_ranks else "$f(f(w))$")
        
        if ind is None:
            ind = rand.randint(len(self.spectra), dtype="int")
        
        
        if plot_type == "hexbin":
            params = dict(edgecolors="red", linewidths=0.2, cmap="Reds_r")
            params.update(plt_args)
            
            d = domain if unify_domains else domain[ind]
            
            hexbin_plot(d, propens[ind],
                        xlbl=xlbl, ylbl=ylbl, log=log, **params)
        elif plot_type == "hexbin_all":
            params = dict(edgecolors="blue", linewidths=0.2, cmap="Blues_r")
            params.update(plt_args)
            
            concat_domain = np.tile(domain, self.n_specs) if unify_domains else np.concatenate(domain)
            concat_propens = np.concatenate(propens)
            
            hexbin_plot(concat_domain, concat_propens, xlbl=xlbl, ylbl=ylbl, 
                        log=log, **params)

        elif plot_type == "residual":
            ylbl = lbl_str + "$\hat{f}(w)/f(w)$"
            params = dict(edgecolors="red", cmap="Reds_r", linewidths=0.2)
            params.update(plt_args)
            
            d = domain if unify_domains else domain[ind]
            resids = residuals(preds, propens[ind], log=False)
            hexbin_plot(d, resids, xlbl=xlbl, ylbl=ylbl, log=log,
                        **params)
            plt.plot(d, np.ones_like(d), '--', linewidth=0.5, color="red")


        elif plot_type == "residual_all":
            ylbl = lbl_str + "$\hat{f}(w)/f(w)$"
            params = dict(edgecolors="blue", cmap="Blues_r", linewidths=0.2)
            params.update(plt_args)
            
            resids = residuals(preds, propens, log=False)
            concat_domain = np.tile(domain, self.n_specs) if unify_domains else np.concatenate(domain)
            concat_resids = np.concatenate(resids)
            hexbin_plot(concat_domain, concat_resids, 
                        xlbl=xlbl, ylbl=ylbl, log=log, **params)
            plt.plot(concat_domain, np.ones_like(concat_domain), '--', linewidth=0.5, color="red")


#        elif plot_type == "means":
#            hexbin_plot(domain, means,
#                        xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
        
        
#        plt.legend()

        if show:
            plt.show()
                
            
        return ind
        
            
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
        if not suite_name:
            suite_name = dir_name.replace("ImprovedSpectrumSuite_", "")
        
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
        return cls(specs, names, suite_name=suite_name)
    




#    def plot(self, plot_type="scatter", log=True, show=False, 
#             unify_domains=True, other_propens=None, ind=None, **plt_args):
#        
#        uni_domain, uni_propens = self.unify_domains()
#        
#        if other_propens is not None:
#            uni_propens = other_propens
#        
#        means, mins, maxs = np.median(uni_propens, axis=0),\
#                            np.min(uni_propens, axis=0),\
#                            np.max(uni_propens, axis=0)
#        
#        lbl_str = "$\log$ " if log else ""
#        xlbl = lbl_str + "rank" # ("rank" if self.ranks else "frequency")
#        ylbl = lbl_str + "frequency" #("frequency" if self.freqs else "normalised frequency")
#        
#        if ind is None:
#            ind = rand.randint(len(self.spectra), dtype="int")
#        
#        if plot_type == "scatter":
#            simple_scatterplot(uni_domain, uni_propens[ind], log=log,
#                               xlbl=xlbl, ylbl=ylbl, linewidth=0., **plt_args)  
#
#        elif plot_type == "scatter_all":
#            for i, (nm, ps) in enumerate(zip(self.names, uni_propens)):
#                simple_scatterplot(uni_domain, ps, log=log, ignore_zeros=True,
#                                   lbl=str(nm), xlbl=xlbl, ylbl=ylbl, alpha=0.5, 
#                                   linewidth=0., **plt_args)
#            
#        elif plot_type == "band":
#            params = dict(alpha=0.5, facecolor="grey")
#            params.update(plt_args)            
#            plt.fill_between(uni_domain, mins, maxs, linewidth=1.5,
#                             interpolate=False, **params)
#        
#        elif plot_type == "hexbin":
#            hexbin_plot(uni_domain, uni_propens[ind],
#                        xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
#        
#        elif plot_type == "hexbin_all":
#            xs = [spec.domain for spec in self.spectra]
#            ys = [spec.propens for spec in self.spectra]
#            
#            multiple_hexbin_plot(xs, ys, labels=map(str, self.names), 
#                                 xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
#
#        elif plot_type == "means":
#            hexbin_plot(uni_domain, means,
#                        xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
#        
#        
#        plt.legend()
#
#        if show:
#            plt.show()
#                
#            
#        return ind
        