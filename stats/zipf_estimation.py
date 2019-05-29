# -*- coding: utf-8 -*-

from stats.stat_functions import get_ranks, get_freqs, get_probs, plt,\
                             Counter, rand, np

#import seaborn as sb

#from stats.plotting import hexbin_plot, simple_scatterplot


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
    
    
    def plot(self, plt_type="hex", log=True, lbl=None, show=False, **plt_args):
        lbl_str = "$\log$ " if log else ""
        xlbl = lbl_str + ("rank" if self.ranks else "frequency")
        ylbl = lbl_str + ("frequency" if self.freqs else "normalised frequency")
        if plt_type == "hex":
            hexbin_plot(self.domain, self.propens,
                        xlbl=xlbl, ylbl=ylbl, log=log, **plt_args)
        elif plt_type == "scatter":
            simple_scatterplot(self.domain, self.propens, log=log,
                               lbl=lbl, xlbl=xlbl, ylbl=ylbl, **plt_args)
            
        if show:
            plt.legend()
            plt.show()
            
            
    def cumulative_mass(self, rank_interval=None, freq_interval=None):
        if rank_interval and freq_interval:
            raise ValueError("BOTH RANK AND FREQ GIvEN FOR INTERVAL!")
            
        if not rank_interval and not freq_interval:
            return self.n_tokens if self.freqs else 1
        
        if rank_interval:
            lower, upper = rank_interval
            domain_arr = np.asarray(self.domain)
            
            relevant_inds = np.argwhere((lower <= domain_arr) & 
                                        (domain_arr < upper)).reshape(-1)
            return sum([self.propens[i] for i in relevant_inds])
        
        if freq_interval:
            lower, upper = freq_interval
            
            propens_arr = np.asarray(self.propens)
            
            relevant_inds = np.argwhere((lower <= propens_arr) &
                                        (propens_arr < upper)).reshape(-1)
                        
            return sum([self.propens[i] for i in relevant_inds])
        
        
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
        
        
    def plot(self, plot_type="scatter", log=True, show=False, **plt_args):
        if plot_type == "scatter":
            for nm, spec in zip(self.names, self.spectra):
                spec.plot(plot_type="scatter", log=log, lbl=nm, show=False, **plt_args)
            
            if plt.show():
                plt.legend()
                plt.show()
            
    def __repr__(self):
        return "_".join(["ImprovedSpectrumSuite", self.suite_name])