# -*- coding: utf-8 -*-

import numpy as np

from stats.zipf_estimation import ImprovedSpectrum, ImprovedSpectrumSuite,\
                            plt, rand, Counter
                            
from stats.heap_estimation import ImprovedHeap, ImprovedHeapSuite

from data.WikiReader import wiki_from_pickles

import seaborn as sb

from stats.plotting import remove_zeros, hexbin_plot

#%%

#if __name__ == "__main__":

if True:
    
    print("\nLOADING SENTENCES...")
    
    lang = "ALS"
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
    sentences = [tuple(s) for title, s_ls in wiki for s in s_ls]
    
    n_tokens = len([w for s in sentences for w in s])
    
    print(len(sentences), 
          n_tokens,
          len(set(w for s in sentences for w in s)))
    
    print("NUM FREQS", len(Counter(Counter(w for s in sentences for w in s).values())))
    
    #%%
    
    print("\nWORD vs SENTENCE vs ARTICLE SPECTRUM...")
    
    ranks = True
    
    spec_arts = ImprovedSpectrum([a for t, a in wiki], "articles", ranks=ranks)
    
    print("ARTICLES", len(spec_arts.domain), len(spec_arts.propens))
    print("RAND", len(spec_arts.rand_split), sum(spec_arts.rand_split), 
          sum(spec_arts.rand_split)/len(spec_arts.rand_split))
    print()
    
    spec_arts.plot_hexbin(lbl="articles")
    
    
    spec_sents = ImprovedSpectrum(sentences, "sentences", ranks=ranks)
    
    print("SENTS", len(spec_sents.domain), len(spec_sents.propens))
    print("RAND", len(spec_sents.rand_split), sum(spec_sents.rand_split), 
          sum(spec_sents.rand_split)/len(spec_sents.rand_split))
    print()
    
    spec_sents.plot_hexbin(lbl="sents")
    
    
    
    spec_words = ImprovedSpectrum(sentences, "words", ranks=ranks)
    
    print("WORDS", len(spec_words.domain), len(spec_words.propens))
    print("RAND", len(spec_words.rand_split), sum(spec_words.rand_split), 
          sum(spec_words.rand_split)/len(spec_words.rand_split))
    print()
        
    spec_words.plot_hexbin(lbl="words")
    
    
    plt.legend()
    plt.savefig("stats/words_vs_sents_ALS_hex", dpi=100)    
    plt.close()
    
    
    #%%
    
#    plt.plot(spec_sents.domain[:10000], spec_sents.propens[:10000], 'o')
    sb.regplot(np.log(spec_sents.domain[:10000]), 
               np.log(spec_sents.propens[:10000]),
               lowess=True)
#    plt.yscale("log")
#    plt.xscale("log")
    
    
    #%%
    print("ZIPF SHAPE SIZE COMPARISON")
    
    def tokens_from(corpus, n):
        cur_n = 0
        for s in corpus:
            if cur_n >= n:
                break
            
            cur_n += len(s)
            yield s
    
    
    l = len([w for s in sentences for w in s])
    
    specs = []
    
    for r in np.linspace(0.02, 1, num=7):
        perm_sents = rand.permutation(sentences)
        
        cur_sents = list(tokens_from(perm_sents, int(l*r)))
        
        print(len(cur_sents), len([w for s in cur_sents for w in s]))
        
        cur_spec = ImprovedSpectrum(cur_sents, ranks=True, freqs=False)
        
        specs.append(cur_spec)
        
        print(r, len(cur_spec.domain), len(cur_spec.propens))
        print("RAND", len(cur_spec.rand_split), sum(cur_spec.rand_split), 
          sum(cur_spec.rand_split)/len(cur_spec.rand_split))
        print()
        
        
        plt.loglog(cur_spec.domain, cur_spec.propens, '.', label=str(round(r, 2)) + " " + str(int(l*r)))
        
#        cur_spec.plot(lbl=str(r*100)+"%")
    
        
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc="center")
    
    plt.savefig("stats/plots/prob_size_comparisons_ALS", dpi=200,
                bbox_inches="tight")
    
    plt.close()
    
    
    
    #%%
    
    print("HEAPS LAW")
    
#    n_words = len(set[w for s in sentences for w in s]
    
    rng = (np.linspace(0.0, 1.0, num=100)*n_tokens).astype("int")
    print(rng)
    heap = ImprovedHeap(sentences, rng, freq=None)
    
    
    #%%
    
    rng = (np.linspace(0.1, 1.0, num=5)*n_tokens).astype("int")
    print(rng)
    heaps = ImprovedHeapSuite(sentences, rng, (None, 1, 2))
    
    #%%
    
    for f, h in heaps.heaps.items():
        sb.regplot(heaps.domain, h.counts, 
                   order=1, ci=50, marker=".",
                   label=str(f))
        
    plt.legend()
    plt.show()
        
    

        

        
        
        
#%%
   
xs, ys = list(map(np.log, remove_zeros(spec_words.domain, spec_words.propens)))

     
#sb.jointplot(xs, ys, kind="hex", cmap="cividis", 
#             bins="log", gridsize=50, mincnt=1)

sb.jointplot(xs, ys, kind="reg", order=2)
        
    
#%%

#hexbin_plot(spec_words.domain, spec_words.propens)

n = 10000

sb.regplot(xs[:n], ys[:n], scatter=False, lowess=True)

plt.plot(xs[:n], ys[:n], '.', color="red")

plt.show()

    
#%%

sb.residplot(xs[:1000], ys[:1000], order=2)
plt.show()
    
    
    
    
    