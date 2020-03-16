# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle    
from filtering.speaker_restriction import filter_speaker_restrict

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    p.add_argument("--hist_len", type=int,
                   help="The history length for the sampling constraint.")
    
    args = p.parse_args()
    return args.lang, args.n_tokens, args.hist_len

if __name__ == "__main__":
    lang, n, hist_len = parse_args()
    m = 10
    
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))
    sents = [s for a in wiki for s in a]
    for m_i in range(m):
        print("started ", m_i, flush=True)
        filtered = list(filter_speaker_restrict(sents, n, hist_len))
        print("filtered ", m_i, flush=True)
    
        name = "_".join((str(n), str(hist_len), str(m_i)))
        corpus_to_pickle(filtered, "results/" + lang + "/SRF", name)