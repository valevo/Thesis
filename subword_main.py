# -*- coding: utf-8 -*-

import os

import sentencepiece as spm

from data.WikiReader import wiki_from_pickles



if __name__ == "__main__":
    folder = "subword/"
    lang = "TR"
    folder += lang + "/"
    
    if not os.path.isdir(folder):
        print("MADE DIR", folder)
        os.makedirs(folder)

    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))        
    sentences = [s for title, s_ls in wiki for s in s_ls]
    
    if not os.path.isfile(folder + lang + ".txt"):
        with open(folder + lang + ".txt", "w") as handle:
            for s in sentences:
                s_str = " ".join(s)
                handle.write(s_str)
                handle.write("\n")
        print("WROTE SENTENCES TO FILE ", folder + lang + ".txt")
    else:
        print("USING EXISTING FILE", folder + lang + ".txt")
        
        
    
    train_file = folder + lang + ".txt"
    vocab_size = "10000"
    model_name = folder + lang + "_" + vocab_size
    model_type = "unigram"
    
    
    print("TRAINING " + model_type + " with V=" + vocab_size + "on\n\t" + train_file)
    
    spm.SentencePieceTrainer.Train(f"--input={train_file}"
                               f" --model_prefix={model_name}"
                               f" --vocab_size={vocab_size}"
                               f" --character_coverage=1.0"
                               f" --model_type=unigram"
                               f" --bos_id=-1"
                               f" --eos_id=-1")
    
    print("DONE TRAINING")
    



#%% TEST
#
#
#sp = spm.SentencePieceProcessor()
#sp.Load(model_name + ".model")
#
##%%
#
#for s in sentences[1000:1050]:
#    s_str = " ".join(s)
#    print(s_str)
#    print(sp.EncodeAsPieces(s_str))
#    print(sp.EncodeAsIds(s_str))
#    print()
    

