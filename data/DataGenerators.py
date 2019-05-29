# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical

from collections import Counter, defaultdict

import os


class DataGenerator:
    def __init__(self, data, vocab_size=None, group=False):
        self.vocab_size = vocab_size
        self.n = np.shape(data)[0]
        if group:
            self.data = self.group(data)
            self.n_batches = np.shape(self.data)[0]
        else:
            self.data = data
            self.n_batches = np.shape(data)[0]

    
    def generate_forever(self, permute=True):
        data_gen = self.generate(permute=permute)
        while True:
            yield from data_gen
            data_gen = self.generate(permute=permute)
    
    
    def generate(self, null_elem=0, permute=True):
        if permute:
            cur_data = rand.permutation(self.data)
        else:
            cur_data = self.data
            
        for batch in cur_data:
            
            if permute:
                cur_batch = rand.permutation(batch)
            else:
                cur_batch = batch
                
            cur_batch_padded = self.pad(cur_batch, null_elem=null_elem)
        
            batch_cat = to_categorical(cur_batch, 
                                       num_classes=self.vocab_size)
            if len(batch_cat.shape) < 3:
                dim1, dim3 = batch_cat.shape
                batch_cat = batch_cat.reshape((dim1, 1, dim3))
                
            yield cur_batch_padded, batch_cat
             
    @staticmethod                   
    def pad(mat, null_elem=0):
        nulls = np.asarray([[null_elem]]*np.shape(mat)[0])
        return np.concatenate([nulls, mat], axis=-1)[:, :-1]
    
    def group(self, data):
        ms = [np.shape(row)[0] for row in data]
        lists = {i: [] for i in range(min(ms), max(ms)+1)}

        for s in data:
            lists[len(s)].append(s)
        
        grouped = [np.asarray(ls) for i, ls in sorted(lists.items())]        
        grouped = list(filter(lambda a: a.size > 0, grouped))
        return grouped
    
    
    
    

class DataGenerator2:           
    def __init__(self, data, vocab_size, max_batch_size=32):
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.n = len(data)
        
        self.grouped_data = self.group(data, as_array=True)
        self.n_groups = len(self.grouped_data)
        
        self.n_batches = int(sum(np.ceil(g.shape[0]/max_batch_size)
                                for g in self.grouped_data))
        
        self.generate = self.generate_lo_mem
        
    
    
    def generate_forever(self, permute=False):
        data_gen = self.generate(permute=permute)
        while True:
            yield from data_gen
            data_gen = self.generate(permute=permute)
    
    
    def group(self, data, null_elem=0, as_array=False):        
        null_elem = 0                
        groups = defaultdict(list)
        
        for s in data:
            padded_s = [null_elem] + s
            for i in range(1, len(s)+1):
                groups[i].append(padded_s[:i+1])                
        
        if not as_array:
            return groups
        
        arr = [np.asarray(ls) for i, ls in sorted(groups.items())]
#        arr = list(filter(lambda a: a.size > 0, arr))
        return arr
    
    
    def generate_lo_mem(self, permute=False):
        for group in self.grouped_data:
            m = group.shape[0]
            if m <= self.max_batch_size:
                yield self.prepare_batch(group)
            else:
                l = m // self.max_batch_size
                for j in range(l):
                    cur_batch = group[j*self.max_batch_size:(j+1)*self.max_batch_size]
                    yield self.prepare_batch(cur_batch)
                
                if m - (l*self.max_batch_size) > 0:
                    cur_batch = group[l*self.max_batch_size:]
                    yield self.prepare_batch(cur_batch)
    
    
    def generate_hi_mem(self, permute=False):
        for group in self.grouped_data:
            m = group.shape[0]
            xs, ys = self.prepare_batch(group)
            if m <= self.max_batch_size:
                yield xs, ys
            else:
                l = m // self.max_batch_size
                for j in range(l):
                    yield xs[j*self.max_batch_size:(j+1)*self.max_batch_size],\
                            ys[j*self.max_batch_size:(j+1)*self.max_batch_size]
                yield xs[l*self.max_batch_size: ], ys[l*self.max_batch_size: ]    
#        if len(batch_cat.shape) < 3:
#            dim1, dim3 = batch_cat.shape
#            batch_cat = batch_cat.reshape((dim1, 1, dim3))
            
            
    def prepare_batch(self, group):
        xs = group[:, :-1]
        ys = to_categorical(group[:, -1], num_classes=self.vocab_size)
        if len(ys.shape) < 3:
            d1, d3 = ys.shape
            ys = ys.reshape((d1, 1, d3))
        
        if not xs.size or not ys.size:
            raise ValueError("SIZE IS 0!\n\n"+str(xs)+"\n"+str(ys)+"\n"+\
                             str(xs.shape)+"\n"+str(ys.shape))
        return xs, ys 




    
class StrToIntTranslator:
    def __init__(self, start_at=0, init_ranks=False):
        self.inverse_dict = None
        self.start_at = start_at

        
    def translate_word(self, w):
        pass
    
    def translate(self, sentences):
        for s in sentences:
            yield list(map(self.translate_word, s))
                
    def invert_dict(self, some_dict=None):
        if not some_dict:
            some_dict = self.index_dict
        return {v: k for k, v in some_dict.items()}
        
    def invert_translation(self, translated_sentences):
        if not self.inverse_dict:
            self.inverse_dict = self.invert_dict()
        for s in translated_sentences:
            yield [self.inverse_dict[w] for w in s]
        

class RankTranslator(StrToIntTranslator):
    def __init__(self, start_at=1, unk_at=np.inf):
        self.unk_at = unk_at
        self.index_dict = None
        super().__init__(start_at=start_at)
        
    def _init_ranks(self, sentences):
        word_counts = Counter((w for s in sentences for w in s))
        word_ranks = {w:r for r, (w, c) in 
                      enumerate(word_counts.most_common(), 
                                start=self.start_at)}
        self.least_rank = min(word_ranks.values())
        self.index_dict = word_ranks
        self.inverse_dict = self.invert_dict()
    
    def translate(self, sentences):
        if not self.index_dict:
            self._init_ranks(sentences)
        for s in sentences:
            yield list(map(self.translate_word, s)) 
    
    def translate_word(self, w):
        if not self.index_dict:
            self.init_ranks()
        cur_r = self.index_dict[w]
        if cur_r >= self.unk_at:
            return self.unk_at
        return cur_r
        
            
class OccurrenceTranslator(StrToIntTranslator):
    def __init__(self, start_at=0, unk_at=np.inf):
        self.cur_ind = start_at
        self.unk_at = unk_at
        self.index_dict = dict()
        super().__init__(start_at=start_at)
        
    def translate_word(self, w):
        if not w in self.index_dict:
            if self.cur_ind < self.unk_at:
                self.index_dict[w] = self.cur_ind
                self.cur_ind += 1 
            else:
                self.index_dict[w] = self.unk_at
                return self.unk_at
        return self.index_dict[w] 


import sentencepiece as spm


class SentencePieceTranslator(StrToIntTranslator):
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file)
        
    def translate(self, sentences, tokenised=True):
        if tokenised:
            for s in sentences:
                yield self.sp.EncodeAsIds(" ".join(s))
        else:
            for s in sentences:
                yield self.sp.EncodeAsIds(s)
        
    def translate_back(self, encoded_sentences):
        for s in encoded_sentences:
            yield self.sp.DecodeIds(s)
        
        