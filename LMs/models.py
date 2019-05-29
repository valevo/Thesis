# -*- coding: utf-8 -*-

#%%

from keras.layers import Input, Dense, Embedding, LSTM,\
                    Bidirectional, TimeDistributed, GRU
from keras.models import Model
import keras.backend as K
from keras.models import load_model

import tensorflow as tf

import json

import numpy as np
import numpy.random as rand

from collections import Counter

import random as rn

import os

#%%

#descriptions = {
#        "LanguageModel1": "Embedding, LSTM, LSTM, SoftMax"}


CONSTANT_LAYER_SIZES = {"embedding": 64,
                           "rnn1": 256,
                           "rnn2": 128}

#%%

class LanguageModel(Model):
    def _default_layer_size(self, vocab_size):
        if vocab_size < 1024:
            return {"embedding": round(vocab_size/8),
                           "rnn1": round(vocab_size/6),
                           "rnn2": round(vocab_size/6)}
            
        else:
            return CONSTANT_LAYER_SIZES
        
    def _get_RNN_cell(self, type_str):
        if type_str.lower() == "lstm":
            return LSTM, "LSTM"
        elif type_str.lower() == "gru":
            return GRU, "GRU"
        else:
            raise NotImplementedError("RNN of type " + type_str)
            
            
    def __init__(self, vocab_size, rnn_type, layer_sizes=None, compile_now=True):
        if not layer_sizes:
            layer_sizes = self._default_layer_size(vocab_size)
        else:
            if not isinstance(layer_sizes, dict):
                raise ValueError("layer_sizes expected to be of type dict!")
                
        RNNCell, cell_name = self._get_RNN_cell(rnn_type)
        
        
        # layers    
        embedding = Embedding(input_dim=vocab_size, 
                             output_dim=layer_sizes["embedding"])
        rnn1 = RNNCell(layer_sizes["rnn1"], 
                          return_sequences=True)        
        rnn2 = RNNCell(layer_sizes["rnn2"], 
                          return_sequences=True)
        output_layer = TimeDistributed(Dense(vocab_size, 
                                             activation="softmax"))
        
        # flow
        sentence = Input(shape=(None,))        
        embedded = embedding(sentence)
        processed1 = rnn1(embedded)
        processed2 = rnn2(processed1)
        predictions = output_layer(processed2)
        
        super().__init__(inputs=sentence, outputs=predictions)
        
        
        self.V = vocab_size
        self.params = {"vocab_size": vocab_size,
                       "rnn_type": cell_name,
                       "layer_sizes": layer_sizes}
        
        if compile_now:
            self.compile_default()
        

    def compile_default(self):
        self.compile("adam",
                     loss="categorical_crossentropy",
                     metrics=["categorical_accuracy",
                              self.relative_entropy])

    def entropy(self, y_true, y_pred):
        return -K.sum(y_pred*K.log(y_pred), axis=-1)
    
    def relative_entropy(self, y_true, y_pred):
        return self.entropy(y_true, y_pred)/K.log(K.cast_to_floatx(self.V))



    def __repr__(self):
        return "LanguageModel <" + self.params["rnn_type"] + ">"
    
    
    def save_model_custom(self, dir_to_save):
        self.save_weights(dir_to_save + "/LanguageModel_weights")
        with open(dir_to_save + "/LanguageModel_parameters.json", "w") as handle:
            json.dump(self.params, handle)
            
    @classmethod  
    def from_saved_custom(cls, save_dir, compile_now=False):            
        with open(save_dir + "/LanguageModel_parameters.json", "r") as handle:
            param_dict = json.load(handle)
        
        lm = cls(param_dict["vocab_size"],
                 param_dict["rnn_type"],
                 param_dict["layer_sizes"],
                 compile_now=compile_now)
        
        lm.load_weights(save_dir + "/LanguageModel_weights")
        
        return lm
        
    




def seed_rngs(seed_val):
    # needs to be set before the program starts
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    rand.seed(seed_val)
    rn.seed(seed_val)
    tf.set_random_seed(seed_val)
    
    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)

        
    

#class LanguageModel1(Model):        
#
#    
#    def __init__(self, vocab_size, layer_sizes=None, compile_now=True):
#        if not layer_sizes:
#            layer_sizes = self.default_layer_size(vocab_size)
#        else:
#            if not isinstance(layer_sizes, dict):
#                raise ValueError("layer_sizes expected to be of type dict!")
#            
#        # layers    
#        embedding = Embedding(input_dim=vocab_size, 
#                             output_dim=layer_sizes["embedding"])
#        lstm1 = LSTM(layer_sizes["rnn1"], 
#                          return_sequences=True)        
#        lstm2 = LSTM(layer_sizes["rnn2"], 
#                          return_sequences=True)
#        output_layer = TimeDistributed(Dense(vocab_size, 
#                                             activation="softmax"))
#        
#        # flow
#        sentence = Input(shape=(None,))        
#        embedded = embedding(sentence)
#        processed1 = lstm1(embedded)
#        processed2 = lstm2(processed1)
#        predictions = output_layer(processed2)
#        
#        super().__init__(inputs=sentence, outputs=predictions)
#        
#        self.description = descriptions[str(self)]
#        self.V = vocab_size
#        self.params = {"vocab_size": vocab_size,
#                       "layer_sizes": layer_sizes}
#        
#        if compile_now:
#            self.compile_default()
#            
#    def compile_default(self):
#        self.compile("adam",
#                     loss="categorical_crossentropy",
#                     metrics=["categorical_accuracy",
#                              self.entropy,
#                              self.relative_entropy])
#        
#        
#    def entropy(self, y_true, y_pred):
#        return -K.sum(y_pred*K.log(y_pred), axis=-1)
#    
#    
#    def relative_entropy(self, y_true, y_pred):
#        return self.entropy(y_true, y_pred)/K.log(K.cast_to_floatx(self.V))
#        
#        
#    def __repr__(self):
#        return "LanguageModel1"
#    
#    
#    def save_model_custom(self, dir_to_save):
#        self.save_weights(dir_to_save + "/LanguageModel1_weights")
#        with open(dir_to_save + "/LanguageModel1_parameters.json", "w") as handle:
#            json.dump(self.params, handle)
#            
#    @classmethod  
#    def from_saved_custom(cls, save_dir, compile_now=False):            
#        with open(save_dir + "/LanguageModel1_parameters.json", "r") as handle:
#            param_dict = json.load(handle)
#        
#        lm = cls(param_dict["vocab_size"],
#                        param_dict["layer_sizes"],
#                        compile_now=compile_now)
#        
#        
#        lm.load_weights(save_dir + "/LanguageModel1_weights")
#        
#        return lm
#    
#    
#    
#
#
#class LanguageModelGRU(Model):        
#    def default_layer_size(self, vocab_size):
#        if vocab_size < 1024:
#            return {"embedding": round(vocab_size/8),
#                           "gru1": round(vocab_size/6),
#                           "gru2": round(vocab_size/6)}
#            
#        else:
#            return CONSTANT_LAYER_SIZES
#            
#    
#    def __init__(self, vocab_size, layer_sizes=None, compile_now=True):
#        if not layer_sizes:
#            layer_sizes = self.default_layer_size(vocab_size)
#        else:
#            if not isinstance(layer_sizes, dict):
#                raise ValueError("layer_sizes expected to be of type dict!")
#            
#        # layers    
#        embedding = Embedding(input_dim=vocab_size, 
#                             output_dim=layer_sizes["embedding"])
#        lstm1 = GRU(layer_sizes["rnn1"], 
#                          return_sequences=True)        
#        lstm2 = GRU(layer_sizes["rnn2"], 
#                          return_sequences=True)
#        output_layer = TimeDistributed(Dense(vocab_size, 
#                                             activation="softmax"))
#        
#        # flow
#        sentence = Input(shape=(None,))        
#        embedded = embedding(sentence)
#        processed1 = lstm1(embedded)
#        processed2 = lstm2(processed1)
#        predictions = output_layer(processed2)
#        
#        super().__init__(inputs=sentence, outputs=predictions)
#        
#        self.description = descriptions[str(self)]
#        self.V = vocab_size
#        self.params = {"vocab_size": vocab_size,
#                       "layer_sizes": layer_sizes}
#        
#        if compile_now:
#            self.compile_default()
#            
#    def compile_default(self):
#        self.compile("adam",
#                     loss="categorical_crossentropy",
#                     metrics=["categorical_accuracy",
#                              self.entropy,
#                              self.relative_entropy])
#        
#        
#    def entropy(self, y_true, y_pred):
#        return -K.sum(y_pred*K.log(y_pred), axis=-1)
#    
#    
#    def relative_entropy(self, y_true, y_pred):
#        return self.entropy(y_true, y_pred)/K.log(K.cast_to_floatx(self.V))
#        
#        
#    def __repr__(self):
#        return "LanguageModelGRU"
#    
#    
#    def save_model_custom(self, dir_to_save):
#        self.save_weights(dir_to_save + "/LanguageModelGRU_weights")
#        with open(dir_to_save + "/LanguageModelGRU_parameters.json", "w") as handle:
#            json.dump(self.params, handle)
#            
#    @classmethod  
#    def from_saved_custom(cls, save_dir, compile_now=False):            
#        with open(save_dir + "/LanguageModelGRU_parameters.json", "r") as handle:
#            param_dict = json.load(handle)
#        
#        lm = cls(param_dict["vocab_size"],
#                        param_dict["layer_sizes"],
#                        compile_now=compile_now)
#        
#        
#        lm.load_weights(save_dir + "/LanguageModelGRU_weights")
#        
#        return lm
#
#
#
#
#
#class LanguageModel2(Model):
#    def default_layer_size(self, vocab_size):
#        if vocab_size < 1024:
#            return {"embedding": round(vocab_size/8),
#                           "lstm1": round(vocab_size/6),
#                           "lstm2": round(vocab_size/6)}
#            
#        else:
#            return {"embedding": 200,
#                           "lstm1": 200,
#                           "lstm2": 200}
#    
#    def __init__(self, vocab_size, layer_sizes=None, compile_now=True):
#        super().__init__(name="LanguageModel2")
#        
#        
#        if not layer_sizes:
#            layer_sizes = self.default_layer_size(vocab_size)
#        else:
#            if not isinstance(layer_sizes, dict):
#                raise ValueError("layer_sizes expected to be of type dict!")
#            
#        # layers    
#        self.embedding = Embedding(input_dim=vocab_size, 
#                             output_dim=layer_sizes["embedding"],
#                             input_shape=(None,))
#        self.lstm1 = LSTM(layer_sizes["lstm1"], 
#                          return_sequences=True)        
#        self.lstm2 = LSTM(layer_sizes["lstm2"], 
#                          return_sequences=True)
#        self.output_layer = TimeDistributed(Dense(vocab_size, 
#                                             activation="softmax"))
#                
##        self.description = descriptions[str(self)]
#        self.V = vocab_size
#        self.params = {"vocab_size": vocab_size,
#                       "layer_sizes": layer_sizes}
#        
#                
#        
#        if compile_now:
#            print("COMPILING!")
#            self.compile_default()
#
#
#    def call(self, inputs):
#        embedded = self.embedding(inputs)
#        processed1 = self.lstm1(embedded)
#        processed2 = self.lstm2(processed1)
#        preds = self.output_layer(processed2)
#        self.outputs = preds
#        if preds is None:
#            raise ValueError("MY ERROR")
#        return preds
#    
#    
#    def compute_output_shape(self, input_shape):
#        shape = tf.TensorShape(input_shape).as_list()
#        shape += [self.num_classes]
#        return tf.TensorShape(shape)
#    
#            
#    def compile_default(self):
#        self.compile("adam",
#                     loss="categorical_crossentropy",
#                     metrics=["categorical_accuracy",
#                              self.entropy,
#                              self.relative_entropy])
#        
#        
#    def entropy(self, y_true, y_pred):
#        return -K.sum(y_pred*K.log(y_pred), axis=-1)
#    
#    
#    def relative_entropy(self, y_true, y_pred):
#        return self.entropy(y_true, y_pred)/K.log(K.cast_to_floatx(self.V))
#        
#        
#    def __repr__(self):
#        return "LanguageModel2"
#    
#    
#    def save_model_custom(self, dir_to_save):
#        self.save_weights(dir_to_save + "/LanguageModel2_weights")
#        with open(dir_to_save + "/LanguageModel2_parameters.json", "w") as handle:
#            json.dump(self.params, handle)
#            
#    @classmethod  
#    def from_saved_custom(cls, save_dir, compile_now=False):            
#        with open(save_dir + "/LanguageModel2_parameters.json", "r") as handle:
#            param_dict = json.load(handle)
#        
#        lm = cls(param_dict["vocab_size"],
#                        param_dict["layer_sizes"],
#                        compile_now=compile_now)
#        
#        
#        lm.load_weights(save_dir + "/LanguageModel2_weights")
#        
#        return lm
