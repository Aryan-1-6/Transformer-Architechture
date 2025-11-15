import numpy as np 
from math import sqrt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from src.MPNeuronInfo import Layer_Dense,Activation_ReLU,Activation_Softmax


class Transformer:
    def __init__(self, corpus, n_heads, embd_dim=512):
        self.embd_dim = embd_dim
        self.tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
        self.vocab = {}
        self.input = []
        self.activation = Activation_Softmax()
        self.learning_rate = 0.01
        self.lookahead = []
        self.n_heads = n_heads
    
    def one_hot(self, Y, max):
        # k = self.indx_map(Y)
        one_hot_Y = []
        for sentence in Y:
            temp = np.zeros((sentence.size, max + 1))
            temp[np.arange(sentence.size), Y] = 1
            one_hot_Y.append(temp)
        return one_hot_Y  

    def vocab_creation(self):
        self.vocab['sos'] = 0
        indx = 1

        for sentence in self.tokenized_corpus:
            for i,word in enumerate(sentence):
                if word not in self.vocab :
                    self.vocab[word] = indx
                else:
                    continue
                indx += 1
        
    def word_embeddings(self):
        data = []
        for i in self.tokenized_corpus:
            temp = ['sos']

            for j in i :
                temp.append(j.lower())

            data.append(temp)

        model = Word2Vec(sentences=data, vector_size=512, window=5, min_count=1, workers=4)
        words = list(model.wv.index_to_key)
        self.words = words
        embeddings_matrix = np.zeros((len(words), model.vector_size))

        for i, word in enumerate(words):
            embeddings_matrix[i] = model.wv[word]

        self.embeddings = embeddings_matrix * sqrt(self.embd_dim)
        
    def positional_encoding(self, n):
        self.num_words = n
        self.position_encodings = np.zeros((self.num_words, self.embd_dim))
        for pos in range(self.num_words):
            for i in range(0, self.embd_dim, 2):
                angle = pos / np.power(10000, (2 * i) / np.float32(self.embd_dim))
                self.position_encodings[pos, i] = np.sin(angle)
                self.position_encodings[pos, i + 1] = np.cos(angle)
        
    
    def layer_normalization(self, x, epsilon=1e-6):
        gamma = np.ones(self.embd_dim)
        beta = np.zeros(self.embd_dim)

        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + epsilon)

        output = gamma * x_normalized + beta
        return output

    def feed_forward(self):
        self.ffn1 = Layer_Dense(self.embd_dim, self.embd_dim * 4, 10)
        self.ffnact = Activation_ReLU()
        self.ffn2 = Layer_Dense(self.embd_dim * 4, self.embd_dim, 9)

    def query_key_value(self):
        self.qlayer = Layer_Dense(self.embd_dim, self.embd_dim,8)
        self.klayer = Layer_Dense(self.embd_dim, self.embd_dim,7)
        self.vlayer = Layer_Dense(self.embd_dim, self.embd_dim,6)

    def residual_connections(self, self_attention_vals, encodings):
        self.res = self_attention_vals + encodings

class Encoder(Transformer):   
    def self_attention(self,ewe,n):
        # dk = self.embd_dim // self.n_heads
        # heads = [[]]

        # for start in range(0, self.embeddings, dk):
        #     for i in range(len(ewe)):
        #         end = start + dk
        #         head = ewe[i][start:end]
        #         heads.append(head)

        self.qlayer.forward(ewe)
        self.klayer.forward(ewe)
        self.vlayer.forward(ewe)

        # self.activation.forward(np.dot(self.qlayer.output, self.klayer.output.T) / sqrt(self.embd_dim))
        self.activation.forward(np.matmul(self.qlayer.output, self.klayer.output.transpose(0, 2, 1) / sqrt(self.embd_dim)))
        attention_vals = np.matmul(self.activation.output, self.vlayer.output)
        # attention_vals = np.dot(self.activation.output, self.vlayer.output)
        self.residual_connections(attention_vals, ewe)

    def ed_key_value(self):
        self.ed_klayer = Layer_Dense(self.embd_dim, self.embd_dim,5)
        self.ed_vlayer = Layer_Dense(self.embd_dim, self.embd_dim,4)

    def map(self, input_tokens):
        self.input = []
        for i,sentence in enumerate(input_tokens):
            self.input.append([self.vocab['sos']])
            for token in sentence:
                if token in self.vocab:
                    self.input[i].append(self.vocab[token])
            self.input[i].append(self.vocab['eos'])

class Decoder(Transformer):
    
    def map(self, input_tokens):
        self.input = []
        for i,sentence in enumerate(input_tokens):
            self.input.append([self.vocab['sos']])
            for token in sentence:
                if token in self.vocab:
                    self.input[i].append(self.vocab[token])

    def label(self, input_tokens):
        self.label = []
        temp = []
        for sentence in input_tokens:
            # self.label.append([])
            for token in sentence:
                if token in self.vocab:
                    temp.append(self.vocab[token])    
            temp.append(self.vocab['eos'])
            self.label.append(temp)
            temp = []
        
    def casual_mask(self, seq_len):
        self.mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        self.mask = np.where(self.mask == 1, -np.inf, 0.0)
        
    def self_attention(self, dwe, m):                              
        self.qlayer.forward(dwe)
        self.klayer.forward(dwe)
        self.vlayer.forward(dwe)
        
        self.midvalue = (np.matmul(self.qlayer.output, self.klayer.output.transpose(0,2,1)) / sqrt(self.embd_dim))
        # self.midvalue = (np.dot(self.qlayer.output, self.klayer.output.T) / sqrt(self.embd_dim))

        self.midvalue += self.mask
        self.activation.forward(self.midvalue)
        attention_vals = np.matmul(self.activation.output, self.vlayer.output)

        self.residual_connections(attention_vals, dwe)

    def ed_attention(self, e):

        self.ed_qlayer.forward(self.res)
        e.ed_klayer.forward(e.res)
        e.ed_vlayer.forward(e.res)

        self.activation2 = Activation_Softmax()
        self.activation2.forward((np.matmul(self.ed_qlayer.output, e.ed_klayer.output.transpose(0,2,1)) / sqrt(self.embd_dim)))

        attention_vals = np.matmul(self.activation2.output, e.ed_vlayer.output)

        self.final_residuals(attention_vals, self.res)

    def ed_query(self):
        self.ed_qlayer = Layer_Dense(self.embd_dim, self.embd_dim ,3)
    
    def final_residuals(self, ed_residuals, d_embed):
        self.final = ed_residuals + d_embed
    
    def next_word(self):
        self.nxtlayer = Layer_Dense(self.embd_dim, self.embd_dim * 4, 0)
        self.nxtactivation = Activation_ReLU()
        self.nxtlayer2 = Layer_Dense(self.embd_dim * 4, self.embd_dim, 1)
        self.activation3 = Activation_Softmax()
        self.vocablayer = Layer_Dense(self.embd_dim, len(self.vocab), 2)
   