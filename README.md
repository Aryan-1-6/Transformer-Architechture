# 🚀 Transformer Architecture

This repository contains a modular implementation of a **Transformer model built entirely from scratch using NumPy**, without using PyTorch or TensorFlow.  
It also includes training notebooks, Word2Vec-based embeddings, and utilities for low-level neuron analysis and debugging.

## 🔶 Components Overview

### 1. `src/transformer.py`
Implements the full Transformer architecture based on the *Attention Is All You Need* paper.

#### ✔ Core Modules
- **Self Attention**
- **Scaled Dot-Product Attention**
- **Feed-Forward Networks (FFN)**
- **Residual Connections + LayerNorm**
- **Positional Encoding**
- **Encoder Layer**
- **Decoder Layer**
- **Masked (causal) attention for decoding**
- **Cross-attention** between encoder → decoder

Supports:
- Batching  
- Sequence-level attention  
- Word2Vec embeddings as token vectors  

---

### 2. `src/MPNeuronInfo.py`
Contains fundamental neural components implemented from scratch:

#### ✔ Layers
- `Layer_Dense`
- `Activation_ReLU`
- `Activation_Softmax`

#### ✔ Loss Function
- `Loss_CrossCategoricalEntropy`

#### ✔ Optimizer
- `OptimizerAdam` (with momentum, RMS, and bias correction)

These mimic deep learning library internals but are written manually for transparency.

---

### 3. Tokenization & Embeddings
The project uses:

- `nltk.word_tokenize` for tokenization  
- `gensim.Word2Vec` for dense vector embeddings  

Workflow:
1. Tokenize English/Spanish sentences  
2. Convert tokens → vectors via Word2Vec  
3. Pass sequence embeddings → Transformer  

---

### 4. Training Notebook (`notebooks/transformer_training.ipynb`)

Shows complete flow:

#### ✔ Data Preprocessing
- Tokenization  
- Vocabulary mapping  
- Embedding lookup  
- Padding & batching  

#### ✔ Training Loop
- Forward pass  
- Loss computation  
- Backpropagation  
- Parameter updates (Adam)  
- Logging loss curves  

#### ✔ Inference Logic
- Start with `<SOS>` token  
- Autoregressive decoding  
- Add positional encodings each step  
- Use encoder output for all decoding steps  

---

## 🛠 Setup

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

