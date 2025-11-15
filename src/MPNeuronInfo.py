import sys
import os
import json
import datetime
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

    
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, seedval=0):
        np.random.seed(seedval)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.matmul(self.inputs.transpose(0,2,1), dvalues)
        self.dbiases = np.sum(dvalues, axis=1, keepdims=True)

        self.dweights = np.mean(self.dweights, axis=0)
        self.dbiases = np.mean(self.dbiases, axis=0)

        self.dinputs = np.dot(dvalues, self.weights.T)

class Loss_CrossCategoricalEntropy():
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true):
        # Clip values to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_pred.shape) == 3:
            batch_size, seq_len, vocab_size = y_pred.shape

            if len(y_true.shape) == 2:
                correct_confidences = y_pred_clipped[
                    np.arange(batch_size)[:, None],  
                    np.arange(seq_len)[None, :],    
                    y_true                           
                ]

            elif len(y_true.shape) == 3:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=-1)

        elif len(y_pred.shape) == 2:
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[np.arange(len(y_pred)), y_true]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        else:
            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

        # Compute negative log likelihoods
        negative_log_likelihoods = -np.log(correct_confidences + 1e-10)

        return negative_log_likelihoods
    
    def backward(self, y_pred, y):
        n = len(y_pred)
        self.dinputs = (y_pred - y)/n
        return self.dinputs
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_Leaky_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0.2 * inputs, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0.2
        

class Activation_Softmax:
    def forward(self, inputs):
        # Numerically stable softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / (np.sum(exp_values, axis=-1, keepdims=True) + 1e-10)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        # iterate over batch and sequence
        for b in range(dvalues.shape[0]):          # batch dimension
            for t in range(dvalues.shape[1]):      # time/sequence dimension
                single_output = self.output[b, t].reshape(-1, 1)
                single_dvalues = dvalues[b, t].reshape(-1, 1)
                
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[b, t] = np.dot(jacobian_matrix, single_dvalues).reshape(-1)


    # def backward(self, dvalues):
    #     # # Vectorized derivative for batch or sequence data
    #     # # shape: (batch, seq_len, vocab_size)
    #     # self.dinputs = self.output * (dvalues - np.sum(dvalues * self.output, axis=-1, keepdims=True))
    #     self.dinputs = np.empty_like(dvalues)

    #     for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
    #         single_output = single_output.reshape(-1, 1)
    #         jacobian_matrix = np.diagflat(single_output) - np.matmul(single_output, single_output.T)
    #         self.dinputs[index] = np.matmul(jacobian_matrix, single_dvalues)
class Optimizer_SGD:

    def __init__(self, learning_rate=0.00001):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


class OptimizerAdam:

    def __init__(self, learning_rate=0.0005, beta1=0.9, beta2=0.98, epsilon=1e-9):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.beta1_t = 1  
        self.beta2_t = 1  
        self.flag = 0
        
    def update_params(self, layer):
        # Update weights

        if self.flag == 0:
            self.m_w = np.zeros_like(layer.weights)
            self.v_w = np.zeros_like(layer.weights)
            self.m_b = np.zeros_like(layer.biases)
            self.v_b = np.zeros_like(layer.biases)
            self.flag == 1
            
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * layer.dweights
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (layer.dweights ** 2)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * layer.dbiases
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (layer.dbiases ** 2)

        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

        m_w_hat = self.m_w / (1 - self.beta1_t)
        v_w_hat = self.v_w / (1 - self.beta2_t)
        m_b_hat = self.m_b / (1 - self.beta1_t)
        v_b_hat = self.v_b / (1 - self.beta2_t)

        layer.weights += -self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases += -self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)