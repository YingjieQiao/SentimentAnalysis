import os
import sys
from datetime import datetime

import numpy as np
from layer import RNNLayer
from output import Softmax


class Model:
    def __init__(self, word_dim, output_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.output_dim = output_dim
        self.U = np.random.uniform(
            -np.sqrt(1 / word_dim), np.sqrt(1 / word_dim), (hidden_dim, word_dim)
        )
        self.W = np.random.uniform(
            -np.sqrt(1 / hidden_dim), np.sqrt(1 / hidden_dim), (hidden_dim, hidden_dim)
        )
        self.V = np.random.uniform(
            -np.sqrt(1 / hidden_dim), np.sqrt(1 / hidden_dim), (output_dim, hidden_dim)
        )

    """
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    """

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(path + "U.npy", self.U)
        np.save(path + "W.npy", self.W)
        np.save(path + "V.npy", self.V)

    def load(self, path):
        self.U = np.load(path + "U.npy")
        self.W = np.load(path + "W.npy")
        self.V = np.load(path + "V.npy")

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.s
            layers.append(layer)
        return layers

    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def calculate_loss(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            dmulv = output.diff(layers[t].mulv, y[t])
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(
                input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv
            )
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.output_dim)
            for i in range(t - 1, max(-1, t - self.bptt_truncate - 1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i - 1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(
                    input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv
                )
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU, dW, dV)

    def sgd_step(self, x, y, learning_rate):

        dU, dW, dV = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
