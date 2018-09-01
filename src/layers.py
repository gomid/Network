import numpy as np


class ReLU:
    def __init__(self):
        self.gI = np.array([]).astype(np.float32)
        self.input = np.array([]).astype(np.float32)
        self.output = np.array([]).astype(np.float32)

    def forward(self, X):
        self.input = X
        self.output = X.clip(0)
        return self.output

    def backward(self, gradient):
        self.gI = gradient * (self.input > 0).astype(np.float32)
        return self.gI


class FullyConnectedLayer:
    def __init__(self, num_input, num_output):
        # layer params
        self.W = np.random.randn(num_input, num_output) * np.sqrt(2 / num_input)
        self.b = np.random.randn(num_output) * np.sqrt(2 / num_input)

        self.gW = np.zeros(self.W.shape).astype(np.float32)
        self.gb = np.zeros(self.b.shape).astype(np.float32)
        self.gI = np.array([]).astype(np.float32)

        # input data and output data
        self.input = np.array([]).astype(np.float32)
        self.output = np.array([]).astype(np.float32)

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, gradient_data):
        self.gW = np.dot(np.transpose(self.input), gradient_data)
        self.gb = np.sum(gradient_data, axis=0)
        self.gI = np.dot(gradient_data, np.transpose(self.W))
        return self.gI


class SoftmaxLayerWithCrossEntropyLoss:
    def __init__(self):
        self.gI = np.array([]).astype(np.float32)
        self.input = np.array([]).astype(np.float32)
        self.output = np.array([]).astype(np.float32)

    def backward(self):
        return self.gI

    def eval(self, X, y):
        self.input = X
        self.output = softmax(X)
        self.gI = delta_cross_entropy(X, y)
        loss, acc = cross_entropy(X, y), np.sum(np.argmax(X, axis=1) == y, axis=0) / y.shape[0]
        return loss, acc


def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(X, y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad/m
    return grad
