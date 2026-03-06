import numpy as np

# Kelas fungsi aktivasi -- Linear, ReLU, Sigmoid, Tanh, Softmax, ELU, Swish
class ActivationFunctions:
    def linear(self, x):
        return x

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    # Bonus: fungsi aktivasi tambahan
    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha*(np.exp(x) - 1))

    # Bonus: fungsi aktivasi tambahan
    def swish(self, x):
        return x*self.sigmoid(x)