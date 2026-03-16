import numpy as np

# Kelas fungsi aktivasi -- Linear, ReLU, Sigmoid, Tanh, Softmax, ELU, Swish
class ActivationFunctions:
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        t = self.tanh(x)
        return 1 - t**2

    def softmax(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        # return per- sample Jacobian matrices with shape (batch, n_class, n_class).
        if s.ndim == 1:
            s = s.reshape(1, -1)
        batch_size, n_class = s.shape
        jacobian = np.zeros((batch_size, n_class, n_class))
        for i in range(batch_size):
            si = s[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(si) - np.dot(si, si.T)
        return jacobian

    # Bonus: fungsi aktivasi tambahan
    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha*(np.exp(x) - 1))

    def elu_derivative(self, x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

    # Bonus: fungsi aktivasi tambahan
    def swish(self, x):
        return x*self.sigmoid(x)

    def swish_derivative(self, x):
        s = self.sigmoid(x)
        return s + x * s * (1 - s)