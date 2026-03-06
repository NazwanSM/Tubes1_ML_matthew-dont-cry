import numpy as np

class Activation:
    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement this method")
    def derivative(self, x):
        raise NotImplementedError("Subclasses must implement this method")
    
class Linear(Activation):
    def __call__(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)
    
class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return (x > 0).astype(float)
    
class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        s = self.__call__(x)
        return s * (1 - s)
    
class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def derivative(self, x):
        t = self.__call__(x)
        return 1 - t**2
    
class Softmax(Activation):
    def __call__(self, x):
        exz = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exz / np.sum(exz, axis=-1, keepdims=True)
    def derivative(self, x):
        s = self.__call__(x)
        # masih blom bner keknya
        return s * (1 - s)

# BONUS
class Sign(Activation):
    def __call__(self, x):
        return np.where(x >= 0, 1, -1)
    def derivative(self, x):
        return np.zeros_like(x)
    
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)