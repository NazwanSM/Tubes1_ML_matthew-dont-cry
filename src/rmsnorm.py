import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.dim = dim
        self.gamma = np.ones((1, dim))
        self.dgamma = np.zeros_like(self.gamma)
        self.cache_input = None
        self.cache_rms = None

    def forward(self, x):
        self.cache_input = x
        # RMS = sqrt(mean(x^2) + eps)
        self.cache_rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_hat = x / self.cache_rms
        return self.gamma * x_hat

    def backward(self, grad_output):
        x = self.cache_input
        rms = self.cache_rms
        n = x.shape[-1]

        # Gradien terhadap x_hat
        dx_hat = grad_output * self.gamma

        # Gradien terhadap rms
        drms = -np.sum(dx_hat * x / (rms ** 2), axis=-1, keepdims=True)

        # Gradien terhadap x
        dx = dx_hat / rms + drms * x / (n * rms)

        # Gradien terhadap gamma
        self.dgamma = np.sum(grad_output * (x / rms), axis=0, keepdims=True)

        return dx

    def update(self, learning_rate):
        self.gamma -= learning_rate * self.dgamma
