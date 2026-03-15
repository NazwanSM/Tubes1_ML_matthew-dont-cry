import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, init_method='uniform', **kwargs):
        self.inputs = None
        
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            
        if init_method == 'zero':
            self.weights = np.zeros((n_inputs, n_neurons))
        elif init_method == 'uniform':
            lower = kwargs.get('lower_bound', -1.0)
            upper = kwargs.get('upper_bound', 1.0)
            self.weights = np.random.uniform(lower, upper, (n_inputs, n_neurons))
        elif init_method == 'normal':
            mean = kwargs.get('mean', 0.0)
            variance = kwargs.get('variance', 1.0)
            std_dev = np.sqrt(variance)
            self.weights = np.random.normal(mean, std_dev, (n_inputs, n_neurons))
        else:
            raise ValueError("Method harus 'zero', 'uniform', atau 'normal'")

        self.biases = np.zeros((1, n_neurons))
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        net = np.dot(inputs, self.weights) + self.biases
        return net

    def backward(self, d_net):
        # dE/dW = dE/d_net * d_net/dW
        self.dweights = np.dot(self.inputs.T, d_net)
        # Gradien untuk bias dE/db
        self.dbiases = np.sum(d_net, axis=0, keepdims=True)
        # Gradien terhadap input dE/dX
        self.dinputs = np.dot(d_net, self.weights.T)
        
        return self.dinputs