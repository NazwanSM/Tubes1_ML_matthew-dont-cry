import numpy as np

class Initialization:
    def zero(self, shape):
        return np.zeros(shape)

    def uniform(self, shape, lower_bound=-1.0, upper_bound=1.0, seed=None):
        rng = np.random.RandomState(seed)
        return rng.uniform(lower_bound, upper_bound, shape)

    def normal(self, shape, mean=0.0, variance=1.0, seed=None):
        rng = np.random.RandomState(seed)
        std_dev = np.sqrt(variance)
        return rng.normal(mean, std_dev, shape)
    
    # Bonus
    def xavier(self, shape, distribution='uniform', seed=None):
        rng = np.random.RandomState(seed)
        n_in, n_out = shape
        std_dev = np.sqrt(2.0 / (n_in + n_out))

        if distribution == 'normal':
            return rng.normal(0, std_dev, shape)
        elif distribution == 'uniform':
            limit = std_dev * np.sqrt(3)
            return rng.uniform(-limit, limit, shape)
        else:
            raise ValueError("Distribution harus 'uniform' atau 'normal'")

    def he(self, shape, distribution='normal', seed=None):
        rng = np.random.RandomState(seed)
        n_in, _ = shape
        std_dev = np.sqrt(2.0 / n_in)

        if distribution == 'normal':
            return rng.normal(0, std_dev, shape)
        elif distribution == 'uniform':
            limit = std_dev * np.sqrt(3)
            return rng.uniform(-limit, limit, shape)
        else:
            raise ValueError("Distribution harus 'uniform' atau 'normal'")
