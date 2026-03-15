import numpy as np

class LossFunctions:
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def mean_squared_error_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # buat menghindari log(0)
        
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))

    def binary_cross_entropy_derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

    def categorical_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # buat menghindari log(0)
        
        return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))

    def categorical_cross_entropy_derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred / y_true.shape[0]