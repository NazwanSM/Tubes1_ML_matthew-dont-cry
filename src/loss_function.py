import numpy as np

# Kelas fungsi loss -- Mean Squared Error, Binary Cross-Entropy, Categorical Cross-Entropy
class LossFunctions:
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # buat menghindari log(0)
        
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))

    def categorical_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # buat menghindari log(0)
        
        return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))