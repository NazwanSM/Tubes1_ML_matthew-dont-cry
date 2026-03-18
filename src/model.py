import pickle
import numpy as np
import matplotlib.pyplot as plt

from layer import DenseLayer
from activation_function import ActivationFunctions
from loss_function import LossFunctions
from initialization import Initialization
from rmsnorm import RMSNorm

class FFNN:
    def __init__(self, layer_sizes, activations, loss, init_method='he', init_params=None, normalization=None):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Jumlah aktivasi ({len(activations)}) harus sama dengan "
                f"jumlah layer transisi ({len(layer_sizes) - 1})"
            )

        self.layer_sizes = layer_sizes
        self.activation_names = activations
        self.loss_name = loss

        # Inisialisasi 
        self._act = ActivationFunctions()
        self._loss_obj = LossFunctions()
        self._init = Initialization()

        self._act_fns = []
        self._act_derivs = []
        for name in activations:
            self._act_fns.append(getattr(self._act, name))
            self._act_derivs.append(getattr(self._act, f"{name}_derivative"))

        self._loss_fn = getattr(self._loss_obj, loss)
        self._loss_deriv = getattr(self._loss_obj, f"{loss}_derivative")

        self._use_combined_softmax_cce = (
            loss == 'categorical_cross_entropy' 
            and len(activations) > 0 
            and activations[-1] == 'softmax'
        )
        
        self.layers = []
        self.norms = []
        if init_params is None:
            init_params = {}

        for i in range(len(layer_sizes) - 1):
            method = init_method if isinstance(init_method, str) else init_method[i]
            params = init_params if isinstance(init_params, dict) else init_params[i]

            # Buat DenseLayer 
            layer = DenseLayer(layer_sizes[i], layer_sizes[i + 1], init_method='zero')

            weight_shape = (layer_sizes[i], layer_sizes[i + 1])
            bias_shape = (1, layer_sizes[i + 1])

            init_fn = getattr(self._init, method)
            layer.weights = init_fn(weight_shape, **params)

            self.layers.append(layer)

            # RMSNorm per layer
            if normalization is not None:
                norm_flag = normalization if isinstance(normalization, bool) else (
                    normalization[i] if i < len(normalization) else False
                )
                if norm_flag:
                    self.norms.append(RMSNorm(layer_sizes[i + 1]))
                else:
                    self.norms.append(None)
            else:
                self.norms.append(None)

        self._net_cache = []   
        self._act_cache = []   
        self._norm_cache = []

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

        self._optimizer_step = 0
        self._adam_m_w = []
        self._adam_v_w = []
        self._adam_m_b = []
        self._adam_v_b = []
        self._adam_m_g = []
        self._adam_v_g = []

    def _progress_bar(current, total, width=28):
        if total <= 0:
            total = 1
        ratio = current / total
        filled = int(width * ratio)
        return "[" + "=" * filled + "." * (width - filled) + "]"

    def _regularization_loss(self, l1_lambda=0.0, l2_lambda=0.0):
        reg_loss = 0.0
        if l1_lambda > 0:
            reg_loss += l1_lambda * sum(np.sum(np.abs(layer.weights)) for layer in self.layers)
        if l2_lambda > 0:
            reg_loss += 0.5 * l2_lambda * sum(np.sum(layer.weights ** 2) for layer in self.layers)
        return reg_loss

    def forward(self, x):
        self._net_cache = []
        self._act_cache = [x]
        self._norm_cache = []

        output = x
        for i, layer in enumerate(self.layers):
            net = layer.forward(output)

            # RMSNorm (antara linear dan aktivasi)
            if self.norms[i] is not None:
                net = self.norms[i].forward(net)
            self._norm_cache.append(net)

            self._net_cache.append(net)
            output = self._act_fns[i](net)
            self._act_cache.append(output)

        return output

    def backward(self, y_true, y_pred):
        n_layers = len(self.layers)

        # gradien awal dari loss function
        if self._use_combined_softmax_cce:
            grad = (y_pred - y_true) / y_true.shape[0]
            # Layer terakhir tanpa activation derivative 
            if self.norms[-1] is not None:
                grad = self.norms[-1].backward(grad)
            grad = self.layers[-1].backward(grad)

            for i in reversed(range(n_layers - 1)):
                act_deriv = self._act_derivs[i](self._net_cache[i])
                grad = grad * act_deriv
                if self.norms[i] is not None:
                    grad = self.norms[i].backward(grad)
                grad = self.layers[i].backward(grad)
        else:
            grad = self._loss_deriv(y_true, y_pred)

            for i in reversed(range(n_layers)):
                if self.activation_names[i] == 'softmax':
                    # Vector-Jacobian product for softmax:
                    # J^T v = s * (v - sum(v*s)).
                    s = self._act_cache[i + 1]
                    grad = s * (grad - np.sum(grad * s, axis=1, keepdims=True))
                else:
                    act_deriv = self._act_derivs[i](self._net_cache[i])
                    grad = grad * act_deriv
                if self.norms[i] is not None:
                    grad = self.norms[i].backward(grad)
                grad = self.layers[i].backward(grad)

    def _initialize_adam_state(self):
        self._optimizer_step = 0
        self._adam_m_w = [np.zeros_like(layer.weights) for layer in self.layers]
        self._adam_v_w = [np.zeros_like(layer.weights) for layer in self.layers]
        self._adam_m_b = [np.zeros_like(layer.biases) for layer in self.layers]
        self._adam_v_b = [np.zeros_like(layer.biases) for layer in self.layers]

        self._adam_m_g = []
        self._adam_v_g = []
        for norm in self.norms:
            if norm is None:
                self._adam_m_g.append(None)
                self._adam_v_g.append(None)
            else:
                self._adam_m_g.append(np.zeros_like(norm.gamma))
                self._adam_v_g.append(np.zeros_like(norm.gamma))

    def update_weights(
        self,
        learning_rate,
        l1_lambda=0.0,
        l2_lambda=0.0,
        optimizer='adam',
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        optimizer = optimizer.lower()
        if optimizer == 'default':
            optimizer = 'adam'

        if optimizer == 'adam':
            if len(self._adam_m_w) != len(self.layers):
                self._initialize_adam_state()

            self._optimizer_step += 1
            t = self._optimizer_step

            for i, layer in enumerate(self.layers):
                grad_w = layer.dweights.copy()
                grad_b = layer.dbiases.copy()

                # gradien regularisasi ke bobot
                if l1_lambda > 0:
                    grad_w += l1_lambda * np.sign(layer.weights)
                if l2_lambda > 0:
                    grad_w += l2_lambda * layer.weights

                # update untuk weights
                self._adam_m_w[i] = beta1 * self._adam_m_w[i] + (1 - beta1) * grad_w
                self._adam_v_w[i] = beta2 * self._adam_v_w[i] + (1 - beta2) * (grad_w ** 2)
                m_hat_w = self._adam_m_w[i] / (1 - beta1 ** t)
                v_hat_w = self._adam_v_w[i] / (1 - beta2 ** t)
                layer.weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

                # update untuk biases
                self._adam_m_b[i] = beta1 * self._adam_m_b[i] + (1 - beta1) * grad_b
                self._adam_v_b[i] = beta2 * self._adam_v_b[i] + (1 - beta2) * (grad_b ** 2)
                m_hat_b = self._adam_m_b[i] / (1 - beta1 ** t)
                v_hat_b = self._adam_v_b[i] / (1 - beta2 ** t)
                layer.biases -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

                # update untuk RMSNorm gamma (jika ada)
                if self.norms[i] is not None:
                    grad_g = self.norms[i].dgamma.copy()
                    self._adam_m_g[i] = beta1 * self._adam_m_g[i] + (1 - beta1) * grad_g
                    self._adam_v_g[i] = beta2 * self._adam_v_g[i] + (1 - beta2) * (grad_g ** 2)
                    m_hat_g = self._adam_m_g[i] / (1 - beta1 ** t)
                    v_hat_g = self._adam_v_g[i] / (1 - beta2 ** t)
                    self.norms[i].gamma -= learning_rate * m_hat_g / (np.sqrt(v_hat_g) + epsilon)

            return

        if optimizer == 'sgd':
            for i, layer in enumerate(self.layers):
                grad_w = layer.dweights.copy()

                # add gradien regularisasi ke bobot
                if l1_lambda > 0:
                    grad_w += l1_lambda * np.sign(layer.weights)
                if l2_lambda > 0:
                    grad_w += l2_lambda * layer.weights

                layer.weights -= learning_rate * grad_w
                layer.biases -= learning_rate * layer.dbiases

                # Update RMSNorm gamma
                if self.norms[i] is not None:
                    self.norms[i].update(learning_rate)
            return

        raise ValueError("optimizer harus 'sgd' atau 'adam'")

    def train(self, x_train, y_train, x_val=None, y_val=None,
              batch_size=32, learning_rate=0.01, epochs=10,
              verbose=1, l1_lambda=0.0, l2_lambda=0.0,
              optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1e-8):
        optimizer = optimizer.lower()
        if optimizer == 'default':
            optimizer = 'adam'
        if optimizer not in ('sgd', 'adam'):
            raise ValueError("optimizer harus 'default', 'sgd', atau 'adam'")

        if optimizer == 'adam':
            # m_t, v_t, bias correction m_hat dan v_hat.
            self._initialize_adam_state()

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

        n_samples = x_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0

            for batch_idx, start in enumerate(range(0, n_samples, batch_size), start=1):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(x_batch)
                batch_loss = self._loss_fn(y_batch, y_pred)
                epoch_loss += batch_loss * len(x_batch)


                self.backward(y_batch, y_pred)
                self.update_weights(
                    learning_rate,
                    l1_lambda,
                    l2_lambda,
                    optimizer=optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon
                )

                if verbose == 1:
                    running_loss = epoch_loss / min(end, n_samples)
                    bar = self._progress_bar(batch_idx, n_batches)
                    print(
                        f"\rEpoch {epoch + 1}/{epochs} {bar} {batch_idx}/{n_batches} "
                        f"train_loss: {running_loss:.4f}",
                        end=""
                    )

            data_loss = epoch_loss / n_samples
            reg_loss = self._regularization_loss(l1_lambda, l2_lambda) / n_samples
            total_train_loss = data_loss + reg_loss
            self.history['train_loss'].append(total_train_loss)

            val_msg = ""
            if x_val is not None and y_val is not None:
                y_val_pred = self.forward(x_val)
                val_loss = self._loss_fn(y_val, y_val_pred) + reg_loss
                self.history['val_loss'].append(val_loss)
                val_msg = f" - val_loss: {val_loss:.4f}"

            if verbose == 1:
                print(
                    f"\rEpoch {epoch + 1}/{epochs} "
                    f"{self._progress_bar(n_batches, n_batches)} {n_batches}/{n_batches} "
                    f"train_loss: {total_train_loss:.4f}{val_msg}"
                )

        return self.history

    def predict(self, x):
        return self.forward(x)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss', linewidth=2)
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_weight_distribution(self, layer_indices=None):
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(layer_indices):
            if idx < 0 or idx >= len(self.layers):
                print(f"Peringatan: Layer index {idx} di luar jangkauan, dilewati.")
                continue

            weights = self.layers[idx].weights.flatten()
            axes[i].hist(weights, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f"Layer {idx + 1} ({self.layer_sizes[idx]}→{self.layer_sizes[idx+1]}) — Distribusi Bobot")
            axes[i].set_xlabel("Nilai Bobot")
            axes[i].set_ylabel("Frekuensi")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices=None):
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(layer_indices):
            if idx < 0 or idx >= len(self.layers):
                print(f"Peringatan: Layer index {idx} di luar jangkauan, dilewati.")
                continue

            gradients = self.layers[idx].dweights.flatten()
            axes[i].hist(gradients, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f"Layer {idx + 1} ({self.layer_sizes[idx]}→{self.layer_sizes[idx+1]}) — Distribusi Gradien")
            axes[i].set_xlabel("Nilai Gradien")
            axes[i].set_ylabel("Frekuensi")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()