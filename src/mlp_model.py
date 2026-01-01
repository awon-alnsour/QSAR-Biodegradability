"""
QSAR Biodegradability Prediction - Custom MLP Implementation
Author: Awon Alnsour
Course: Data Modelling and Machine Intelligence
University of Sheffield

This script defines a custom Multi-Layer Perceptron (MLP) used for QSAR 
biodegradability prediction. The MLP supports:

1. One hidden layer with ReLU activation
2. Sigmoid output for binary classification
3. Dropout regularization to prevent overfitting
4. Mini-batch gradient descent training with Adam optimizer
5. Binary cross-entropy loss function
6. Prediction and probability output compatible with scikit-learn interface

Dependencies:
- numpy
"""

import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    """
    Sigmoid activation function.
    Maps any real value to (0,1), useful for probability outputs.
    """
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    output = np.empty_like(x)
    output[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    output[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
    return output

def sigmoid_derivative(sigmoid_output):
    """
    Derivative of the sigmoid function.
    Used during backpropagation.
    """
    return sigmoid_output * (1 - sigmoid_output)

def relu(x):
    """
    ReLU activation function.
    Returns max(0, x) to introduce non-linearity and avoid vanishing gradients.
    """
    return np.maximum(0, x)

def relu_derivative(pre_activation):
    """
    Derivative of ReLU function.
    Used during backpropagation to compute gradients for hidden layer weights.
    """
    return (pre_activation > 0).astype(float)

# Loss function
def binary_cross_entropy(true_labels, predicted_probs, eps=1e-12, sample_weights=None):
    """
    Binary cross-entropy loss.
    Measures the distance between true labels and predicted probabilities.
    Supports optional sample weights.
    """
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
    loss_values = -(true_labels * np.log(predicted_probs) +
                    (1 - true_labels) * np.log(1 - predicted_probs))
    if sample_weights is None:
        return np.mean(loss_values)
    else:
        return np.sum(loss_values * sample_weights) / np.sum(sample_weights)

# Multi-Layer Perceptron class
class SimpleMLP:
    """
    Two-layer MLP for binary classification.

    Features:
    - One hidden layer with ReLU activation
    - Sigmoid output layer
    - Dropout regularization
    - Mini-batch training with Adam optimizer
    - Supports weighted loss and early stopping
    """

    def __init__(self, input_size, hidden_size=64, learning_rate=0.001, l2_reg=0.0,
                 batch_size=32, dropout_prob=0.0, seed=42):
        """
        Initialize weights, biases, hyperparameters, and Adam optimizer state.
        """
        rng = np.random.RandomState(seed)

        # Initialize hidden layer weights and bias
        self.hidden_weights = rng.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.hidden_bias = np.zeros(hidden_size)

        # Initialize output layer weights and bias
        self.output_weights = rng.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.output_bias = np.zeros(1)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob

        # Adam optimizer state
        self.moment1 = {k: 0 for k in ["hidden_weights", "hidden_bias",
                                       "output_weights", "output_bias"]}
        self.moment2 = {k: 0 for k in ["hidden_weights", "hidden_bias",
                                       "output_weights", "output_bias"]}
        self.adam_step = 0

    # Dropout
    def _apply_dropout(self, activations):
        """
        Apply dropout to hidden layer activations.
        Randomly zeros out activations with probability dropout_prob.
        Scales remaining activations by 1 / (1 - dropout_prob).
        """
        if self.dropout_prob <= 0:
            return activations, None
        mask = (np.random.rand(*activations.shape) > self.dropout_prob).astype(float)
        return activations * mask / (1 - self.dropout_prob), mask

    # Adam optimizer
    def _adam_update(self, param_name, gradient):
        """
        Perform Adam optimizer update on a parameter.
        Includes L2 regularization if specified.
        """
        self.adam_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        gradient += self.l2_reg * getattr(self, param_name)
        self.moment1[param_name] = beta1 * self.moment1[param_name] + (1 - beta1) * gradient
        self.moment2[param_name] = beta2 * self.moment2[param_name] + (1 - beta2) * (gradient ** 2)
        m_hat = self.moment1[param_name] / (1 - beta1 ** self.adam_step)
        v_hat = self.moment2[param_name] / (1 - beta2 ** self.adam_step)
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    # Forward pass
    def forward(self, inputs, training=True):
        """
        Forward propagation through the network.
        Returns output activations and cache for backpropagation.
        """
        hidden_pre = inputs.dot(self.hidden_weights) + self.hidden_bias
        hidden_activation = relu(hidden_pre)

        if training:
            hidden_activation, dropout_mask = self._apply_dropout(hidden_activation)
        else:
            dropout_mask = None

        output_pre = hidden_activation.dot(self.output_weights) + self.output_bias
        output_activation = sigmoid(output_pre).reshape(-1)

        cache = {
            "inputs": inputs,
            "hidden_pre": hidden_pre,
            "hidden_activation": hidden_activation,
            "dropout_mask": dropout_mask,
            "output_pre": output_pre,
            "output_activation": output_activation
        }
        return output_activation, cache

    # Prediction
    def predict_proba(self, inputs):
        """
        Predict probabilities for input samples.
        """
        output, _ = self.forward(inputs, training=False)
        return output

    def predict(self, inputs, threshold=0.5):
        """
        Predict class labels for input samples using a threshold.
        """
        return (self.predict_proba(inputs) >= threshold).astype(int)

    # Mini-batch generator
    def _generate_batches(self, n_samples):
        """
        Yield shuffled mini-batches of indices for training.
        """
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            yield indices[start:start + self.batch_size]

    # Training
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100,
              class_weights=None, verbose=True, early_stop_rounds=10):
        """
        Train the MLP using mini-batch gradient descent with Adam optimizer.
        Supports class weighting and early stopping on validation loss.
        """
        n_samples = X_train.shape[0]

        # Compute sample weights if provided
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y_train])
        else:
            sample_weights = None

        best_val_loss = np.inf
        rounds_without_improvement = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            for batch_idx in self._generate_batches(n_samples):
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                weights_batch = sample_weights[batch_idx] if sample_weights is not None else None

                # Forward pass
                predictions, cache = self.forward(X_batch, training=True)

                # Gradient of loss w.r.t output
                if weights_batch is None:
                    grad_output = (-(y_batch / (predictions + 1e-12)) +
                                   (1 - y_batch) / (1 - predictions + 1e-12)) / len(y_batch)
                else:
                    raw_grad = (-(y_batch / (predictions + 1e-12)) +
                                (1 - y_batch) / (1 - predictions + 1e-12))
                    grad_output = raw_grad * weights_batch / np.sum(weights_batch)

                # Backpropagation
                dz_output = grad_output * sigmoid_derivative(predictions)
                hidden_activation = cache["hidden_activation"]
                inputs_batch = cache["inputs"]
                grad_output_weights = hidden_activation.T.dot(dz_output.reshape(-1, 1))
                grad_output_bias = dz_output.sum()
                da_hidden = dz_output.reshape(-1, 1).dot(self.output_weights.T)
                dz_hidden = da_hidden * relu_derivative(cache["hidden_pre"])
                if cache["dropout_mask"] is not None:
                    dz_hidden *= cache["dropout_mask"] / (1 - self.dropout_prob)
                grad_hidden_weights = inputs_batch.T.dot(dz_hidden)
                grad_hidden_bias = dz_hidden.sum(axis=0)

                # Update parameters
                self.hidden_weights -= self._adam_update("hidden_weights", grad_hidden_weights)
                self.hidden_bias -= self._adam_update("hidden_bias", grad_hidden_bias)
                self.output_weights -= self._adam_update("output_weights", grad_output_weights)
                self.output_bias -= self._adam_update("output_bias", grad_output_bias)

            # Monitoring and early stopping
            if verbose:
                train_loss = binary_cross_entropy(y_train, self.predict_proba(X_train))
                train_acc = (self.predict(X_train) == y_train).mean()
                msg = f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}"

                if X_val is not None:
                    val_pred = self.predict_proba(X_val)
                    val_loss = binary_cross_entropy(y_val, val_pred)
                    val_acc = (self.predict(X_val) == y_val).mean()
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"

                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        rounds_without_improvement = 0
                        best_weights = (self.hidden_weights.copy(),
                                        self.hidden_bias.copy(),
                                        self.output_weights.copy(),
                                        self.output_bias.copy())
                    else:
                        rounds_without_improvement += 1
                        if rounds_without_improvement >= early_stop_rounds:
                            print(msg + " --> early stopping")
                            self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias = best_weights
                            return self

                print(msg)

        return self
