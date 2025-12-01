# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 22:07:38 2025

@author: kommu
"""

import numpy as np

np.random.seed(42)

# Data
X = np.array(([2, 9],
              [1, 5],
              [3, 6]), dtype=float)
y = np.array(([92],
              [86],
              [89]), dtype=float)

# Normalise inputs & outputs
X = X / np.amax(X, axis=0)      # scale features column-wise
y = y / 100.0                   # scale target to [0,1]

# Activation and derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivatives_sigmoid(x):
    # x is expected to be sigmoid(x) (i.e. activated output)
    return x * (1.0 - x)

# Hyperparameters
epochs = 7000
lr = 0.1

# Network architecture
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Weight & bias initialization (small random values)
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epochs):
    # Forward propagation
    hinp = np.dot(X, wh) + bh                # (3,3)
    hlayer_act = sigmoid(hinp)               # (3,3)

    outinp = np.dot(hlayer_act, wout) + bout # (3,1)
    output = sigmoid(outinp)                 # (3,1)

    # Compute error
    error = y - output                       # (3,1)
    # Backpropagation
    out_grad = error * derivatives_sigmoid(output)              # (3,1)
    # Error term for hidden layer (propagate back)
    hidden_error = out_grad.dot(wout.T)                        # (3,3)
    hidden_grad = hidden_error * derivatives_sigmoid(hlayer_act)  # (3,3)

    # Update weights and biases
    wout += hlayer_act.T.dot(out_grad) * lr    # (3,1)
    bout += np.sum(out_grad, axis=0, keepdims=True) * lr  # (1,1)

    wh += X.T.dot(hidden_grad) * lr            # (2,3)
    bh += np.sum(hidden_grad, axis=0, keepdims=True) * lr  # (1,3)

    # Optional: print loss occasionally
    if (i % 1000) == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {i:5d} - MSE: {loss:.6f}")

# Final results
print("\nInput (scaled):\n", X)
print("Actual Output (scaled):\n", y)
print("Predicted Output (scaled):\n", output)
print("Predicted Output (original scale):\n", output * 100)
