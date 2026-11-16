import numpy as np
import pandas as pd

# ============================================================
#  PART 1: Backpropagation on XOR Problem
# ============================================================
print("=" * 50)
print("  PART 1: Backpropagation on XOR Problem")
print("=" * 50)

# XOR input and expected output
X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
y = np.array(([0], [1], [1], [0]))

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(out):
    return out * (1 - out)

# Hyperparameters
epoch = 500
lr = 1

# Network architecture
inputlayer_neurons  = 2
hiddenlayer_neurons = 3
output_neurons      = 1

# Initialize weights and biases
wh   = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh   = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epoch):
    # Forward pass
    hinp1      = np.dot(X, wh)
    hinp       = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1    = np.dot(hlayer_act, wout)
    outinp     = outinp1 + bout
    output     = sigmoid(outinp)

    # Backward pass
    EO          = y - output
    outgrad     = derivatives_sigmoid(output)
    d_output    = EO * outgrad
    EH          = d_output.dot(wout.T)
    hiddengrad  = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Weight updates
    wout += hlayer_act.T.dot(d_output) * lr
    wh   += X.T.dot(d_hiddenlayer) * lr

print("\nInput:")
print(X)
print("\nActual Output:")
print(y)
print("\nPredicted Output (raw probabilities):")
print(output)

print("\nPredicted Output (thresholded at 0.5):")
for out in output:
    print(1 if out > 0.5 else 0)

# ==================== Accuracy Section (XOR) ====================
xor_preds = np.array([1 if out > 0.5 else 0 for out in output]).reshape(-1, 1)
xor_correct = np.sum(xor_preds == y)
xor_total   = len(y)
xor_accuracy = (xor_correct / xor_total) * 100

print("\n" + "=" * 45)
print("   ACCURACY SECTION — XOR Problem")
print("=" * 45)
print(f"Correct Predictions : {xor_correct} / {xor_total}")
print(f"Accuracy            : {xor_accuracy:.2f}%")
print("=" * 45)


# ============================================================
#  PART 2: Backpropagation on Iris Dataset
# ============================================================
print("\n" + "=" * 50)
print("  PART 2: Backpropagation on Iris Dataset")
print("=" * 50)

# Load and preprocess data
data    = pd.read_csv("Iris (1).csv")
X_iris  = data.drop(columns=["Id", "Species"]).values.astype(float)
species_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
y_iris  = data["Species"].map(species_map).values

# Normalize features
X_iris  = (X_iris - X_iris.mean(axis=0)) / X_iris.std(axis=0)
Y_iris  = np.eye(len(species_map))[y_iris]   # one-hot encode

print(f"\nDataset shape: {data.shape}")
print(f"Classes: {list(species_map.keys())}")

# Network architecture
np.random.seed(42)
input_dim, hidden_dim, output_dim = 4, 8, 3
learning_rate = 0.1
epochs = 5000

# Initialize weights (He initialization)
wh_iris   = np.random.randn(input_dim, hidden_dim)  * np.sqrt(2.0 / input_dim)
bh_iris   = np.zeros((1, hidden_dim))
wout_iris = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
bout_iris = np.zeros((1, output_dim))

# Activation functions
def sigmoid_iris(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x     = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training loop
print(f"\nTraining for {epochs} epochs...")
for _ in range(epochs):
    # Forward pass
    hidden_iris = sigmoid_iris(X_iris @ wh_iris + bh_iris)
    output_iris = softmax(hidden_iris @ wout_iris + bout_iris)

    # Backward pass
    d_output_iris  = (output_iris - Y_iris) / X_iris.shape[0]
    d_wout_iris    = hidden_iris.T @ d_output_iris
    d_bout_iris    = np.sum(d_output_iris, axis=0, keepdims=True)
    hidden_error_iris = d_output_iris @ wout_iris.T
    hidden_grad_iris  = hidden_iris * (1 - hidden_iris) * hidden_error_iris
    d_wh_iris      = X_iris.T @ hidden_grad_iris
    d_bh_iris      = np.sum(hidden_grad_iris, axis=0, keepdims=True)

    # Weight updates
    wout_iris -= learning_rate * d_wout_iris
    bout_iris -= learning_rate * d_bout_iris
    wh_iris   -= learning_rate * d_wh_iris
    bh_iris   -= learning_rate * d_bh_iris

# Final predictions
final_hidden_iris = sigmoid_iris(X_iris @ wh_iris + bh_iris)
final_output_iris = softmax(final_hidden_iris @ wout_iris + bout_iris)
predicted = np.argmax(final_output_iris, axis=1)

# ==================== Accuracy Section (Iris) ====================
correct_iris = np.sum(predicted == y_iris)
total_iris   = len(y_iris)
accuracy     = np.mean(predicted == y_iris) * 100

# Per-class accuracy
reverse_map = {v: k for k, v in species_map.items()}

print("\n" + "=" * 45)
print("   ACCURACY SECTION — Iris Dataset")
print("=" * 45)
print(f"Training Accuracy   : {accuracy:.2f}%")
print(f"Correct Predictions : {correct_iris} / {total_iris}")
print(f"Misclassified       : {total_iris - correct_iris} / {total_iris}")
print("-" * 45)
print("Per-Class Accuracy:")
for cls_idx, cls_name in reverse_map.items():
    mask     = y_iris == cls_idx
    cls_acc  = np.mean(predicted[mask] == y_iris[mask]) * 100
    cls_corr = np.sum(predicted[mask] == y_iris[mask])
    cls_total = np.sum(mask)
    print(f"  {cls_name:<20}: {cls_acc:.2f}%  ({cls_corr}/{cls_total})")
print("=" * 45)
