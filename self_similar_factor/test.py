from structure import *
import pickle
import numpy as np
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

argErrCode = 3
if len(sys.argv) != 2:
    print("Wrong number of arguments")
    print("Example: python test.py ./exact_func")
    exit(argErrCode)

data_dir = str(sys.argv[1])

in_pkl_test_file = data_dir + "/test_dataset.pkl"
in_pkl_train_file = data_dir + "/train_dataset.pkl" # Added to find training bounds
weights_file = data_dir + "/model_weights.pth"

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# 1. Load the testing data
print(f"Loading test data from {in_pkl_test_file}...")
with open(in_pkl_test_file, "rb") as fptr:
    test_data = pickle.load(fptr)

X_test = test_data['X_test']
Y_test = test_data['Y_test']

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

# 1.5 Load training data to find the X bounds for plotting
has_train_bounds = False
try:
    with open(in_pkl_train_file, "rb") as fptr:
        train_data = pickle.load(fptr)
    X_train = train_data['X_train']
    train_min_x = np.min(X_train)
    train_max_x = np.max(X_train)
    has_train_bounds = True
    print(f"Training data bounds found: [{train_min_x:.4f}, {train_max_x:.4f}]")
except FileNotFoundError:
    print(f"Warning: {in_pkl_train_file} not found. Cannot mark training range on plot.")

# 2. Instantiate the model
# NOTE: In your train.py, you instantiated the model with 3 layers: model=self_similar_model(1,1,3).
input_dim = 1
output_dim = 1
num_layers = 3
model = self_similar_model(input_dim, output_dim, num_layers).to(device)

# 3. Load the trained weights
print(f"Loading model weights from {weights_file}...")
try:
    model.load_state_dict(torch.load(weights_file, map_location=device))
except FileNotFoundError:
    print(f"Error: Weights file not found at {weights_file}. Please run train.py first.")
    exit(1)

model.eval()  # Set model to evaluation mode

# 4. Print the trained coefficients
print("\n" + "=" * 40)
print("       TRAINED MODEL COEFFICIENTS")
print("=" * 40)

# Print A
if isinstance(model.A, nn.Parameter):
    print(f"A (Learned) = {model.A.item():.6f}")
else:
    print(f"A (Constant) = {model.A.item():.6f}")

# Print layer parameters (a_i and n_i)
for i, layer in enumerate(model.layers):
    # Extract raw weights for a_i
    raw_a_real = layer.linear.real_linear.weight.data
    raw_a_imag = layer.linear.imag_linear.weight.data

    # Apply softplus to real part to get the effective a_real used in forward pass
    effective_a_real = F.softplus(raw_a_real).item()
    effective_a_imag = raw_a_imag.item()

    # Extract n_i
    n_val = layer.power.n.data.item()  # This is a complex number
    n_real = n_val.real
    n_imag = n_val.imag

    print(f"\nLayer {i}:")
    print(f"  a_{i} = {effective_a_real:.6f} + {effective_a_imag:.6f}j")
    print(f"  n_{i} = {n_real:.6f} + {n_imag:.6f}j")
print("=" * 40 + "\n")

# 5. Evaluate the model
with torch.no_grad():
    predictions, _ = model(X_test_tensor)
    # print(predictions)
    # Compute Test MSE (same logic as train.py)
    mse_loss = torch.mean(torch.abs(predictions - Y_test_tensor.to(torch.cfloat)) ** 2)
    print(f"Test MSE Loss: {mse_loss.item():.6e}")

# 6. Prepare data for plotting
# The model outputs complex numbers, but our target is purely real.
# We will plot the real part of the predictions.
predictions_np = predictions.cpu().numpy()
predictions_real = np.real(predictions_np)

# 7. Plot the results
plt.figure(figsize=(10, 6))

# Mark the training range if we successfully loaded the training data
if has_train_bounds:
    # Adds a light gray shaded region for the training data range
    plt.axvspan(train_min_x, train_max_x, color='gray', alpha=0.2, label="Training Data Range")
    # Adds dotted vertical lines at the exact boundaries
    plt.axvline(x=train_min_x, color='gray', linestyle=':')
    plt.axvline(x=train_max_x, color='gray', linestyle=':')

plt.plot(X_test, Y_test, label="True Function (Exact)", color='blue', linewidth=2)
plt.plot(X_test, predictions_real, label="Model Prediction", color='red', linestyle='dashed', linewidth=2)

plt.title("Model Predictions vs True Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

# Save the plot and show it
plot_path = data_dir + "/test_results.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
# plt.show()