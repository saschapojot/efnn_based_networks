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
in_pkl_train_file = data_dir + "/train_dataset.pkl"  # Added to find training bounds
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

# 1.5 Load training data to find the X bounds for plotting and MSE splitting
has_train_bounds = False
train_min_x = 0.0  # Default fallback
try:
    with open(in_pkl_train_file, "rb") as fptr:
        train_data = pickle.load(fptr)
    X_train = train_data['X_train']
    train_min_x = float(np.min(X_train))
    train_max_x = float(np.max(X_train))
    has_train_bounds = True
    print(f"Training data bounds found: [{train_min_x:.4f}, {train_max_x:.4f}]")
except FileNotFoundError:
    print(f"Warning: {in_pkl_train_file} not found. Cannot mark training range or split MSE.")


# 2. Instantiate the model
# Using train_min_x as x_left to enforce the boundary constraint from the updated structure.py
input_dim=1
num_neurons=3
num_layers=3 #layers 0,1,2,3,.. num_layers
left_boundary=0
left_boundary_value=1
model=efnn(input_dim, num_layers, num_neurons,left_boundary,left_boundary_value).to(device)
# 3. Load the trained weights
print(f"Loading model weights from {weights_file}...")
try:
    model.load_state_dict(torch.load(weights_file, map_location=device))
except FileNotFoundError:
    print(f"Error: Weights file not found at {weights_file}. Please run train.py first.")
    exit(1)

model.eval()  # Set model to evaluation mode
# 4. Perform predictions
print("Evaluating model on test data...")
with torch.no_grad():
    Y_pred_tensor = model(X_test_tensor)

Y_pred = Y_pred_tensor.cpu().numpy()

# 5. Calculate Metrics
mse_total = F.mse_loss(Y_pred_tensor, Y_test_tensor).item()
print(f"\n--- Evaluation Metrics ---")
print(f"Total Test MSE: {mse_total:.6e}")

if has_train_bounds:
    # Split MSE into interpolation (inside training bounds) and extrapolation (outside)
    X_test_flat = X_test.flatten()
    in_bounds_mask = (X_test_flat >= train_min_x) & (X_test_flat <= train_max_x)
    out_bounds_mask = ~in_bounds_mask

    if np.any(in_bounds_mask):
        mse_in = np.mean((Y_pred[in_bounds_mask] - Y_test[in_bounds_mask])**2)
        print(f"Test MSE (Interpolation / Inside bounds): {mse_in:.6e}")
    if np.any(out_bounds_mask):
        mse_out = np.mean((Y_pred[out_bounds_mask] - Y_test[out_bounds_mask])**2)
        print(f"Test MSE (Extrapolation / Outside bounds): {mse_out:.6e}")
print("--------------------------\n")

# 6. Plotting
print("Generating plots...")
plt.figure(figsize=(10, 6))

# Sort the values by X so the line plot connects nicely
sort_idx = np.argsort(X_test.flatten())
X_plot = X_test.flatten()[sort_idx]
Y_true_plot = Y_test.flatten()[sort_idx]
Y_pred_plot = Y_pred.flatten()[sort_idx]

# Plot True vs Predicted
plt.plot(X_plot, Y_true_plot, label="True Values", color="blue", linewidth=2)
plt.plot(X_plot, Y_pred_plot, label="Predicted Values", color="red", linestyle="--", linewidth=2)

# Mark the training boundaries if available
if has_train_bounds:
    plt.axvline(x=train_min_x, color='gray', linestyle=':', label='Train Min Bound')
    plt.axvline(x=train_max_x, color='gray', linestyle=':', label='Train Max Bound')

# Mark the hard constraint boundary
plt.scatter([left_boundary], [left_boundary_value], color='green', zorder=5, s=100, label='Hard Constraint (Boundary)')

# Formatting the plot
plt.xlabel("Input (X)")
plt.ylabel("Output (Y)")
plt.title("Model Predictions vs True Values")
plt.legend()
plt.grid(True, alpha=0.3)

# Save and show the plot
plot_path = data_dir + "/test_results_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {plot_path}")

# plt.show()