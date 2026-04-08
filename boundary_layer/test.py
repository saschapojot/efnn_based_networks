import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from structure import efnn

def exact_solution(x, a, b, fa, fb, eps):
    """
    Computes the analytical solution based on the provided PDF.
    Handles complex roots if 1 - 4*eps < 0.
    """
    # Use complex numbers to handle negative values under the square root
    discriminant = 1.0 - 4.0 * eps + 0j

    # Calculate lambda_1 and lambda_2
    lambda1 = (-1.0 + np.sqrt(discriminant)) / (2.0 * eps)
    lambda2 = (-1.0 - np.sqrt(discriminant)) / (2.0 * eps)

    # Calculate coefficients A and B
    num_A = fa - np.exp(lambda2 * (a - b)) * fb
    den_A = np.exp(lambda1 * a) - np.exp(lambda1 * b) * np.exp(lambda2 * (a - b))
    A = num_A / den_A

    num_B = fb - fa * np.exp(lambda1 * (b - a))
    den_B = np.exp(lambda2 * b) - np.exp(lambda1 * (b - a)) * np.exp(lambda2 * a)
    B = num_B / den_B

    # Calculate exact y(x)
    y = A * np.exp(lambda1 * x) + B * np.exp(lambda2 * x)

    # The physical solution is real, so we take the real part
    return np.real(y)

# --- 1. Parameters (Must match train.py) ---
in_dim = 1
num_layers = 3
num_neurons = 3
a = -1
b = 1.0
fa = 0         # y(0) = 1
fb = 2         # y(1) = 0
eps = 0.05

data_dir = "./output"
model_path = os.path.join(data_dir, f"pinn_model_eps{eps}.pth")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load the Model ---
model = efnn(in_dim, num_layers, num_neurons, a, b, fa, fb, eps).to(device)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run train.py first to generate the model.")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# --- 3. Generate Data for Plotting ---
# Create 500 evenly spaced points between a and b
x_np = np.linspace(a, b, 500).reshape(-1, 1)

# Convert to PyTorch tensor for the model
x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)

# Get PINN predictions
with torch.no_grad():
    y_pred_tensor = model(x_tensor)
y_pred_np = y_pred_tensor.cpu().numpy()

# Get Analytical solution
y_exact_np = exact_solution(x_np, a, b, fa, fb, eps)

# Calculate Mean Squared Error between PINN and Exact
mse = np.mean((y_pred_np - y_exact_np) ** 2)
print(f"Mean Squared Error vs Analytical: {mse:.6e}")

# --- 4. Plot the Results ---
plt.figure(figsize=(8, 6))

# Plot exact solution as a solid line
plt.plot(x_np, y_exact_np, label="Analytical Solution", color='blue', linewidth=2)

# Plot PINN prediction as dashed line
plt.plot(x_np, y_pred_np, label="PINN Prediction", color='red', linestyle='--', linewidth=2.5)

# Formatting the plot
plt.title(f"EFNN+PINN vs Analytical Solution (eps={eps})", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y(x)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

# Save and show the plot
plot_path = os.path.join(data_dir, f"comparison_plot_eps{eps}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

# plt.show()