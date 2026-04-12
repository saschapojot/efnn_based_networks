import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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


argErrCode = 4
if len(sys.argv) != 3:
    print("wrong number of arguments")
    print("example: python test.py num_layers num_neurons")
    sys.exit(argErrCode)

# Read hyperparameters from command line arguments
num_layers = int(sys.argv[1])
num_neurons = int(sys.argv[2])

# Constants matching train.py
in_dim = 1
a = -1.0
b = 1.0
fa = 0.0
fb = 2.0
eps = 0.01
N = 800
Q = 2000

# Locate the saved model directory
data_dir = f"./output_eps{eps}/num_layers{num_layers}/num_neurons{num_neurons}/N{N}/Q{Q}/"
model_path = os.path.join(data_dir, f"model_eps{eps}.pth")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and load weights
model = efnn(in_dim, num_layers, num_neurons, a, b, fa, fb, eps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Set model to evaluation mode

# Generate dense test points for smooth plotting
num_test_points = 1000
x_test_np = np.linspace(a, b, num_test_points)

# Convert to tensor for the model
x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1).to(device)

# Get numerical predictions from the model
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
    y_pred_np = y_pred_tensor.cpu().squeeze().numpy()

# Get exact analytical solution
y_exact_np = exact_solution(x_test_np, a, b, fa, fb, eps)

# Calculate Mean Squared Error
mse = np.mean((y_pred_np - y_exact_np)**2)
print(f"Mean Squared Error vs Analytical: {mse:.6e}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_test_np, y_exact_np, label='Analytical Solution', color='blue', linewidth=2)
plt.plot(x_test_np, y_pred_np, '--', label='EFNN Numerical Solution', color='red', linewidth=2)

plt.title(f"EFNN vs Analytical (eps={eps}, layers={num_layers}, neurons={num_neurons})")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Save the plot
plot_path = os.path.join(data_dir, f"comparison_eps{eps}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")