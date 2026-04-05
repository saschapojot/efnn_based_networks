from structure import *
import pickle
import numpy as np
import sys
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import os

def physics_loss(model, x):
    """
    Computes the physics residual loss for the equation: eps * y'' + y' + y = 0
    """
    # 1. Ensure x requires gradients so we can differentiate with respect to it
    x.requires_grad_(True)

    # 2. Forward pass to get y (the model's prediction)
    y = model(x)

    # 3. Compute the first derivative: y' = dy/dx
    y_x = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,   # Crucial for higher order derivatives and backprop
        retain_graph=True
    )[0]

    # 4. Compute the second derivative: y'' = d^2y/dx^2
    y_xx = torch.autograd.grad(
        outputs=y_x,
        inputs=x,
        grad_outputs=torch.ones_like(y_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # 5. Formulate the differential equation residual
    # eps * y'' + y' + y = 0
    # We use the eps stored inside the model
    residual = model.eps * y_xx + y_x + y

    # 6. The loss is the Mean Squared Error of the residual (we want it to be 0)
    loss_ode = torch.mean(residual ** 2)

    return loss_ode

argErrCode = 3
# sys.argv[0] is the script name, sys.argv[1] is the first argument
if len(sys.argv) != 2:
    print("wrong number of arguments")
    print("example: python train.py num_epochs")
    exit(argErrCode)

# Read num_epochs from the first command line argument
num_epochs = int(sys.argv[1])

# Hardcode the directory where the model will be saved
data_dir = "./output"

# --- 1. Initialize parameters and model ---
in_dim = 1
num_layers = 3
num_neurons = 6
a = 0.0          # Left boundary x = 0
b = 1.0          # Right boundary x = 1
fa = 0         # y(0) = 1
fb = 2         # y(1) = 0
eps = 0.1         # The epsilon parameter in your PDE

learning_rate = 1e-2
learning_rate_final = 1e-4
weight_decay = 1e-5
decrease_over = 50

step_of_decrease = num_epochs // decrease_over
print(f"step_of_decrease={step_of_decrease}")
gamma = (learning_rate_final / learning_rate) ** (1 / step_of_decrease)
print(f"gamma={gamma}")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=" + str(device))

batch_size = 50 # Define batch size
num_collocation_points = 1000

# --- 2. Initialize Model, Optimizer, and Scheduler ---
# Create the model and move it to the selected device
model = efnn(in_dim, num_layers, num_neurons, a, b, fa, fb, eps).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every 'decrease_over' epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)

# Ensure the data directory exists for saving outputs
os.makedirs(data_dir, exist_ok=True)

# --- 3. Training Loop ---
print("Starting training...")
start_time = datetime.now()

for epoch in range(1, num_epochs + 1):
    model.train()

    # --- MODIFIED SAMPLING FOR SMALL EPSILON ---
    num_uniform = num_collocation_points // 2
    num_boundary = num_collocation_points - num_uniform
    # 1. Uniform points across the whole domain [a, b]
    x_uniform = a + (b - a) * torch.rand((num_uniform, 1), device=device)

    # 2. Points highly concentrated near the boundary layer (x = 0)
    # We sample heavily in the region [0, 5 * eps]
    x_boundary = a + (5 * eps) * torch.rand((num_boundary, 1), device=device)

    # Combine and shuffle
    x_all = torch.cat([x_uniform, x_boundary], dim=0)


# Shuffle indices for mini-batching
    permutation = torch.randperm(num_collocation_points)

    epoch_loss = 0.0
    num_batches = 0

    # Mini-batch training
    for i in range(0, num_collocation_points, batch_size):
        # Get batch indices and corresponding points
        indices = permutation[i : i + batch_size]
        x_batch = x_all[indices]

        optimizer.zero_grad()

        # Calculate the physics loss for the current batch
        loss = physics_loss(model, x_batch)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    # Update learning rate at the end of the epoch
    scheduler.step()

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / num_batches

    # Print progress
    if epoch % 2 == 0 or epoch == 1 or epoch == num_epochs:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:5d}/{num_epochs} | Avg Physics Loss: {avg_loss:.6e} | LR: {current_lr:.6e}")

end_time = datetime.now()
print(f"Training completed in {end_time - start_time}")

# --- 4. Save the Model ---
model_path = os.path.join(data_dir, f"pinn_model_eps{eps}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully to {model_path}")