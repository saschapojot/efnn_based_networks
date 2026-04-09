from structure import *
import torch
import numpy as np
import sys
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import os
import scipy.special
from pathlib import Path

def J_and_derivs(n, x):
    """
    Computes the Jacobi polynomial J_n^{2,2} and its 1st and 2nd derivatives at points x
    using numerically stable recursive evaluation.
    """
    # Evaluate J_n^{2,2}(x) directly
    J_val = scipy.special.eval_jacobi(n, 2, 2, x)

    # 1st derivative
    if n >= 1:
        dJ_val = 0.5 * (n + 5) * scipy.special.eval_jacobi(n - 1, 3, 3, x)
    else:
        dJ_val = np.zeros_like(x)

    # 2nd derivative
    if n >= 2:
        d2J_val = 0.25 * (n + 5) * (n + 6) * scipy.special.eval_jacobi(n - 2, 4, 4, x)
    else:
        d2J_val = np.zeros_like(x)

    return J_val, dJ_val, d2J_val

def generate_quadrature(alpha,beta,n,M):
    """
    Generates Gauss-Jacobi quadrature nodes and weights, and evaluates
    the Jacobi polynomial J_n^{2,2} and its derivatives at these nodes.
    Args:
        alpha: Power of (1-x) in the weight function
        beta: Power of (1+x) in the weight function
        n: degree of the Jacobi polynomial
        M: Number of quadrature nodes

    Returns: quadrature nodes and weights for w^{alpha,beta}(x), value of J, J', J''
    """
    # Generate M Gauss-Jacobi quadrature nodes and weights
    x, weights = scipy.special.roots_jacobi(M, alpha, beta)
    # Evaluate J_n^{2,2} and its derivatives at the quadrature nodes x
    J_val, dJ_val, d2J_val = J_and_derivs(n, x)

    return x, weights, J_val, dJ_val, d2J_val

def precompute_matrices(alpha, beta, N, Q):
    """
    Precomputes the quadrature nodes, weights, and Jacobi polynomial terms
    for n = 0, 1, ..., N-1 and M = Q quadrature nodes.
    Args:
        alpha: Power of (1-x) in the weight function
        beta: Power of (1+x) in the weight function
        N:  Number of test functions (degrees n = 0 to N-1)
        Q: Number of quadrature nodes (indices m = 0 to Q-1)

    Returns:
         x: Quadrature nodes (1D array of size Q)
         weights: Quadrature weights (1D array of size Q)
         J_mat: Matrix of J_n^{2,2}(x_m) of shape (N, Q)
         dJ_mat: Matrix of 1st derivatives of shape (N, Q)
         d2J_mat: Matrix of 2nd derivatives of shape (N, Q)
    """
    # Nodes and weights only depend on Q, alpha, and beta.
    # We can get them by calling generate_quadrature for any n (e.g., n=0).
    x, weights, _, _, _ = generate_quadrature(alpha, beta, 0, Q)

    # Initialize matrices of shape (N, Q)
    #Each row corresponds to a specific polynomial degree n
    #Each column corresponds to a specific quadrature node xm
    J_mat = np.zeros((N, Q))
    dJ_mat = np.zeros((N, Q))
    d2J_mat = np.zeros((N, Q))
    # Fill the matrices for each degree n
    for n in range(N):
        J_val, dJ_val, d2J_val = J_and_derivs(n, x)
        J_mat[n, :] = J_val
        dJ_mat[n, :] = dJ_val
        d2J_mat[n, :] = d2J_val
    return x, weights, J_mat, dJ_mat, d2J_mat


eps=0.01
#Number of test functions (degrees n = 0 to N-1)
N=800
#Number of quadrature nodes (indices m = 0 to Q-1)
Q=2000
#term 0,2
x02,weights02, J_mat02, dJ_mat02, d2J_mat02=precompute_matrices(0,2,N,Q)
#term 1,1
x11,weights11, J_mat11, dJ_mat11, d2J_mat11=precompute_matrices(1,1,N,Q)
#term 1,2
x12,weights12, J_mat12, dJ_mat12, d2J_mat12=precompute_matrices(1,2,N,Q)
#term 2,1
x21,weights21, J_mat21, dJ_mat21, d2J_mat21=precompute_matrices(2,1,N,Q)

#term 2,0
x20,weights20, J_mat20, dJ_mat20, d2J_mat20=precompute_matrices(2,0,N,Q)

#term 2,2
x22,weights22, J_mat22, dJ_mat22, d2J_mat22=precompute_matrices(2,2,N,Q)
# Convert precomputed numpy arrays to PyTorch tensors globally so they don't need to be converted every training step
# We need x02 to be shape (Q, 1) for the neural network input
x02_t = torch.tensor(x02, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights02_t = torch.tensor(weights02, dtype=torch.float32)
J_mat02_t = torch.tensor(J_mat02, dtype=torch.float32)


x11_t = torch.tensor(x11, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights11_t = torch.tensor(weights11, dtype=torch.float32)
J_mat11_t = torch.tensor(J_mat11, dtype=torch.float32)

x12_t = torch.tensor(x12, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights12_t = torch.tensor(weights12, dtype=torch.float32)
J_mat12_t = torch.tensor(J_mat12, dtype=torch.float32)
dJ_mat12_t = torch.tensor(dJ_mat12, dtype=torch.float32)

x21_t = torch.tensor(x21, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights21_t = torch.tensor(weights21, dtype=torch.float32)
J_mat21_t = torch.tensor(J_mat21, dtype=torch.float32)
dJ_mat21_t = torch.tensor(dJ_mat21, dtype=torch.float32)

x20_t = torch.tensor(x20, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights20_t = torch.tensor(weights20, dtype=torch.float32)
J_mat20_t = torch.tensor(J_mat20, dtype=torch.float32)

x22_t = torch.tensor(x22, dtype=torch.float32, requires_grad=True).unsqueeze(1)
weights22_t = torch.tensor(weights22, dtype=torch.float32)
J_mat22_t = torch.tensor(J_mat22, dtype=torch.float32)
dJ_mat22_t = torch.tensor(dJ_mat22, dtype=torch.float32)
d2J_mat22_t = torch.tensor(d2J_mat22, dtype=torch.float32)

def physical_loss(model):
    """
    computes loss using quadratures
    Args:
        model:

    Returns:

    """
    #loss for 0,2
    y02 = model(x02_t).squeeze() # Squeeze turns shape (Q, 1) into (Q,)
    # loss for 0,2: 2 * eps * \int_{-1}^{1} dx w^{0,2}(x) J_n^{2,2}(x) y(x)
    # weights02_t * y02 performs element-wise multiplication for the Q nodes
    # torch.matmul multiplies the (N, Q) matrix by the (Q,) vector to sum over the Q nodes, resulting in an (N,) vector
    loss02 = 2 * eps * torch.matmul(J_mat02_t, weights02_t * y02)

    #loss for 1,1
    y11 = model(x11_t).squeeze()
    # loss for 1,1: -8 * eps * \int_{-1}^{1} dx w^{1,1}(x) J_n^{2,2}(x) y(x)
    loss11 = -8 * eps * torch.matmul(J_mat11_t, weights11_t * y11)

    #loss for 1,2
    y12 = model(x12_t).squeeze()
    # loss for 1,2: \int_{-1}^{1} dx w^{1,2}(x) (-4 * eps * \partial_x J_n^{2,2}(x) + 2 * J_n^{2,2}(x)) y(x)
    # We first compute the term inside the parenthesis for all N and Q
    term12 = -4 * eps * dJ_mat12_t + 2 * J_mat12_t
    # Then we multiply by weights and y(x), and sum over Q
    loss12 = torch.matmul(term12, weights12_t * y12)

    #loss for 2,1
    y21 = model(x21_t).squeeze()
    # loss for 2,1: \int_{-1}^{1} dx w^{2,1}(x) (4 * eps * \partial_x J_n^{2,2}(x) - 2 * J_n^{2,2}(x)) y(x)
    term21 = 4 * eps * dJ_mat21_t - 2 * J_mat21_t
    loss21 = torch.matmul(term21, weights21_t * y21)

    #loss for 2,0
    y20 = model(x20_t).squeeze()
    # loss for 2,0: 2 * eps * \int_{-1}^{1} dx w^{2,0}(x) J_n^{2,2}(x) y(x)
    loss20 = 2 * eps * torch.matmul(J_mat20_t, weights20_t * y20)

    #loss for 2,2
    #loss for 2,2
    y22 = model(x22_t).squeeze()
    # loss for 2,2: \int_{-1}^{1} dx w^{2,2}(x) (eps * \partial_x^2 J_n^{2,2}(x) - \partial_x J_n^{2,2}(x) + J_n^{2,2}(x)) y(x)
    term22 = eps * d2J_mat22_t - dJ_mat22_t + J_mat22_t
    loss22 = torch.matmul(term22, weights22_t * y22)

    loss=loss02+loss11+loss12+loss21+loss20+loss22

    # To train the network, we minimize the sum (or mean) of the squared residuals
    scalar_loss = torch.sum(loss**2)

    return scalar_loss



argErrCode = 4
# sys.argv[0] is the script name, sys.argv[1] is the first argument
if len(sys.argv) != 4:
    print("wrong number of arguments")
    print("example: python train.py num_epochs num_layers num_neurons" )
    exit(argErrCode)

# Read num_epochs from the first command line argument
num_epochs = int(sys.argv[1])
num_layers=int(sys.argv[2])
num_neurons=int(sys.argv[3])

in_dim = 1
a = -1         # Left boundary x = -1
b = 1.0          # Right boundary x = 1
fa = 0         # y(0) = 0
fb = 2         # y(1) = 2
# Hardcode the directory where the model will be saved
data_dir = f"./output/num_layers{num_layers}/num_neurons{num_neurons}/N{N}/Q{Q}/"
Path(data_dir).mkdir(exist_ok=True,parents=True)

learning_rate = 1e-3
learning_rate_final = 1e-5
weight_decay = 1e-5
decrease_over = 50

step_of_decrease = num_epochs // decrease_over
print(f"step_of_decrease={step_of_decrease}")
gamma = (learning_rate_final / learning_rate) ** (1 / step_of_decrease)
print(f"gamma={gamma}")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=" + str(device))
model = efnn(in_dim, num_layers, num_neurons, a, b, fa, fb, eps).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every 'decrease_over' epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)

# ==========================================
# TRAINING LOOP
# ==========================================
print("Starting training...")
start_time = datetime.now()
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Calculate the physical loss
    loss = physical_loss(model)

    # Backpropagation
    loss.backward()

    # Update the weights
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Print progress every 2 epochs or on the last epoch
    if epoch % 2 == 0 or epoch == num_epochs - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch:5d}/{num_epochs} | Loss: {loss.item():.6e} | LR: {current_lr:.6e}")

    # Save the model every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        checkpoint_path = os.path.join(data_dir, f"model_eps{eps}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

end_time = datetime.now()
print(f"Training completed in {end_time - start_time}")

# Save the final trained model
model_path = os.path.join(data_dir, f"model_eps{eps}.pth")
torch.save(model.state_dict(), model_path)
print(f"Final model saved to {model_path}")