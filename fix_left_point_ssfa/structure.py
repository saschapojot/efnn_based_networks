import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class constrained_complex_linear(nn.Module):
    """
    Acts as a linear layer without bias to compute (1 + x @ a_i).
    Enforces Re(a_i) > 0 using softplus on the real weights.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.real_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.imag_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        # Apply softplus to weights to ensure Re(a) > 0
        real_part = F.linear(x, F.softplus(self.real_linear.weight))
        imag_part = self.imag_linear(x)

        # Add 1.0 to the real part to form exactly 1 + ax
        return torch.complex(1.0 + real_part, imag_part)

class power_nonlinearity(nn.Module):
    """
    Applies the complex power nonlinearity: z^{n_i}
    """

    def __init__(self, output_dim):
        super().__init__()
        self.n = nn.Parameter(torch.randn(1, output_dim, dtype=torch.cfloat))

    def forward(self, z):
        return z ** self.n



class self_similar_factor_layer(nn.Module):
    """
    A single layer computing: (1 + a_i * x)^{n_i}
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = constrained_complex_linear(input_dim, output_dim)
        self.power = power_nonlinearity(output_dim)

    def forward(self, x, epsilon=1e-8):
        # 1. Linear transformation (now directly outputs 1 + ax)
        base = self.linear(x)
        # 2. Calculate conditional regularization
        n = self.power.n
        mask = (n.real < 0).float()
        penalty = 1.0 / (torch.abs(base) ** 2 + epsilon)
        reg_loss = torch.mean(mask * penalty)
        # 3. Power nonlinearity
        factor = self.power(base)
        return factor, reg_loss


class self_similar_model(nn.Module):
    r"""
    The full model that grows layer by layer.
    Formula: F_L = A * \prod_{i=0}^L (1 + a_i * x)^{n_i}
    """

    def __init__(self, input_dim, output_dim, num_layers, A_val=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if A_val is None:
            # A is a learnable parameter
            self.A = nn.Parameter(torch.randn(1, output_dim, dtype=torch.float))
        else:
            # A is a fixed constant
            A_tensor = torch.full((1, output_dim), float(A_val), dtype=torch.float)
            self.register_buffer('A', A_tensor)

        # Instantiate all layers upfront
        self.layers = nn.ModuleList([
            self_similar_factor_layer(input_dim, output_dim)
            for _ in range(num_layers + 1)
        ])

    def forward(self, x):
        F_val = self.A
        total_reg_loss = 0.0
        # Multiplicative skip connections: F_val = F_val * factor
        for layer in self.layers:
            factor, reg_loss = layer(x)
            F_val = F_val * factor
            total_reg_loss = total_reg_loss + reg_loss

        return F_val, total_reg_loss


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
