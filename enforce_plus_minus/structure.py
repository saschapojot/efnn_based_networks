import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


class constrained_complex_linear_up(nn.Module):
    """
    Acts as a linear layer without bias to compute (1 + x @ a_i).
    does not enforce Re(a_i) > 0, this is for numerator factors
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.real_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.imag_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        real_part = self.real_linear(x)
        imag_part = self.imag_linear(x)
        # Add 1.0 to the real part to form exactly 1 + ax
        return torch.complex(1.0 + real_part, imag_part)



class constrained_complex_linear_down(nn.Module):
    """
    Acts as a linear layer without bias to compute (1 + x @ a_i).
    Enforces Re(a_i) > 0 using softplus on the real weights.
    This is for denominator factors
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
    # Added a flag to optionally enforce Re(n) > 0
    def __init__(self, output_dim, enforce_positive_real=True):
        super().__init__()
        self.n = nn.Parameter(torch.randn(1, output_dim, dtype=torch.cfloat))
        self.enforce_positive_real = enforce_positive_real

    @property
    def effective_n(self):
        """Returns the actual power used in the forward pass."""
        if self.enforce_positive_real:
            # Enforce Re(n) > 0 using softplus on the real part
            return torch.complex(F.softplus(self.n.real), self.n.imag)
        return self.n

    def forward(self, z):
        # Use effective_n instead of self.n directly
        return z ** self.effective_n

class  irrational_factor(nn.Module):
    """
    computing (1+ai *x)**ni/(1+bi*x)**mi
    """

    def __init__(self, input_dim, output_dim,enforce_positive_real=True):
        super().__init__()
        self.linear_up = constrained_complex_linear_up(input_dim, output_dim)
        self.linear_down=constrained_complex_linear_down(input_dim, output_dim)

        self.power_up=power_nonlinearity(output_dim,enforce_positive_real)
        self.power_down=power_nonlinearity(output_dim,enforce_positive_real)

    def forward(self, x, epsilon=1e-8):
        #1. linear transformation for numerator
        base_up=self.linear_up(x)
        #2. power nonlinearity
        factor_up=self.power_up(base_up)

        #3. linear transformation for denominator
        base_down=self.linear_down(x)
        #4. Calculate conditional regularization
        penalty = 1.0 / (torch.abs(base_down) ** 2 + epsilon)
        reg_loss = torch.mean(penalty)
        #5. power nonlinearity
        factor_down=self.power_down(base_down)
        factor =factor_up/factor_down
        return factor,reg_loss
class self_similar_model(nn.Module):
    r"""
    The full model that grows layer by layer.
    Formula: F_L = A * \prod_{i=0}^L (1 + a_i * x)^{n_i}/(1+bi*x)^mi
    """

    def __init__(self, input_dim, output_dim, num_layers,enforce_positive_real=True, A_val=None):
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
            irrational_factor(input_dim, output_dim,enforce_positive_real)
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
