import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
class efnn(nn.Module):
    def __init__(self, in_dim, num_layers, num_neurons, a, b, fa, fb, eps):
        super().__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.a = torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a
        self.b = torch.tensor(b, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b
        self.fa = torch.tensor(fa, dtype=torch.float32) if not isinstance(fa, torch.Tensor) else fa
        self.fb = torch.tensor(fb, dtype=torch.float32) if not isinstance(fb, torch.Tensor) else fb
        self.eps = eps

        # Sub-network for A_j or B_j
        # Outputs a vector of size `num_neurons`
        def make_net():
            return nn.Sequential(
                nn.Linear(in_dim, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            )

        # We need num_layers + 1 networks for A_0 to A_n
        self.A_nets = nn.ModuleList([make_net() for _ in range(num_layers + 1)])

        # We need num_layers + 1 networks for B_0 to B_n
        self.B_nets = nn.ModuleList([make_net() for _ in range(num_layers + 1)])

    def forward(self, x):
        self.a = self.a.to(x.device)
        self.b = self.b.to(x.device)
        self.fa = self.fa.to(x.device)
        self.fb = self.fb.to(x.device)

        # Distances to boundaries, shape: [batch_size, 1]
        dist_a = x - self.a
        dist_b = x - self.b

        # ---------------------------------------------------------
        # 1. Compute I_a (Continued tanh on the left boundary)
        # ---------------------------------------------------------
        # Base case (j = 0)
        A0_val = self.A_nets[0](x) * dist_a
        F_a = torch.tanh(1.0 + A0_val)

        # First layer (j = 1)
        A1_val = self.A_nets[1](x) * dist_a
        S_a = 1.0 + A1_val * F_a

        # Subsequent layers (j = 2 to num_layers)
        for j in range(2, self.num_layers + 1):
            F_a = torch.tanh(S_a)
            Aj_val = self.A_nets[j](x) * dist_a
            S_a = 1.0 + Aj_val * F_a

        # Final activation and sum across neurons
        Ia_vec = torch.tanh(S_a)
        Ia = Ia_vec.sum(dim=1, keepdim=True)

        # ---------------------------------------------------------
        # 2. Compute I_b (Continued tanh on the right boundary)
        # ---------------------------------------------------------
        # Base case (j = 0)
        B0_val = self.B_nets[0](x) * dist_b
        F_b = torch.tanh(1.0 + B0_val)

        # First layer (j = 1)
        B1_val = self.B_nets[1](x) * dist_b
        S_b = 1.0 + B1_val * F_b

        # Subsequent layers (j = 2 to num_layers)
        for j in range(2, self.num_layers + 1):
            F_b = torch.tanh(S_b)
            Bj_val = self.B_nets[j](x) * dist_b
            S_b = 1.0 + Bj_val * F_b

        # Final activation and sum across neurons
        Ib_vec = torch.tanh(S_b)
        Ib = Ib_vec.sum(dim=1, keepdim=True)

        # ---------------------------------------------------------
        # 3. Final Output Assembly (Equation 45 adjusted for sum)
        # ---------------------------------------------------------
        # Because we summed `num_neurons` components, the boundary value of Ia and Ib
        # is now `num_neurons * tanh(1)`. We adjust the denominator to cancel this out.
        tanh1_sum = self.num_neurons * torch.tanh(torch.tensor(1.0, device=x.device))

        term_a = ((self.b - x) / (self.b - self.a)) * (self.fa / tanh1_sum) * Ia
        term_b = ((x - self.a) / (self.b - self.a)) * (self.fb / tanh1_sum) * Ib

        y = term_a + term_b

        return y



# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
