import torch.nn as nn
import torch

from torch.utils.data import Dataset, DataLoader
# 1. Define a custom Exponential Activation layer
class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class SechActivation(nn.Module):
    def forward(self, x):
        # sech(x) = 1 / cosh(x)
        return 1.0 / torch.cosh(x)
class efnn(nn.Module):
    def __init__(self, in_dim, num_layers, num_neurons,a,b,fa,fb,eps):
        super().__init__()
        self.num_layers = num_layers
        self.a = torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a
        self.b = torch.tensor(b, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b
        self.fa = torch.tensor(fa, dtype=torch.float32) if not isinstance(fa, torch.Tensor) else fa
        self.fb = torch.tensor(fb, dtype=torch.float32) if not isinstance(fb, torch.Tensor) else fb
        self.eps=eps

        # Effective field layers (F_i)
        self.effective_field_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_dim if i == 1 else num_neurons, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_dim, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers+1)]
        )


    def forward(self, x):
        self.a=self.a.to(x.device)
        self.b=self.b.to(x.device)
        self.fa=self.fa.to(x.device)
        self.fb=self.fb.to(x.device)
        y=x-self.a
        S = y
        for i in range(1, self.num_layers + 1):
            # Compute effective field layer Fi
            Fi = self.effective_field_layers[i - 1](S)
            # Compute quasi-particle layer Si
            Si = self.quasi_particle_layers[i - 1](y) * Fi
            # Update S for the next layer
            S = Si

        E_raw = S.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
        # --- HARD CONSTRAINT ENFORCEMENT ---
        dist_left = x - self.a
        dist_right = self.b - x

        # 1. Boundary Interpolation (Exact at boundaries)
        # When x = a, dist_left = 0, so this term becomes exactly fa
        # When x = b, dist_right = 0, so this term becomes exactly fb
        boundary_interp = (dist_right * self.fa + dist_left * self.fb) / (self.b - self.a)
        # 2. Network Contribution (Zero at boundaries)
        # Instead of multiplying by dist_left (which suppresses the network linearly),
        # we use an exponential boundary layer modifier.
        # This evaluates to exactly 0 at x = a, but rises sharply with a gradient of 1/eps,
        # naturally capturing the singularly perturbed boundary layer without forcing
        # the network to output massive values.
        bl_modifier = 1.0 - torch.exp(-dist_left / self.eps)

        # We keep dist_right as is, because there is no steep boundary layer at x = b.
        E_final = boundary_interp + (bl_modifier * dist_right) * E_raw

        return E_final


