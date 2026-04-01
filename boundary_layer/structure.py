import torch.nn as nn
import torch

from torch.utils.data import Dataset, DataLoader

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
                # ReciprocalActivation(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_dim, num_neurons),
                nn.Tanh(),
                # ReciprocalActivation(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

    def forward(self, x):
        self.a=self.a.to(x.device)
        self.b=self.b.to(x.device)
        self.fa=self.fa.to(x.device)
        self.fb=self.fb.to(x.device)
        S = x
        for i in range(1, self.num_layers + 1):
            # Compute effective field layer Fi
            Fi = self.effective_field_layers[i - 1](S)
            # Compute quasi-particle layer Si
            Si = self.quasi_particle_layers[i - 1](x) * Fi
            # Update S for the next layer
            S = Si

        E_raw = S.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
        # --- HARD CONSTRAINT ENFORCEMENT ---
        dist_left=torch.norm(x - self.a, p=2, dim=1, keepdim=True)
        dist_right=torch.norm( self.b-x, p=2, dim=1, keepdim=True)
        # 1. Boundary Interpolation (Exact at boundaries)
        # When x = a, dist_left = 0, so this term becomes exactly fa
        # When x = b, dist_right = 0, so this term becomes exactly fb
        boundary_interp = (dist_right * self.fa + dist_left * self.fb) / (dist_left + dist_right )
        # 2. Network Contribution (Zero at boundaries)
        # Multiplying E_raw by (dist_left * dist_right) ensures the network's
        # output is zeroed out at x=a and x=b.
        E_final = boundary_interp + (dist_left * dist_right) * E_raw

        return E_final


