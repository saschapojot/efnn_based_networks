import torch.nn as nn
import torch


from torch.utils.data import Dataset, DataLoader

class efnn(nn.Module):
    def __init__(self, in_dim, num_layers, num_neurons,left_boundary,left_boundary_value):
        super().__init__()
        self.num_layers = num_layers
        self.left_boundary = torch.tensor(left_boundary, dtype=torch.float32) if not isinstance(left_boundary, torch.Tensor) else left_boundary
        self.left_boundary_value = torch.tensor(left_boundary_value, dtype=torch.float32) if not isinstance(left_boundary_value, torch.Tensor) else left_boundary_value

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

    def forward(self, S0):
        """

        :param S0 (torch.Tensor): Input spin configurations, shape (batch_size, in_dim).
        :return:   torch.Tensor: Predicted energy, shape (batch_size, 1).
        """
        self.left_boundary = self.left_boundary.to(S0.device)
        self.left_boundary_value = self.left_boundary_value.to(S0.device)
        # Initialize S as the input spin configuration
        S = S0
        for i in range(1, self.num_layers + 1):
            # Compute effective field layer Fi
            Fi = self.effective_field_layers[i - 1](S)
            # Compute quasi-particle layer Si
            Si = self.quasi_particle_layers[i - 1](S0) * Fi
            # Update S for the next layer
            S = Si

        # Output layer to compute energy
        E_raw = S.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
        # --- HARD CONSTRAINT ENFORCEMENT ---
        # Calculate the distance between S0 and the left_boundary.
        # If S0 is multi-dimensional, we take the norm across the features.
        # If left_boundary is a scalar, it broadcasts automatically.
        distance = torch.norm(S0 - self.left_boundary, p=2, dim=1, keepdim=True)
        # Final Energy = Boundary_Value + Distance * Raw_Network_Output
        # When S0 == left_boundary, distance is 0, so E_final == left_boundary_value
        E_final = self.left_boundary_value + distance * E_raw
        return E_final




# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]