import torch
import torch.nn as nn

class GreenGenerator(nn.Module):
    def __init__(self, latent_dim, n_features):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_features)
        self.act = nn.ReLU()
        # Total params: (latent_dim*32+32*32+32*n_features+32+32+n_features)

    def forward(self, z):
        x = self.act(self.fc1(z))
        x = self.act(self.fc2(x))
        return self.fc3(x)

class GreenDiscriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()
        # Small size for energy efficiency

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out_act(self.fc3(x))

# ASCII summary (replace with torchinfo summary in code)
# GreenGenerator:
# (Input) -> [Linear(16 -> 32)] -> ReLU -> [Linear(32 -> 32)] -> ReLU -> [Linear(32 -> n_features)]
# GreenDiscriminator:
# (Input) -> [Linear(n_features -> 32)] -> ReLU -> [Linear(32 -> 16)] -> ReLU -> [Linear(16 -> 1)] -> Sigmoid

# For quantization/pruning:
# - Add nn.utils.prune for pruning.
# - Convert activations to quantized types (optionally in eval).
