import torch
import torch.nn as nn

    
class GreenGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.model(x)
    
    
class GreenDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.model(x)





