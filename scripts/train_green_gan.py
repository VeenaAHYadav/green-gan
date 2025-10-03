import torch
import torch.nn as nn
import torch.optim as optim


latent_dim = 16         
n_features = 77         
batch_size = 128
lr = 0.0002
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GreenGenerator(nn.Module):
    def __init__(self, latent_dim, n_features):
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


# Instantiate models
G = GreenGenerator(latent_dim, n_features).to(device)
D = GreenDiscriminator(n_features).to(device)

# Loss and optimizers
adv_criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)


real_data_samples = 10000
real_dataset = torch.rand(real_data_samples, n_features).to(device)


for epoch in range(num_epochs):
    for i in range(0, real_data_samples, batch_size):
        
        real_x = real_dataset[i:i+batch_size]
        current_batch = real_x.size(0)

        # Labels
        real_y = torch.ones(current_batch, 1).float().to(device)
        fake_y = torch.zeros(current_batch, 1).float().to(device)

        
        z = torch.randn(current_batch, latent_dim).to(device)
        fake_x = G(z)

        D_loss = adv_criterion(D(real_x), real_y) + adv_criterion(D(fake_x.detach()), fake_y)

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        
        z = torch.randn(current_batch, latent_dim).to(device)
        fake_x = G(z)
        G_loss = adv_criterion(D(fake_x), real_y)  # trick D

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}]  D_loss: {D_loss.item():.4f}  G_loss: {G_loss.item():.4f}")

print("Training complete!")
