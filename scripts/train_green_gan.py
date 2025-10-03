import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.green_gan import GreenGenerator, GreenDiscriminator
from models.ids_models import TinyIDSNet
from models.regularizers import energy_regularizer

# Load data
X_train = np.load('data/processed/train.npz')['X']
X_train = torch.tensor(X_train, dtype=torch.float32)
n_features = X_train.size(1)

G = GreenGenerator(latent_dim=16, n_features=n_features)
D = GreenDiscriminator(n_features=n_features)

# IDS model for attack guidance
ids_model = TinyIDSNet(n_features=n_features)
# (Load pretrained weights...)

# Training setup (loss, optimizer)
adv_criterion = nn.BCELoss()
optim_G = torch.optim.Adam(G.parameters(), lr=1e-3)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-3)

batch_size = 128
epochs = 3  # demo setting; use more in full mode

dl = DataLoader(X_train, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, real_x in enumerate(dl):
        z = torch.randn(real_x.size(0), 16)
        fake_x = G(z)

        ## Train D
        real_y = torch.ones(real_x.size(0), 1)
        fake_y = torch.zeros(real_x.size(0), 1)
        D_loss = adv_criterion(D(real_x), real_y) + adv_criterion(D(fake_x.detach()), fake_y)
        optim_D.zero_grad()
        D_loss.backward()
        optim_D.step()

        ## Train G: fool D and attack IDS, penalize energy
        attack_labels = torch.ones(real_x.size(0), dtype=torch.long)  # Label: should be attack samples
        ids_logits = ids_model(fake_x)
        ids_probs = nn.Softmax(dim=1)(ids_logits)[:,1]
        fool_loss = -torch.log(ids_probs + 1e-7).mean()

        G_loss = adv_criterion(D(fake_x), real_y) + fool_loss + 0.1*energy_regularizer(real_x, fake_x)
        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()

