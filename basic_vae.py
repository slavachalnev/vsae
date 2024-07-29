import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the simplified VAE model with separate mu and logvar layers
class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        
        # Encoder (separate layers for mu and logvar)
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder (single layer)
        self.fc_decoder = nn.Linear(latent_dim, input_dim)
    
    def encode(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return torch.sigmoid(self.fc_decoder(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

# Main training loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_dim = 784  # 28x28 MNIST images
    latent_dim = 20
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3

    # Load MNIST dataset
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = SimpleVAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch {epoch}, Loss: {train_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'simple_vae_model.pth')

if __name__ == '__main__':
    main()