import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class CategoricalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_categories):
        super(CategoricalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * num_categories)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * num_categories, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def sample(self, num_samples):
        # Sample from a discrete uniform categorical distribution
        z = torch.randint(0, self.num_categories, (num_samples, self.latent_dim), device=next(self.parameters()).device)
        z_one_hot = F.one_hot(z, num_classes=self.num_categories).float()
        return z_one_hot
    
    def encode(self, x):
        return self.encoder(x).view(-1, self.latent_dim, self.num_categories)
    
    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret
    
    def decode(self, z):
        return self.decoder(z.view(-1, self.latent_dim * self.num_categories))
    
    def forward(self, x, temperature=1.0):
        logits = self.encode(x.view(-1, 784))
        z = self.gumbel_softmax(logits, temperature)
        return self.decode(z), logits

def loss_function(recon_x, x, logits):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    q = F.softmax(logits, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    KLD = torch.sum(q * (log_q - torch.log(torch.tensor(1.0 / logits.size(-1)))))
    
    return BCE + KLD

def train(model, train_loader, optimizer, device, temperature):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, logits = model(data, temperature)
        loss = loss_function(recon_batch, data, logits)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def plot_generated_images(images, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_images.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_dim = 784  # 28x28 MNIST images
    latent_dim = 20
    num_categories = 10
    batch_size = 128
    epochs = 40
    learning_rate = 1e-3
    initial_temperature = 1.0
    min_temperature = 0.1
    anneal_rate = 0.02

    # Load MNIST dataset
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = CategoricalVAE(input_dim, latent_dim, num_categories).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    temperature = initial_temperature
    for epoch in range(1, epochs + 1):
        temperature = max(initial_temperature * math.exp(-anneal_rate * epoch), min_temperature)
        train_loss = train(model, train_loader, optimizer, device, temperature)
        print(f'Epoch {epoch}, Loss: {train_loss:.4f}, Temperature: {temperature:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'categorical_vae_model.pth')

    # Sample and generate images
    num_samples = 16
    model.eval()
    with torch.no_grad():
        z = model.sample(num_samples)
        generated_images = model.decode(z)

    # Visualize the generated images
    plot_generated_images(generated_images)


if __name__ == '__main__':
    main()
