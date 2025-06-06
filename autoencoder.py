import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Özellik vektörü
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        for batch, _ in dataloader:
            batch = batch.view(batch.size(0), -1)
            output = model(batch)
            loss = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        loss_list.append(avg_loss)
    return loss_list

def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(x.size(0), -1)
            encoded = model.encoder(x)
            features.append(encoded.numpy())
            labels.append(y.numpy())
    return np.vstack(features), np.concatenate(labels)

if __name__ == "__main__":
    import os
    os.makedirs("results/plots", exist_ok=True)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = train_autoencoder(model, train_loader, criterion, optimizer, epochs=10)

    features, labels = extract_features(model, test_loader)
    np.save("results/features.npy", features)
    np.save("results/labels.npy", labels)

    plt.plot(losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/plots/loss_curve.png")
