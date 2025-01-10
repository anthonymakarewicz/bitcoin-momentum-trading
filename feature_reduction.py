import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        
        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        z = self.encoder(x)        # compress to latent
        x_recon = self.decoder(z)  # reconstruct
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def fit(self, X, epochs=20, batch_size=128, lr=1e-3, device='cpu'):
        """
        Train the autoencoder on data X (numpy array or torch tensor).
        
        Args:
            X (np.ndarray or torch.Tensor): data of shape (n_samples, input_dim).
            epochs (int): number of training epochs.
            batch_size (int): mini-batch size.
            lr (float): learning rate.
            device (str): 'cpu' or 'cuda' for GPU.
        """
        self.to(device)

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        # Create dataset & dataloader (AE target = input)
        dataset = TensorDataset(X_tensor, X_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(device)
                
                # Forward prop
                x_recon = self(batch_x)
                loss = self.criterion(x_recon, batch_x)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
            
            avg_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return self

    def transform(self, X, batch_size=128, device='cpu'):
        """
        Encode input data X into the latent space.

        Args:
            X (np.ndarray or torch.Tensor): shape (n_samples, input_dim).
            batch_size (int): batch size for inference.
            device (str): 'cpu' or 'cuda' for GPU.

        Returns:
            z_all (torch.Tensor): shape (n_samples, latent_dim).
        """
        self.to(device)

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)        
        data_loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
        
        z_list = []
        self.eval()
        with torch.no_grad():
            for (batch_x,) in data_loader:
                batch_x = batch_x.to(device)
                z = self.encode(batch_x)
                z_list.append(z.cpu())
        
        z_all = torch.cat(z_list, dim=0)
        z_all = pd.DataFrame(
            z_all.numpy(), 
            index=X.index, 
            columns=[f"encoded_{i+1}" for i in range(z_all.shape[1])]
        )

        return z_all