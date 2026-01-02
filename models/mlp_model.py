import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPModel(BaseModel):
    def __init__(self):
        super().__init__("MLP")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def search_hyperparameters(self, x_use, y_use, hidden_architectures=None,
                               weight_decays=None, lr=0.001, batch_size=64,
                               iterations=10, epochs=10, dropout=0.2):
        """Search for best architecture and weight decay.

        Args:
            x_use: Training features
            y_use: Training labels
            hidden_architectures: List of hidden layer architectures to try
            weight_decays: Array of weight decay values to try
            lr: Learning rate
            batch_size: Batch size for training
            iterations: Number of random train/val splits
            epochs: Number of training epochs
            dropout: Dropout rate

        Returns:
            Dictionary with best hyperparameters and validation metrics
        """
        if hidden_architectures is None:
            hidden_architectures = [
                [8], [16], [32],
                [8, 8], [16, 16], [32, 32],
                [8, 8, 8], [16, 16, 16], [32, 32, 32]
            ]
        if weight_decays is None:
            weight_decays = [0]

        input_size = x_use.shape[1]
        results = []

        for hidden_sizes in tqdm(hidden_architectures, desc=f"{self.model_name} Architecture"):
            for wd in tqdm(weight_decays, desc="Weight Decay", leave=False):
                iter_rmses = []
                iter_losses = []

                for iteration in tqdm(range(iterations), desc="Iteration", leave=False):
                    X_train, X_val, y_train, y_val = train_test_split(
                        x_use, y_use, test_size=0.2, random_state=iteration
                    )

                    X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)

                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    model = MLP(input_size, hidden_sizes, dropout=dropout).to(self.device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    train_losses = []

                    for epoch in range(epochs):
                        model.train()
                        epoch_loss = 0
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()

                        avg_loss = epoch_loss / len(train_loader)
                        train_losses.append(avg_loss)

                    model.eval()
                    with torch.no_grad():
                        y_val_pred = model(X_val_tensor).cpu().numpy().flatten()
                        y_val_pred = np.maximum(y_val_pred, 0)
                        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

                    iter_rmses.append(val_rmse)
                    iter_losses.append(train_losses)

                loss_array = np.array(iter_losses)
                mean_losses = np.mean(loss_array, axis=0)
                std_losses = np.std(loss_array, axis=0)

                results.append({
                    'hidden': hidden_sizes,
                    'wd': wd,
                    'val_rmse_mean': np.mean(iter_rmses),
                    'val_rmse_std': np.std(iter_rmses),
                    'val_rmses': iter_rmses,
                    'train_losses': iter_losses,
                    'train_loss_mean': mean_losses,
                    'train_loss_std': std_losses
                })

        best_result = min(results, key=lambda x: x['val_rmse_mean'])

        self.best_params = {
            'hidden_sizes': best_result['hidden'],
            'weight_decay': best_result['wd'],
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout': dropout
        }
        self.validation_rmse = float(best_result['val_rmse_mean'])

        return {
            'best_params': self.best_params,
            'validation_rmse': float(best_result['val_rmse_mean']),
            'validation_std': float(best_result['val_rmse_std'])
        }

    def train(self, x_train, y_train, hidden_sizes=None, weight_decay=None,
              lr=None, batch_size=None, epochs=None, dropout=None):
        """Train the model with best hyperparameters.

        Args:
            x_train: Training features
            y_train: Training labels
            hidden_sizes: List of hidden layer sizes
            weight_decay: Weight decay for optimizer
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            dropout: Dropout rate
        """
        if hidden_sizes is None and self.best_params is not None:
            hidden_sizes = self.best_params['hidden_sizes']
            weight_decay = self.best_params['weight_decay']
            lr = self.best_params['lr']
            batch_size = self.best_params['batch_size']
            epochs = self.best_params['epochs']
            dropout = self.best_params['dropout']
        elif hidden_sizes is None:
            raise ValueError("Hyperparameters must be provided or searched first")

        input_size = x_train.shape[1]

        train_dataset = TensorDataset(
            torch.FloatTensor(x_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = MLP(input_size, hidden_sizes, dropout=dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

    def predict(self, x):
        """Make predictions on input data.

        Args:
            x: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        X_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor).cpu().detach()
            predictions = np.array(output.tolist()).flatten()

        return predictions
