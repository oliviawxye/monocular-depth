import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class CNN(nn.Module):
    def __init__(self, input_channels, conv_channels, kernel_sizes, pool_sizes,
                 fc_hidden_sizes, input_height, input_width, dropout=0.2):
        super(CNN, self).__init__()

        conv_layers = []
        prev_channels = input_channels
        current_h, current_w = input_height, input_width

        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(conv_channels, kernel_sizes, pool_sizes)):
            conv_layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())

            current_h = current_h - kernel_size + 1
            current_w = current_w - kernel_size + 1

            if pool_size > 1:
                conv_layers.append(nn.MaxPool2d(pool_size))
                current_h = current_h // pool_size
                current_w = current_w // pool_size

            prev_channels = out_channels

        self.conv_network = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        fc_layers = []
        flattened_size = prev_channels * current_h * current_w
        prev_size = flattened_size

        for hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, hidden_size))
            fc_layers.append(nn.BatchNorm1d(hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        fc_layers.append(nn.Linear(prev_size, 1))
        self.fc_network = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_network(x)
        x = self.flatten(x)
        x = self.fc_network(x)
        return x


class CNNModel(BaseModel):
    def __init__(self, input_channels=3, input_height=30, input_width=40):
        super().__init__("CNN")
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def search_hyperparameters(self, x_use, y_use, architectures=None,
                               weight_decays=None, lr=0.001, batch_size=64,
                               iterations=1, epochs=20, dropout=0.2):
        """Search for best CNN architecture.

        Args:
            x_use: Training features (should be reshaped images)
            y_use: Training labels
            architectures: List of architecture dictionaries
            weight_decays: Array of weight decay values to try
            lr: Learning rate
            batch_size: Batch size for training
            iterations: Number of random train/val splits
            epochs: Number of training epochs
            dropout: Dropout rate

        Returns:
            Dictionary with best hyperparameters and validation metrics
        """
        if architectures is None:
            architectures = [
                {'conv_channels': [4, 4], 'kernel_sizes': [3, 3], 'pool_sizes': [2, 1], 'fc_hidden_sizes': []},
                {'conv_channels': [8, 16], 'kernel_sizes': [3, 3], 'pool_sizes': [2, 1], 'fc_hidden_sizes': []},
                {'conv_channels': [4, 4, 8], 'kernel_sizes': [3, 3, 3], 'pool_sizes': [2, 1, 1], 'fc_hidden_sizes': []},
                {'conv_channels': [8, 16, 32], 'kernel_sizes': [3, 3, 3], 'pool_sizes': [2, 1, 1], 'fc_hidden_sizes': []},
            ]
        if weight_decays is None:
            weight_decays = [0]

        results = []

        for cnn_arch in tqdm(architectures, desc=f"{self.model_name} Architecture"):
            for wd in tqdm(weight_decays, desc="Weight Decay", leave=False):
                iter_rmses = []
                iter_losses = []

                for iteration in tqdm(range(iterations), desc="Iteration", leave=False):
                    X_train, X_val, y_train, y_val = train_test_split(
                        x_use, y_use, test_size=0.2, shuffle=True, random_state=iteration
                    )

                    X_train_tensor = torch.FloatTensor(X_train).view(-1, self.input_channels, self.input_height, self.input_width).to(self.device)
                    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
                    X_val_tensor = torch.FloatTensor(X_val).view(-1, self.input_channels, self.input_height, self.input_width).to(self.device)

                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    model = CNN(
                        input_channels=self.input_channels,
                        conv_channels=cnn_arch['conv_channels'],
                        kernel_sizes=cnn_arch['kernel_sizes'],
                        pool_sizes=cnn_arch['pool_sizes'],
                        fc_hidden_sizes=cnn_arch['fc_hidden_sizes'],
                        input_height=self.input_height,
                        input_width=self.input_width,
                        dropout=dropout
                    ).to(self.device)
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
                    'architecture': cnn_arch,
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
            'conv_channels': best_result['architecture']['conv_channels'],
            'kernel_sizes': best_result['architecture']['kernel_sizes'],
            'pool_sizes': best_result['architecture']['pool_sizes'],
            'fc_hidden_sizes': best_result['architecture']['fc_hidden_sizes'],
            'weight_decay': best_result['wd'],
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout': dropout,
            'input_channels': self.input_channels,
            'input_height': self.input_height,
            'input_width': self.input_width
        }
        self.validation_rmse = float(best_result['val_rmse_mean'])

        return {
            'best_params': self.best_params,
            'validation_rmse': float(best_result['val_rmse_mean']),
            'validation_std': float(best_result['val_rmse_std'])
        }

    def train(self, x_train, y_train, conv_channels=None, kernel_sizes=None,
              pool_sizes=None, fc_hidden_sizes=None, weight_decay=None,
              lr=None, batch_size=None, epochs=None, dropout=None):
        """Train the model with best hyperparameters.

        Args:
            x_train: Training features (should be reshaped images)
            y_train: Training labels
            conv_channels: List of convolutional channel sizes
            kernel_sizes: List of kernel sizes
            pool_sizes: List of pooling sizes
            fc_hidden_sizes: List of fully connected layer sizes
            weight_decay: Weight decay for optimizer
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            dropout: Dropout rate
        """
        if conv_channels is None and self.best_params is not None:
            conv_channels = self.best_params['conv_channels']
            kernel_sizes = self.best_params['kernel_sizes']
            pool_sizes = self.best_params['pool_sizes']
            fc_hidden_sizes = self.best_params['fc_hidden_sizes']
            weight_decay = self.best_params['weight_decay']
            lr = self.best_params['lr']
            batch_size = self.best_params['batch_size']
            epochs = self.best_params['epochs']
            dropout = self.best_params['dropout']
        elif conv_channels is None:
            raise ValueError("Hyperparameters must be provided or searched first")

        X_train_tensor = torch.FloatTensor(x_train).view(-1, self.input_channels, self.input_height, self.input_width).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = CNN(
            input_channels=self.input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes,
            fc_hidden_sizes=fc_hidden_sizes,
            input_height=self.input_height,
            input_width=self.input_width,
            dropout=dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

    def predict(self, x):
        """Make predictions on input data.

        Args:
            x: Input features (should be reshaped images)

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        X_tensor = torch.FloatTensor(x).view(-1, self.input_channels, self.input_height, self.input_width).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor).cpu().detach()
            predictions = np.array(output.tolist()).flatten()

        return predictions
