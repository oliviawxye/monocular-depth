import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class RegressionModel(BaseModel):
    def __init__(self, degree=1):
        super().__init__(f"PolynomialRegression_degree{degree}")
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()

    def search_hyperparameters(self, x_use, y_use, regularization_params=None,
                               batch_size=64, iterations=10):
        """Search for best regularization parameter.

        Args:
            x_use: Training features
            y_use: Training labels
            regularization_params: Array of regularization parameters to try
            batch_size: Batch size for SGD
            iterations: Number of random train/val splits

        Returns:
            Dictionary with best hyperparameters and validation metrics
        """
        if regularization_params is None:
            regularization_params = np.logspace(-10, 10, 20)

        rmse_training = []

        for _ in tqdm(range(iterations), desc=f"{self.model_name} Iterations"):
            rmse_iter = []

            x_train, x_valid, y_train, y_valid = train_test_split(
                x_use, y_use, test_size=0.2, shuffle=True
            )

            n_batches = len(x_train) // batch_size

            for param in regularization_params:
                model = SGDRegressor(
                    max_iter=1000,
                    learning_rate='adaptive',
                    eta0=0.001,
                    alpha=param,
                    random_state=42
                )

                poly = PolynomialFeatures(degree=self.degree)
                scaler = StandardScaler()

                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size

                    x_batch = x_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]

                    if len(x_batch) == 0:
                        continue

                    x_batch_poly = poly.fit_transform(x_batch) if i == 0 else poly.transform(x_batch)
                    x_batch_scaled = scaler.fit_transform(x_batch_poly) if i == 0 else scaler.transform(x_batch_poly)

                    model.partial_fit(x_batch_scaled, y_batch)

                valid_predictions = []
                for i in range(0, len(x_valid), batch_size):
                    x_valid_batch = x_valid[i:i+batch_size]
                    x_valid_poly = poly.transform(x_valid_batch)
                    x_valid_scaled = scaler.transform(x_valid_poly)
                    valid_predictions.extend(model.predict(x_valid_scaled))

                y_pred = np.array(valid_predictions)
                y_pred = np.maximum(y_pred, 0)
                rmse_iter.append(np.sqrt(mean_squared_error(y_valid, y_pred)))

            rmse_training.append(rmse_iter)

        rmse_training = np.array(rmse_training)
        mean_rmse = rmse_training.mean(axis=0)
        std_rmse = rmse_training.std(axis=0)

        valid_mask = mean_rmse < 400
        if valid_mask.any():
            valid_mean_rmse = mean_rmse[valid_mask]
            valid_params = regularization_params[valid_mask]
            best_idx = np.argmin(valid_mean_rmse)
            best_alpha = valid_params[best_idx]
            best_rmse = valid_mean_rmse[best_idx]
            best_std = std_rmse[valid_mask][best_idx]
        else:
            best_idx = np.argmin(mean_rmse)
            best_alpha = regularization_params[best_idx]
            best_rmse = mean_rmse[best_idx]
            best_std = std_rmse[best_idx]

        self.best_params = {
            'alpha': float(best_alpha),
            'degree': self.degree,
            'batch_size': batch_size
        }
        self.validation_rmse = float(best_rmse)

        return {
            'best_params': self.best_params,
            'validation_rmse': float(best_rmse),
            'validation_std': float(best_std),
            'all_rmse_mean': mean_rmse.tolist(),
            'all_rmse_std': std_rmse.tolist()
        }

    def train(self, x_train, y_train, alpha=None, batch_size=64):
        """Train the model with best hyperparameters.

        Args:
            x_train: Training features
            y_train: Training labels
            alpha: Regularization parameter
            batch_size: Batch size for SGD
        """
        if alpha is None and self.best_params is not None:
            alpha = self.best_params['alpha']
        elif alpha is None:
            raise ValueError("alpha must be provided or hyperparameters must be searched first")

        self.model = SGDRegressor(
            max_iter=1000,
            learning_rate='adaptive',
            eta0=0.001,
            alpha=alpha,
            random_state=42
        )

        self.poly = PolynomialFeatures(degree=self.degree)
        self.scaler = StandardScaler()

        n_batches = len(x_train) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            if len(x_batch) == 0:
                continue

            x_batch_poly = self.poly.fit_transform(x_batch) if i == 0 else self.poly.transform(x_batch)
            x_batch_scaled = self.scaler.fit_transform(x_batch_poly) if i == 0 else self.scaler.transform(x_batch_poly)

            self.model.partial_fit(x_batch_scaled, y_batch)

    def predict(self, x):
        """Make predictions on input data.

        Args:
            x: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        batch_size = 64
        predictions = []

        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            x_poly = self.poly.transform(x_batch)
            x_scaled = self.scaler.transform(x_poly)
            predictions.extend(self.model.predict(x_scaled))

        return np.array(predictions)
