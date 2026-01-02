import numpy as np
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class SVRModel(BaseModel):
    def __init__(self):
        super().__init__("SVR")
        self.scaler = StandardScaler()

    def search_hyperparameters(self, x_use, y_use, epsilons=None, Cs=None,
                               gammas=None, iterations=10):
        """Search for best hyperparameters using grid search.

        Args:
            x_use: Training features
            y_use: Training labels
            epsilons: Array of epsilon values to try
            Cs: Array of C values to try
            gammas: Array of gamma values to try
            iterations: Number of random train/val splits

        Returns:
            Dictionary with best hyperparameters and validation metrics
        """
        if epsilons is None:
            epsilons = np.logspace(-3, 2, 6)
        if Cs is None:
            Cs = np.logspace(-3, 5, 10)
        if gammas is None:
            gammas = np.logspace(-6, 1, 10)

        rmse_all_iterations = []

        for i in tqdm(range(iterations), desc=f"{self.model_name} Iterations"):
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_use, y_use, test_size=0.2, random_state=i
            )

            rmse_E = []
            for epsilon in tqdm(epsilons, desc="Epsilon", leave=False):
                rmse_C = []
                for c in tqdm(Cs, desc="C", leave=False):
                    rmse_G = []
                    for gamma in tqdm(gammas, desc="Gamma", leave=False):
                        svr = SVR(kernel='rbf', gamma=gamma, C=c, epsilon=epsilon)
                        svr.fit(x_train, y_train)
                        y_pred = svr.predict(x_valid)
                        y_pred = np.maximum(y_pred, 0)
                        rmse_G.append(np.sqrt(mean_squared_error(y_valid, y_pred)))

                    rmse_C.append(rmse_G)

                rmse_E.append(rmse_C)

            rmse_all_iterations.append(np.array(rmse_E))

        rmse_all_iterations = np.array(rmse_all_iterations)
        rmse = np.mean(rmse_all_iterations, axis=0)
        rmse_std = np.std(rmse_all_iterations, axis=0)

        best_global = np.unravel_index(np.argmin(rmse), rmse.shape)
        best_epsilon = epsilons[best_global[0]]
        best_C = Cs[best_global[1]]
        best_gamma = gammas[best_global[2]]
        best_rmse = rmse[best_global]
        best_std = rmse_std[best_global]

        self.best_params = {
            'epsilon': float(best_epsilon),
            'C': float(best_C),
            'gamma': float(best_gamma),
            'kernel': 'rbf'
        }
        self.validation_rmse = float(best_rmse)

        return {
            'best_params': self.best_params,
            'validation_rmse': float(best_rmse),
            'validation_std': float(best_std)
        }

    def train(self, x_train, y_train, epsilon=None, C=None, gamma=None):
        """Train the model with best hyperparameters.

        Args:
            x_train: Training features
            y_train: Training labels
            epsilon: Epsilon parameter for SVR
            C: C parameter for SVR
            gamma: Gamma parameter for SVR
        """
        if epsilon is None and self.best_params is not None:
            epsilon = self.best_params['epsilon']
            C = self.best_params['C']
            gamma = self.best_params['gamma']
        elif epsilon is None or C is None or gamma is None:
            raise ValueError("Hyperparameters must be provided or searched first")

        self.model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
        self.model.fit(x_train, y_train)

    def predict(self, x):
        """Make predictions on input data.

        Args:
            x: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(x)
