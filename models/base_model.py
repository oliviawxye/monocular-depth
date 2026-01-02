import json
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.best_params = None
        self.model = None
        self.validation_rmse = None
        self.test_rmse = None

    @abstractmethod
    def search_hyperparameters(self, x_use, y_use, **kwargs):
        """Search for best hyperparameters using validation data.

        Args:
            x_use: Training features
            y_use: Training labels
            **kwargs: Additional parameters for hyperparameter search

        Returns:
            Dictionary with best hyperparameters and validation metrics
        """
        pass

    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        """Train the model with given hyperparameters.

        Args:
            x_train: Training features
            y_train: Training labels
            **kwargs: Model hyperparameters
        """
        pass

    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data.

        Args:
            x_test: Test features
            y_test: Test labels

        Returns:
            RMSE on test set
        """
        y_pred = self.predict(x_test)
        y_pred = np.maximum(y_pred, 0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        self.test_rmse = rmse
        return rmse

    @abstractmethod
    def predict(self, x):
        """Make predictions on input data.

        Args:
            x: Input features

        Returns:
            Predictions
        """
        pass

    def save_hyperparameters(self, filepath):
        """Save hyperparameters to JSON file.

        Args:
            filepath: Path to save hyperparameters
        """
        params_dict = {
            'model_name': self.model_name,
            'best_params': self.best_params,
            'validation_rmse': float(self.validation_rmse) if self.validation_rmse is not None else None,
            'test_rmse': float(self.test_rmse) if self.test_rmse is not None else None
        }

        with open(filepath, 'w') as f:
            json.dump(params_dict, f, indent=4)

    def load_hyperparameters(self, filepath):
        """Load hyperparameters from JSON file.

        Args:
            filepath: Path to load hyperparameters from
        """
        with open(filepath, 'r') as f:
            params_dict = json.load(f)

        self.model_name = params_dict['model_name']
        self.best_params = params_dict['best_params']
        self.validation_rmse = params_dict.get('validation_rmse')
        self.test_rmse = params_dict.get('test_rmse')

        return self.best_params
