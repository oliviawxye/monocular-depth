import os
import json
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from data_utils import load_split_data
from models import RegressionModel, SVRModel, MLPModel, CNNModel


def load_config(config_path='config.json'):
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def evaluate_regression(x_use, y_use, x_test, y_test, models_dir):
    """Evaluate Linear Regression model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        models_dir: Directory containing saved hyperparameters

    Returns:
        Dictionary with test RMSE and predictions
    """
    print("\n" + "="*50)
    print("Evaluating Linear Regression")
    print("="*50)

    if not os.path.exists(f'{models_dir}/regression_linear.json'):
        print("No saved hyperparameters found. Skipping.")
        return None

    regression = RegressionModel(degree=1)
    params = regression.load_hyperparameters(f'{models_dir}/regression_linear.json')
    print(f"Loaded hyperparameters: {params}")

    regression.train(x_use, y_use, alpha=params['alpha'])
    test_rmse = regression.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    y_pred = regression.predict(x_test)
    y_pred = np.maximum(y_pred, 0)

    return {
        'test_rmse': float(test_rmse),
        'predictions': y_pred
    }


def evaluate_svr(x_use, y_use, x_test, y_test, models_dir):
    """Evaluate SVR model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        models_dir: Directory containing saved hyperparameters

    Returns:
        Dictionary with test RMSE and predictions
    """
    print("\n" + "="*50)
    print("Evaluating SVR")
    print("="*50)

    if not os.path.exists(f'{models_dir}/svr.json'):
        print("No saved hyperparameters found. Skipping.")
        return None

    svr = SVRModel()
    params = svr.load_hyperparameters(f'{models_dir}/svr.json')
    print(f"Loaded hyperparameters: {params}")

    svr.train(x_use, y_use, epsilon=params['epsilon'], C=params['C'], gamma=params['gamma'])
    test_rmse = svr.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    y_pred = svr.predict(x_test)
    y_pred = np.maximum(y_pred, 0)

    return {
        'test_rmse': float(test_rmse),
        'predictions': y_pred
    }


def evaluate_mlp(x_use, y_use, x_test, y_test, models_dir):
    """Evaluate MLP model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        models_dir: Directory containing saved hyperparameters

    Returns:
        Dictionary with test RMSE and predictions
    """
    print("\n" + "="*50)
    print("Evaluating MLP")
    print("="*50)

    if not os.path.exists(f'{models_dir}/mlp.json'):
        print("No saved hyperparameters found. Skipping.")
        return None

    mlp = MLPModel()
    params = mlp.load_hyperparameters(f'{models_dir}/mlp.json')
    print(f"Loaded hyperparameters: {params}")

    mlp.train(
        x_use, y_use,
        hidden_sizes=params['hidden_sizes'],
        weight_decay=params['weight_decay'],
        lr=params['lr'],
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        dropout=params['dropout']
    )
    test_rmse = mlp.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    y_pred = mlp.predict(x_test)
    y_pred = np.maximum(y_pred, 0)

    return {
        'test_rmse': float(test_rmse),
        'predictions': y_pred
    }


def evaluate_cnn(x_use, y_use, x_test, y_test, models_dir):
    """Evaluate CNN model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        models_dir: Directory containing saved hyperparameters

    Returns:
        Dictionary with test RMSE and predictions
    """
    print("\n" + "="*50)
    print("Evaluating CNN")
    print("="*50)

    if not os.path.exists(f'{models_dir}/cnn.json'):
        print("No saved hyperparameters found. Skipping.")
        return None

    cnn = CNNModel()
    params = cnn.load_hyperparameters(f'{models_dir}/cnn.json')
    print(f"Loaded hyperparameters: {params}")

    cnn.train(
        x_use, y_use,
        conv_channels=params['conv_channels'],
        kernel_sizes=params['kernel_sizes'],
        pool_sizes=params['pool_sizes'],
        fc_hidden_sizes=params['fc_hidden_sizes'],
        weight_decay=params['weight_decay'],
        lr=params['lr'],
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        dropout=params['dropout']
    )
    test_rmse = cnn.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    y_pred = cnn.predict(x_test)
    y_pred = np.maximum(y_pred, 0)

    return {
        'test_rmse': float(test_rmse),
        'predictions': y_pred
    }


def main(config_path='config.json', models_to_evaluate=None):
    """Main function to evaluate all or selected models.

    Args:
        config_path: Path to configuration file
        models_to_evaluate: List of model names to evaluate (None = all models)
    """
    # Load configuration
    config = load_config(config_path)

    data_dir = config['data']['data_dir']
    models_dir = config['data']['output_dir']

    # Load data
    print("Loading data...")
    x_use, x_test, y_use, y_test = load_split_data(data_dir)

    print(f"Training data shape: {x_use.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test set size: {len(y_test)}")

    # Define available models
    available_models = {
        'regression': lambda: evaluate_regression(x_use, y_use, x_test, y_test, models_dir),
        'svr': lambda: evaluate_svr(x_use, y_use, x_test, y_test, models_dir),
        'mlp': lambda: evaluate_mlp(x_use, y_use, x_test, y_test, models_dir),
        'cnn': lambda: evaluate_cnn(x_use, y_use, x_test, y_test, models_dir)
    }

    # Determine which models to evaluate
    if models_to_evaluate is None:
        models_to_evaluate = list(available_models.keys())

    # Evaluate models
    results = {}
    for model_name in models_to_evaluate:
        if model_name in available_models:
            result = available_models[model_name]()
            if result is not None:
                results[model_name] = result
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")

    if not results:
        print("\nNo models were evaluated.")
        return

    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"{'Model':<20s} | {'Test RMSE':<10s}")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name:<20s} | {result['test_rmse']:10.4f}")

    best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
    print(f"\nBest model: {best_model[0]} with RMSE: {best_model[1]['test_rmse']:.4f}")

    # Print prediction statistics
    print("\n" + "="*50)
    print("Prediction Statistics")
    print("="*50)
    for model_name, result in results.items():
        y_pred = result['predictions']

        zero_predictions = (y_pred == 0).sum() / len(y_pred) * 100
        large_errors = (np.abs(y_test - y_pred) > 1.0).sum() / len(y_pred) * 100
        max_error = np.abs(y_test - y_pred).max()
        errors = y_test - y_pred

        print(f"\n{model_name}:")
        print(f"  {zero_predictions:.1f}% of predictions are 0.0")
        print(f"  {large_errors:.1f}% of predictions have error > 1 meter")
        print(f"  Worst prediction off by: {max_error:.2f} meters")
        print(f"  Error range: [{errors.min():.2f}, {errors.max():.2f}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate depth estimation models')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--models', nargs='+', choices=['regression', 'svr', 'mlp', 'cnn'],
                        help='Specific models to evaluate (default: all models)')

    args = parser.parse_args()

    main(args.config, args.models)
