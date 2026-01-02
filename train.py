import os
import json
import argparse
import numpy as np
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


def train_regression(x_use, y_use, x_test, y_test, config, output_dir, iterations=None):
    """Train Linear Regression model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        config: Configuration dictionary for regression
        output_dir: Directory to save model hyperparameters
        iterations: Number of iterations (overrides config if provided)

    Returns:
        Dictionary with validation and test RMSE
    """
    print("\n" + "="*50)
    print("Training Linear Regression (degree=1)")
    print("="*50)

    regression_linear = RegressionModel(degree=config['degree'])

    # Create regularization params from config
    reg_params = np.logspace(
        config['regularization_params']['start'],
        config['regularization_params']['stop'],
        config['regularization_params']['num'],
        base=config['regularization_params']['base']
    )

    # Use provided iterations or config iterations
    iter_count = iterations if iterations is not None else config['iterations']

    results = regression_linear.search_hyperparameters(
        x_use, y_use,
        regularization_params=reg_params,
        batch_size=config['batch_size'],
        iterations=iter_count
    )
    print(f"Best params: {regression_linear.best_params}")
    print(f"Validation RMSE: {regression_linear.validation_rmse:.4f}")

    regression_linear.train(x_use, y_use)
    test_rmse = regression_linear.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    regression_linear.save_hyperparameters(f'{output_dir}/regression_linear.json')

    return {
        'validation_rmse': regression_linear.validation_rmse,
        'test_rmse': test_rmse
    }


def train_svr(x_use, y_use, x_test, y_test, config, output_dir, iterations=None):
    """Train SVR model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        config: Configuration dictionary for SVR
        output_dir: Directory to save model hyperparameters
        iterations: Number of iterations (overrides config if provided)

    Returns:
        Dictionary with validation and test RMSE
    """
    print("\n" + "="*50)
    print("Training SVR")
    print("="*50)

    svr = SVRModel()

    # Create hyperparameter grids from config
    epsilons = np.logspace(
        config['epsilons']['start'],
        config['epsilons']['stop'],
        config['epsilons']['num'],
        base=config['epsilons']['base']
    )
    Cs = np.logspace(
        config['Cs']['start'],
        config['Cs']['stop'],
        config['Cs']['num'],
        base=config['Cs']['base']
    )
    gammas = np.logspace(
        config['gammas']['start'],
        config['gammas']['stop'],
        config['gammas']['num'],
        base=config['gammas']['base']
    )

    # Use provided iterations or config iterations
    iter_count = iterations if iterations is not None else config['iterations']

    results = svr.search_hyperparameters(
        x_use, y_use,
        epsilons=epsilons,
        Cs=Cs,
        gammas=gammas,
        iterations=iter_count
    )
    print(f"Best params: {svr.best_params}")
    print(f"Validation RMSE: {svr.validation_rmse:.4f}")

    svr.train(x_use, y_use)
    test_rmse = svr.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    svr.save_hyperparameters(f'{output_dir}/svr.json')

    return {
        'validation_rmse': svr.validation_rmse,
        'test_rmse': test_rmse
    }


def train_mlp(x_use, y_use, x_test, y_test, config, output_dir, iterations=None):
    """Train MLP model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        config: Configuration dictionary for MLP
        output_dir: Directory to save model hyperparameters
        iterations: Number of iterations (overrides config if provided)

    Returns:
        Dictionary with validation and test RMSE
    """
    print("\n" + "="*50)
    print("Training MLP")
    print("="*50)

    mlp = MLPModel()

    # Use provided iterations or config iterations
    iter_count = iterations if iterations is not None else config['iterations']

    results = mlp.search_hyperparameters(
        x_use, y_use,
        hidden_architectures=config['architectures'],
        weight_decays=config['weight_decays'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        iterations=iter_count,
        epochs=config['epochs'],
        dropout=config['dropout']
    )
    print(f"Best params: {mlp.best_params}")
    print(f"Validation RMSE: {mlp.validation_rmse:.4f}")

    mlp.train(x_use, y_use)
    test_rmse = mlp.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    mlp.save_hyperparameters(f'{output_dir}/mlp.json')

    return {
        'validation_rmse': mlp.validation_rmse,
        'test_rmse': test_rmse
    }


def train_cnn(x_use, y_use, x_test, y_test, config, output_dir, iterations=None):
    """Train CNN model.

    Args:
        x_use: Training features
        y_use: Training labels
        x_test: Test features
        y_test: Test labels
        config: Configuration dictionary for CNN
        output_dir: Directory to save model hyperparameters
        iterations: Number of iterations (overrides config if provided)

    Returns:
        Dictionary with validation and test RMSE
    """
    print("\n" + "="*50)
    print("Training CNN")
    print("="*50)

    cnn = CNNModel(
        input_channels=config['input_channels'],
        input_height=config['input_height'],
        input_width=config['input_width']
    )

    # Use provided iterations or config iterations
    iter_count = iterations if iterations is not None else config['iterations']

    results = cnn.search_hyperparameters(
        x_use, y_use,
        architectures=config['architectures'],
        weight_decays=config['weight_decays'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        iterations=iter_count,
        epochs=config['epochs'],
        dropout=config['dropout']
    )
    print(f"Best params: {cnn.best_params}")
    print(f"Validation RMSE: {cnn.validation_rmse:.4f}")

    cnn.train(x_use, y_use)
    test_rmse = cnn.evaluate(x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    cnn.save_hyperparameters(f'{output_dir}/cnn.json')

    return {
        'validation_rmse': cnn.validation_rmse,
        'test_rmse': test_rmse
    }


def main(config_path='config.json', quick_mode=False, models_to_train=None):
    """Main function to train all or selected models.

    Args:
        config_path: Path to configuration file
        quick_mode: If True, use 1 iteration for all models
        models_to_train: List of model names to train (None = all models)
    """
    # Load configuration
    config = load_config(config_path)

    data_dir = config['data']['data_dir']
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    x_use, x_test, y_use, y_test = load_split_data(data_dir)

    print(f"Training data shape: {x_use.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Set iterations based on quick mode
    iterations = 1 if quick_mode else None
    if quick_mode:
        print("\nQUICK MODE: Using 1 iteration for all models\n")

    # Define available models
    available_models = {
        'regression': lambda: train_regression(x_use, y_use, x_test, y_test,
                                              config['regression'], output_dir, iterations),
        'svr': lambda: train_svr(x_use, y_use, x_test, y_test,
                                config['svr'], output_dir, iterations),
        'mlp': lambda: train_mlp(x_use, y_use, x_test, y_test,
                                config['mlp'], output_dir, iterations),
        'cnn': lambda: train_cnn(x_use, y_use, x_test, y_test,
                                config['cnn'], output_dir, iterations)
    }

    # Determine which models to train
    if models_to_train is None:
        models_to_train = list(available_models.keys())

    # Train models
    models_results = {}
    for model_name in models_to_train:
        if model_name in available_models:
            models_results[model_name] = available_models[model_name]()
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")

    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    for model_name, results in models_results.items():
        print(f"{model_name:20s} | Val RMSE: {results['validation_rmse']:7.4f} | Test RMSE: {results['test_rmse']:7.4f}")

    print(f"\nHyperparameters saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train depth estimation models')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced hyperparameter search space for faster training')
    parser.add_argument('--models', nargs='+', choices=['regression', 'svr', 'mlp', 'cnn'],
                        help='Specific models to train (default: all models)')

    args = parser.parse_args()

    main(args.config, args.quick, args.models)
