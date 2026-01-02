# Depth Estimation ML Pipeline

A structured machine learning pipeline for depth estimation using multiple models: Linear Regression, Support Vector Regression (SVR), Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN).

## Project Structure

```
syde522-final/
├── config.json            # Hyperparameter configuration file
├── data_utils.py          # Data loading and preprocessing utilities
├── models/                # Model classes
│   ├── __init__.py
│   ├── base_model.py      # Abstract base class
│   ├── regression_model.py
│   ├── svr_model.py
│   ├── mlp_model.py
│   └── cnn_model.py
├── train.py               # Main training script
├── evaluate.py            # Main evaluation script
├── data/                  # Training/test data (created by data_utils.py)
│   ├── x_use.npy
│   ├── x_test.npy
│   ├── y_use.npy
│   └── y_test.npy
└── trained_models/        # Saved hyperparameters (created by train.py)
    ├── regression_linear.json
    ├── svr.json
    ├── mlp.json
    └── cnn.json
```

## Quick Start

### 1. Training All Models

```bash
# Full training with all hyperparameter searches
python train.py

# Quick mode (reduced search space for testing)
python train.py --quick

# Train specific models only
python train.py --models mlp cnn

# Use custom config file
python train.py --config my_config.json
```

### 2. Evaluating Models

```bash
# Evaluate all trained models
python evaluate.py

# Evaluate specific models only
python evaluate.py --models mlp cnn

# Use custom config file
python evaluate.py --config my_config.json
```

## Configuration

All hyperparameters are stored in `config.json`. You can easily modify this file to experiment with different settings **without changing the training code**.

### Configuration Structure

```json
{
    "data": {
        "data_dir": "data",
        "output_dir": "trained_models"
    },
    "regression": {
        "degree": 1,
        "regularization_params": {...},
        "batch_size": 64,
        "iterations": 10
    },
    "svr": {
        "epsilons": {...},
        "Cs": {...},
        "gammas": {...},
        "iterations": 10
    },
    "mlp": {
        "architectures": [[8], [16], ...],
        "lr": 0.001,
        "batch_size": 64,
        "iterations": 10,
        "epochs": 10,
        "dropout": 0.2
    },
    "cnn": {
        "architectures": [...],
        "lr": 0.001,
        "batch_size": 64,
        "iterations": 1,
        "epochs": 20,
        "dropout": 0.2,
        "input_channels": 3,
        "input_height": 30,
        "input_width": 40
    }
}
```

### Modifying Hyperparameters

To change hyperparameters, simply edit `config.json`:

```json
{
    "mlp": {
        "architectures": [
            [64, 64],        // Add a new architecture
            [128, 64, 32]    // Try deeper networks
        ],
        "epochs": 20,        // Increase training epochs
        "lr": 0.0001         // Try lower learning rate
    }
}
```

## Model Classes

Each model inherits from `BaseModel` and provides a consistent interface:

### Common Methods

- `search_hyperparameters(x_use, y_use, **kwargs)` - Perform hyperparameter search
- `train(x_train, y_train, **kwargs)` - Train the model
- `predict(x)` - Make predictions
- `evaluate(x_test, y_test)` - Evaluate on test set
- `save_hyperparameters(filepath)` - Save to JSON
- `load_hyperparameters(filepath)` - Load from JSON

### Example Usage

```python
from models import MLPModel
from data_utils import load_split_data

# Load data
x_use, x_test, y_use, y_test = load_split_data('data')

# Create and train model
mlp = MLPModel()
mlp.search_hyperparameters(x_use, y_use,
                           hidden_architectures=[[16, 16], [32, 32]],
                           epochs=10)
mlp.train(x_use, y_use)

# Evaluate
test_rmse = mlp.evaluate(x_test, y_test)
print(f"Test RMSE: {test_rmse:.4f}")

# Save hyperparameters
mlp.save_hyperparameters('trained_models/mlp.json')
```

## Training Functions

The training script is organized into separate functions for each model:

- `train_regression(x_use, y_use, x_test, y_test, config, output_dir)`
- `train_svr(x_use, y_use, x_test, y_test, config, output_dir)`
- `train_mlp(x_use, y_use, x_test, y_test, config, output_dir)`
- `train_cnn(x_use, y_use, x_test, y_test, config, output_dir)`

All functions are called from the `main()` function, which orchestrates the entire training pipeline.

## Evaluation Functions

Similarly, evaluation is organized into separate functions:

- `evaluate_regression(x_use, y_use, x_test, y_test, models_dir)`
- `evaluate_svr(x_use, y_use, x_test, y_test, models_dir)`
- `evaluate_mlp(x_use, y_use, x_test, y_test, models_dir)`
- `evaluate_cnn(x_use, y_use, x_test, y_test, models_dir)`

## Data Utilities

The `data_utils.py` module provides functions for data handling:

- `load_original_data(filepath)` - Load raw NYU Depth V2 data
- `transpose_data_to_pytorch(images, depths)` - Transpose to PyTorch format
- `downsample_data(images, depths, target_size)` - Downsample images
- `prepare_data_for_training(images, depths)` - Flatten and split data
- `save_split_data(x_use, x_test, y_use, y_test)` - Save to numpy files
- `load_split_data(data_dir)` - Load from numpy files

## Saved Hyperparameters Format

Hyperparameters are saved in JSON format:

```json
{
    "model_name": "MLP",
    "best_params": {
        "hidden_sizes": [16, 16],
        "weight_decay": 0,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "dropout": 0.2
    },
    "validation_rmse": 0.8234,
    "test_rmse": 0.8156
}
```

## Command Line Arguments

### train.py

- `--config`: Path to configuration file (default: `config.json`)
- `--quick`: Use quick mode for faster training (original configs with 1 iteration only)
- `--models`: Specific models to train (choices: `regression`, `svr`, `mlp`, `cnn`)

### evaluate.py

- `--config`: Path to configuration file (default: `config.json`)
- `--models`: Specific models to evaluate (choices: `regression`, `svr`, `mlp`, `cnn`)

## Examples

```bash
# Train only MLP and CNN in quick mode
python train.py --quick --models mlp cnn

# Evaluate only the best performing model
python evaluate.py --models cnn

# Use a custom configuration for experimentation
python train.py --config experiments/config_experiment1.json

# Full training pipeline
python train.py && python evaluate.py
```

Refactor from iPynb to classic ML pipeline supported (but not entirely done) by Claude Code