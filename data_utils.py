"""This file provides utility functions for loading and manipulating data from 
the NYU Depth V2 dataset"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split

def load_original_data(filepath = 'nyu_depth_v2_labeled.mat'):
    """Loads the RBG image and depth map data from the nyu depth v2 matlab file.

    Returns: Tuple of two matrices
        (N, 3, 640, 480): 1449-sized matrix of RGB images (N, C, W, H)
        (N, 640, 480): 1449-sized matrix of depth values (N, W, H)
    """
    f = h5py.File(filepath, 'r')

    images = np.array(f['images']) # (N, 3, 640, 480), (N, C, W, H)
    depths = np.array(f['depths']) # (N, 640, 480), (N, W, H)
    
    f.close()
    
    return images, depths

def transpose_data_to_pytorch(images, depths):
    """Transpose data to a PyTorch-friendly format

    Args:
        images (N, 3, 640, 480): Original image matrix
        depths (N, 640, 480): Original depth map matrix

    Returns: Tuple of two matrices
        (N, 3, 480, 640): 1449-sized matrix of RGB images (N, C, H, W)
        (N, 480, 640): 1449-sized matrix of depth values (N, H, W)
    """
    # (N, 3, 480, 640), (N, C, H, W)
    images_tranposed = np.transpose(images, (0, 1, 3, 2))   
    # (N, 480, 640), (N, H, W)
    depths_transposed = np.transpose(depths, (0, 2, 1))   
    
    return images_tranposed, depths_transposed

def downsample_data(images_tranposed, depths_tranposed, target_size = (30, 40)):
    """Downsample data after tranposing

    Args:
        images (N, 3, 480, 640): 1449-sized matrix of RGB images (N, C, H, W)
        depths (N, 480, 640): 1449-sized matrix of depth values (N, H, W)
        target_size (tuple, optional): Target size (3:4 aspect ratio).
            Defaults to (30, 40).

    Returns: Tuple of two matrices
        (N, 3, 30, 40): 1449-sized matrix of RGB images (N, C, H, W)
        (N, 30, 40): 1449-sized matrix of depth values (N, H, W)
    """
    
    images_downsampled = np.array([
        resize(img.transpose(1, 2, 0), target_size, anti_aliasing=True) 
        for img in images_tranposed
    ])
    images_downsampled = images_downsampled.transpose(0, 3, 1, 2) 
    
    depths_downsampled = np.array([
        resize(depth, target_size, anti_aliasing=True)
        for depth in depths_tranposed
    ])
    
    return images_downsampled, depths_downsampled

def save_data_to_npy(images_tp_ds, depths_tp_ds):
    np.save('images_resized.npy', images_tp_ds) # (N, C, H, W)
    np.save('depths_resized.npy', depths_tp_ds) # (N, H, W)
    
def load_data_from_npy():
    images_resized = np.load('images_resized.npy') # (N, C, H, W)
    depths_resized = np.load('depths_resized.npy') # (N, H, W)
    return images_resized, depths_resized

def prepare_data_for_training(images, depths, test_size=0.2, random_state=42):
    """Prepare data for training by flattening images and averaging depth maps.

    Args:
        images (N, 3, H, W): Image data
        depths (N, H, W): Depth map data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (x_use, x_test, y_use, y_test)
    """

    X = images.reshape(images.shape[0], -1)
    y_mean = depths.reshape(depths.shape[0], -1).mean(axis=1)

    x_use, x_test, y_use, y_test = train_test_split(
        X, y_mean, test_size=test_size, random_state=random_state, shuffle=True
    )

    return x_use, x_test, y_use, y_test

def save_split_data(x_use, x_test, y_use, y_test, data_dir='data'):
    """Save train/test split data to numpy files.

    Args:
        x_use: Training features
        x_test: Test features
        y_use: Training labels
        y_test: Test labels
        data_dir: Directory to save files (default: 'data')
    """
    os.makedirs(data_dir, exist_ok=True)

    np.save(f'{data_dir}/x_use.npy', x_use)
    np.save(f'{data_dir}/x_test.npy', x_test)
    np.save(f'{data_dir}/y_use.npy', y_use)
    np.save(f'{data_dir}/y_test.npy', y_test)

def load_split_data(data_dir='data'):
    """Load train/test split data from numpy files.

    Args:
        data_dir: Directory containing the data files (default: 'data')

    Returns:
        Tuple of (x_use, x_test, y_use, y_test)
    """
    x_use = np.load(f'{data_dir}/x_use.npy', allow_pickle=True)
    x_test = np.load(f'{data_dir}/x_test.npy', allow_pickle=True)
    y_use = np.load(f'{data_dir}/y_use.npy', allow_pickle=True)
    y_test = np.load(f'{data_dir}/y_test.npy', allow_pickle=True)

    return x_use, x_test, y_use, y_test

def plot_rand_images(images, depths):
    # Select 5 random images
    indices = np.random.choice(images.shape[0], 5, replace=False)

    fig, axes = plt.subplots(5, 2, figsize=(7, 15))

    for idx, img_idx in enumerate(indices):
        # RGB image - convert from (C, H, W) to (H, W, C)
        img = images[img_idx].transpose(1, 2, 0)
        axes[idx, 0].imshow(img.astype(np.uint8))
        axes[idx, 0].set_title(f'Image {img_idx}')
        axes[idx, 0].axis('off')

        # Depth map - already (H, W)
        depth = depths[img_idx]
        axes[idx, 1].imshow(depth, cmap='plasma')
        axes[idx, 1].set_title(f'Depth {img_idx}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images, depths = load_original_data()
    images_transposed, depths_transposed = transpose_data_to_pytorch(images, 
                                                                     depths)
    images_tp_ds, depths_tp_ds = downsample_data(images_transposed, 
                                                 depths_transposed)
    save_data_to_npy(images_tp_ds, depths_tp_ds)