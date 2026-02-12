import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RobotTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for robot trajectory sequences sliding window over a DataFrame.
    """
    
    def __init__(self, df, input_steps=20, target_steps=10):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing the trajectory data (columns: vx, vy, omega, etc.)
            input_steps (int): Number of input steps
            target_steps (int): Number of target steps
        """
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.sequence_length = input_steps + target_steps
        
        # Helper to get numpy array from column if exists, preferably filtered/estimated
        def get_col(names):
            for name in names:
                if name in df.columns:
                    return df[name].values.astype(np.float32)
            raise ValueError(f"Could not find any of {names} in dataframe columns: {df.columns}")

        # Extract columns as numpy arrays for fast access
        self.vx = get_col(['vx'])
        self.vy = get_col(['vy'])
        self.omega = get_col(['omega'])
        
        self.vx_cmd = get_col(['vx_cmd'])
        self.vy_cmd = get_col(['vy_cmd'])
        self.omega_cmd = get_col(['omega_cmd'])
        
        # Only keep theta, remove x and y
        self.theta = get_col(['filtered_theta', 'theta', 'Orientation'])
        
        self.data_len = len(df)
        
        # Calculate number of possible sequences
        # We slide the window by 1 step
        self.num_sequences = max(0, self.data_len - self.sequence_length + 1)
    
    def __len__(self):
        """Returns the total number of sequences."""
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Returns a single sample as tensors.
        
        Args:
            idx (int): Index of the sample (start index of the sliding window)
            
        Returns:
            dict: Dictionary containing:
                - 'input_x': Tensor of shape (input_steps, 7) - [vx, vy, omega, vx_cmd, vy_cmd, omega_cmd, theta]
                - 'target_x': Tensor of shape (target_steps, 7) - Same format as input
                - 'sequence_idx': Index of the sample
        """
        # Calculate indices
        start_idx = idx
        mid_idx = idx + self.input_steps
        end_idx = idx + self.sequence_length
        
        # Stack features for the entire window first or slice individually
        # Slicing individually and then stacking is usually fine
        
        # Define feature list for easy stacking
        # Features: [vx, vy, omega, vx_cmd, vy_cmd, omega_cmd, theta]
        
        # Input part
        input_x = np.stack([
            self.vx[start_idx:mid_idx],
            self.vy[start_idx:mid_idx],
            self.omega[start_idx:mid_idx],
            self.theta[start_idx:mid_idx],
            self.vx_cmd[start_idx:mid_idx],
            self.vy_cmd[start_idx:mid_idx],
            self.omega_cmd[start_idx:mid_idx],
        ], axis=1) # (input_steps, 7)
        
        # Target part
        target_x = np.stack([
            self.vx[mid_idx:end_idx],
            self.vy[mid_idx:end_idx],
            self.omega[mid_idx:end_idx],
            self.theta[mid_idx:end_idx],
            self.vx_cmd[mid_idx:end_idx],
            self.vy_cmd[mid_idx:end_idx],
            self.omega_cmd[mid_idx:end_idx],
        ], axis=1) # (target_steps, 7)
        
        return {
            'input_x': torch.from_numpy(input_x),
            'target_x': torch.from_numpy(target_x),
            'sequence_idx': idx
        }