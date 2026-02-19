import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file: str, input_length: int = 1, horizon: int = 10):
        """
        Loads the robot dataset and prepares sequences for both feedforward and recurrent models.
        
        Args:
            csv_file (str): Path to the processed dataset CSV.
            input_length (int): Number of steps to feed as input (1 for MLPs, >1 for RNN warmup).
            horizon (int): Number of future steps to unroll and predict.
        """
        self.input_length = input_length
        self.horizon = horizon
        df = pd.read_csv(csv_file)
        
        # State Vector s_k in Global Frame (F_G)
        # [x, y, theta, vx, vy, omega]
        self.states = torch.tensor(
            df[['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']].values, 
            dtype=torch.float32
        )
        
        # Command Vector u_k in Local Frame (F_L)
        # [vx_cmd, vy_cmd, omega_cmd]
        self.commands = torch.tensor(
            df[['vx_cmd', 'vy_cmd', 'omega_cmd']].values, 
            dtype=torch.float32
        )

    def __len__(self):
        # We need enough data to cover the input window + the future horizon
        return len(self.states) - (self.input_length + self.horizon) + 1

    def __getitem__(self, idx):
        """
        Extracts a continuous sequence covering both the input history and the future horizon.
        """
        total_states_needed = self.input_length + self.horizon
        
        # Extract states: from index to index + input_length + horizon
        seq_s = self.states[idx : idx + total_states_needed]
        
        # Extract commands: To predict the final state, we only need commands up to the 
        # second-to-last state. Therefore, we need (total_states_needed - 1) commands.
        seq_u = self.commands[idx : idx + total_states_needed - 1]
        
        return seq_s, seq_u


def get_trajectory_dataloaders(csv_file: str, input_length: int = 1, horizon: int = 10, batch_size: int = 32, train_split: float = 0.8):
    """
    Creates chronologically split DataLoaders.
    """
    dataset = TrajectoryDataset(csv_file, input_length=input_length, horizon=horizon)
    
    # Chronological split to prevent data leakage from the future into the past
    train_size = int(train_split * len(dataset))
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    # Shuffle is usually safe for the training set because the sequences internally maintain chronological order
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader