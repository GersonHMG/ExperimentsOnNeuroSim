import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from utils.utils import State, Command

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_length: int, target_length: int, stride: int = 1):
        """
        Args:
            df: The trajectory dataframe.
            window_length: How many historical steps to use as input.
            target_length: How many future steps to predict.
            stride: How many rows to skip between each sample sequence.
        """
        self.window_length = window_length
        self.target_length = target_length
        self.stride = stride
        
        self.total_seq_len = window_length + target_length
        
        # Adjust total samples calculation to account for the stride jump
        self.num_samples = (len(df) - self.total_seq_len) // self.stride + 1
        
        if self.num_samples <= 0:
            raise ValueError(
                f"DataFrame length ({len(df)}) must be >= "
                f"window_length + target_length ({self.total_seq_len})."
            )

        state_cols = ['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']
        cmd_cols = ['vx_cmd', 'vy_cmd', 'omega_cmd']
        
        self.state_data = torch.tensor(df[state_cols].values, dtype=torch.float32)
        self.cmd_data = torch.tensor(df[cmd_cols].values, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[State, Command, State, Command]:
        
        # Calculate the actual starting row in the dataframe based on the stride
        start_idx = idx * self.stride
        
        mid_idx = start_idx + self.window_length
        end_idx = mid_idx + self.target_length
        
        # Slice using the strided indices
        raw_input_state = self.state_data[start_idx:mid_idx]
        raw_input_cmd = self.cmd_data[start_idx:mid_idx]
        
        raw_target_state = self.state_data[mid_idx:end_idx]
        raw_future_cmd = self.cmd_data[mid_idx:end_idx]
        
        return (
            State.from_tensor(raw_input_state),
            Command.from_tensor(raw_input_cmd),
            State.from_tensor(raw_target_state),
            Command.from_tensor(raw_future_cmd)
        )