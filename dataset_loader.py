import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from utils.utils import State, Command

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_length: int, target_length: int):
        """
        Args:
            df: The trajectory dataframe.
            window_length: How many historical steps to use as input.
            target_length: How many future steps to predict (the autoregressive horizon).
        """
        self.window_length = window_length
        self.target_length = target_length
        
        # The total sequence length needed for one sample
        self.total_seq_len = window_length + target_length
        
        # Adjust total samples so we don't index out of bounds at the end of the dataframe
        self.num_samples = len(df) - self.total_seq_len + 1
        
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
        """
        Retrieves a historical sequence and a future target sequence.
        
        Returns:
            input_states: Historical states shape (window_length,)
            input_cmds: Historical commands shape (window_length,)
            target_states: Future ground-truth states shape (target_length,)
            future_cmds: Future planned commands shape (target_length,)
        """
        # Calculate the split indices
        mid_idx = idx + self.window_length
        end_idx = mid_idx + self.target_length
        
        # Slice the historical inputs
        raw_input_state = self.state_data[idx:mid_idx]
        raw_input_cmd = self.cmd_data[idx:mid_idx]
        
        # Slice the future targets (and the future commands driving them)
        raw_target_state = self.state_data[mid_idx:end_idx]
        raw_future_cmd = self.cmd_data[mid_idx:end_idx]
        
        return (
            State.from_tensor(raw_input_state),
            Command.from_tensor(raw_input_cmd),
            State.from_tensor(raw_target_state),
            Command.from_tensor(raw_future_cmd)
        )