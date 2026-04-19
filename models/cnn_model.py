import torch
from utils.utils import State, Command
from typing import Tuple
from torch import nn

class CNNModel(nn.Module):

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

        # Total input channels = 3 (state) + 3 (cmd) = 9
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=4) 
        )
        
        # Fully connected head
        # 64 channels * 4 temporal steps = 256 input features
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3) 
        )

        self.cnn = nn.Sequential(
            self.feature_extractor,
            self.regression_head
        )

    
    def wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Normalizes to signed angle system (-pi, pi]."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))


    def forward(
        self,
        state_tensors: torch.Tensor, 
        cmd_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict the next state based on historical states and commands.
        Args:
            state_tensors (torch.Tensor): Historical state sequence.
                Shape: (batch_size, window_length, 6) 
                Features: (x, y, theta, vx, vy, omega)
            cmd_tensors (torch.Tensor): Historical command sequence.
                Shape: (batch_size, window_length, 3) 
                Features: (vx_cmd, vy_cmd, omega_cmd)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_state_tensor (torch.Tensor): The predicted next state. 
                  Shape: (batch_size, 6)
        """
        velocity_tensors = state_tensors[..., 3:6]
        x = torch.cat([velocity_tensors, cmd_tensors], dim=-1)
        # (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        res_vx = out[:, 0]
        res_vy = out[:, 1]
        res_omega = out[:, 2]

        # Extract the last state
        last_state_tensor = state_tensors[:, -1, :] 
        last_cmd_tensor = cmd_tensors[:, -1, :]
        s_k = State.from_tensor(last_state_tensor)
        u_k = Command.from_tensor(last_cmd_tensor)

        vx = s_k.vx + res_vx
        vy = s_k.vy + res_vy
        omega = s_k.omega + res_omega

        x = s_k.x + vx*self.dt
        y = s_k.y + vy*self.dt
        theta = self.wrap_angle(s_k.theta + res_omega*self.dt)

        next_state = State(
            x     = x,
            y     = y,
            theta = theta,
            vx    = vx,
            vy    = vy,
            omega = omega,
        )
        return next_state
