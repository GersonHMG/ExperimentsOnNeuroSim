import torch
from utils.utils import State, Command
from torch import nn
import numpy as np

class KinematicModel(nn.Module):
    def __init__(self, vmax=5.0, wmax=5.0, kp_linear=0.8, kp_angular=1.0):
        super().__init__()
        self.vmax = vmax
        self.wmax = wmax
        
        # Register gains as buffers so they automatically move to the correct GPU/CPU device
        self.register_buffer('kp_linear', torch.tensor(kp_linear, dtype=torch.float32))
        self.register_buffer('kp_angular', torch.tensor(kp_angular, dtype=torch.float32))

    def forward(self, v_curr: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_curr: Current velocities (Batch, 3) -> [vx, vy, omega]
            v_target: Desired target velocities (Batch, 3) -> [vx_target, vy_target, omega_target]
        Returns:
            v_next: Physically achievable next velocities (Batch, 3)
        """
        # --- 1. Proportional Control Logic ---
        # Calculate the error between where we want to be and where we are
        error = v_target - v_curr
        
        # Apply the P-gain (Step = Kp * Error)
        p_step_linear = self.kp_linear * error[:, :2]
        p_step_angular = self.kp_angular * error[:, 2:3]
        
        # Calculate the requested velocity before limits are applied
        v_req_linear = v_curr[:, :2] + p_step_linear
        v_req_angular = v_curr[:, 2:3] + p_step_angular
        
        # --- 2. Kinematic Saturation ---
        # Calculate the magnitude of the requested linear velocity
        v_linear_norm = torch.norm(v_req_linear, dim=1, keepdim=True)
        
        # Find the scaling factor to keep it under vmax (preserves movement direction)
        scale = torch.clamp(self.vmax / (v_linear_norm + 1e-8), max=1.0)
        
        # Scale linear velocities proportionally
        v_next_linear = v_req_linear * scale
        
        # Clip angular velocity independently
        v_next_angular = torch.clamp(v_req_angular, min=-self.wmax, max=self.wmax)
        
        # --- 3. Recombine and Output ---
        return torch.cat([v_next_linear, v_next_angular], dim=1)


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
            nn.AdaptiveMaxPool1d(output_size=2) 
        )
        
        # Fully connected head
        # 64 channels * 4 temporal steps = 256 input features
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=3) 
        )

        self.cnn = nn.Sequential(
            self.feature_extractor,
            self.regression_head
        )

        self.kinematic = KinematicModel()

    
    def wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Normalizes to signed angle system (-pi, pi]."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def to_local(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Creates a rotation matrix to transform from Global to Local frame.
        Args:
            theta (torch.Tensor): The heading angles. Shape: (Batch, Window)
        Returns: 
            torch.Tensor: Rotation matrix of shape (Batch, Window, 3, 3)
        """
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # R(-theta) matrix for Global -> Local
        row1 = torch.stack([c, s, zeros], dim=-1)
        row2 = torch.stack([-s, c, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)
        
        return torch.stack([row1, row2, row3], dim=-2)
    
    def to_global(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Creates a rotation matrix to transform from Local to Global frame.
        Args:
            theta (torch.Tensor): The heading angles. Shape: (Batch, Window)
        Returns: 
            torch.Tensor: Rotation matrix of shape (Batch, Window, 3, 3)
        """
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # R(theta) matrix for Local -> Global
        row1 = torch.stack([c, -s, zeros], dim=-1)
        row2 = torch.stack([s, c, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)
        
        return torch.stack([row1, row2, row3], dim=-2)

    def forward(
        self,
        state_tensors: torch.Tensor, 
        cmd_tensors: torch.Tensor,
    ) -> State:
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
        theta_tensors = state_tensors[..., 2]

        R_global = self.to_global(theta_tensors)
        cmd_global_tensors = torch.matmul(R_global, cmd_tensors.unsqueeze(-1)).squeeze(-1)

        x = torch.cat([velocity_tensors, cmd_global_tensors], dim=-1)
        # (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        res_vx = out[:, 0]
        res_vy = out[:, 1]
        res_omega = out[:, 2]

        last_state_tensor = state_tensors[:, -1, :] 
        last_cmd_tensor = cmd_tensors[:, -1, :]

        # Kinematic model
        s_k = State.from_tensor(last_state_tensor)
        #v_curr_global = last_state_tensor[:, 3:6]
        #R_local = self.to_local(s_k.theta)
        #v_curr_local = torch.matmul(R_local, v_curr_global.unsqueeze(-1)).squeeze(-1)
        #v_kinematic_local = self.kinematic(v_curr_local, last_cmd_tensor)
        #R_global = self.to_global(s_k.theta)
        #v_kinematic_global = torch.matmul(R_global, v_kinematic_local.unsqueeze(-1)).squeeze(-1)
        #kin_vx = v_kinematic_global[:, 0]
        #kin_vy = v_kinematic_global[:, 1]
        #kin_omega = v_kinematic_global[:, 2]

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
