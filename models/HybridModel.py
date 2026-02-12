import torch
import torch.nn as nn
from .PhysicsModel import OmniRobotDynamic

class HybridModel(nn.Module):
    def __init__(self, dt=0.016, hidden_size=64):
        super().__init__()
        
        # 1. Physics-Based Component (Differential Equation)
        self.physics = OmniRobotDynamic(dt=dt)
        
        # 2. Data-Driven Component (Residual Network)
        # Input: State (4) + Command (3) = 7
        # Output: Residual Optimizations for Velocities (3: vx, vy, omega)
        self.residual_net = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3) # Corrections for vx, vy, omega
        )
        
        # Initialize residuals small so training starts from physics baseline
        with torch.no_grad():
            self.residual_net[-1].weight.fill_(0.0)
            self.residual_net[-1].bias.fill_(0.0)

    def forward(self, current_state, commands):
        # --- Physics Prediction ---
        # next_state_phys: [vx, vy, omega, theta]
        # shape: (Batch, 4)
        next_state_phys = self.physics(current_state, commands)
        
        # --- Residual Prediction ---
        # error correction based on current state and command contexts
        # (e.g. at high speeds, physics model might under-predict friction)
        inputs = torch.cat([current_state, commands], dim=1)
        velocity_residuals = self.residual_net(inputs)
        
        # --- Combine ---
        # Add residuals to the physics prediction
        # We correct the rates/velocities (vx, vy, omega)
        # Indices 0, 1, 2 corresponding to vx, vy, omega
        
        pred_vx    = next_state_phys[:, 0] + velocity_residuals[:, 0]
        pred_vy    = next_state_phys[:, 1] + velocity_residuals[:, 1]
        pred_omega = next_state_phys[:, 2] + velocity_residuals[:, 2]
        
        # Update Theta consistently with the new Omega
        # Physics theta = old_theta + old_omega * dt
        # New theta     = old_theta + (old_omega + res_omega) * dt
        #               = Physics theta + res_omega * dt
        pred_theta = next_state_phys[:, 3] + velocity_residuals[:, 2] * self.physics.dt
        
        # Normalize theta to [-pi, pi]
        pred_theta = torch.atan2(torch.sin(pred_theta), torch.cos(pred_theta))
        
        return torch.stack([pred_vx, pred_vy, pred_omega, pred_theta], dim=1)
