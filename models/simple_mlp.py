import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command

class SimpleMLP(DynamicsBase):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # The input is 6D: 3 velocities (vx, vy, omega) + 3 commands (vx_cmd, vy_cmd, omega_cmd)
        # The output is 3D: the change (delta) for the 3 velocity variables only
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Output is now 3D
        )

    def compute_delta(self, state: State, command: Command) -> State:
        """
        Calculates the change in state. Velocities are converted to the local frame 
        for the neural network, then rotated back to the global frame.
        Positions are calculated via pure kinematics.
        """
        # 1. Extract tensors
        s_tensor = state.as_tensor()
        u_tensor = command.as_tensor()
        
        # 2. Extract heading (theta) and global velocities
        # Keeping slicing as `2:3` preserves the dimension for batch broadcasting
        theta = s_tensor[..., 2:3]
        vx_global = s_tensor[..., 3:4]
        vy_global = s_tensor[..., 4:5]
        omega = s_tensor[..., 5:6] # Omega is identical in both frames
        
        # 3. Pre-compute Trigonometry
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # 4. Transform Global Velocities to Local Frame
        vx_local = vx_global * c + vy_global * s
        vy_local = -vx_global * s + vy_global * c
        v_local = torch.cat([vx_local, vy_local, omega], dim=-1)
        
        # 5. Concatenate LOCAL velocities and LOCAL commands
        x = torch.cat([v_local, u_tensor], dim=-1)
        
        # 6. Predict the change in LOCAL velocities
        delta_v_local = self.net(x)
        
        # Unpack the local predictions
        dvx_local = delta_v_local[..., 0:1]
        dvy_local = delta_v_local[..., 1:2]
        domega = delta_v_local[..., 2:3]
        
        # 7. Transform Predicted LOCAL Velocity Deltas back to GLOBAL Frame
        dvx_global = dvx_local * c - dvy_local * s
        dvy_global = dvx_local * s + dvy_local * c
        delta_v_global = torch.cat([dvx_global, dvy_global, domega], dim=-1)
        
        # 8. Calculate position deltas using pure kinematics (global velocity * dt)
        # s_tensor[..., 3:6] gives the raw global velocities
        delta_pos = s_tensor[..., 3:6] * self.dt
        
        # 9. Reassemble the full 6D global delta tensor
        delta_tensor = torch.cat([delta_pos, delta_v_global], dim=-1)
        
        return State.from_tensor(delta_tensor)