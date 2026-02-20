import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command

class RNNModel(DynamicsBase):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The input is 6D: 3 local velocities + 3 local commands
        self.gru = nn.GRU(
            input_size=6, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Output is 3D: the change (delta) for the 3 local velocity variables
        self.fc = nn.Linear(hidden_dim, 3)
        
        # Internal hidden state for the sequence
        self.h = None

    def reset_hidden(self, batch_size: int, device: torch.device):
        """
        Resets the internal hidden state. Called at the start of a sequence.
        """
        self.h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def compute_delta(self, state: State, command: Command) -> State:
        """
        Calculates the change in state. Velocities are computed via an RNN in the 
        local frame. Positions are calculated via pure kinematics.
        """
        # 1. Extract tensors
        s_tensor = state.as_tensor()
        u_tensor = command.as_tensor()
        
        # 2. Extract heading (theta) and global velocities
        theta = s_tensor[..., 2:3]
        vx_global = s_tensor[..., 3:4]
        vy_global = s_tensor[..., 4:5]
        omega = s_tensor[..., 5:6] 
        
        # 3. Pre-compute Trigonometry
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # 4. Transform Global Velocities to Local Frame
        vx_local = vx_global * c + vy_global * s
        vy_local = -vx_global * s + vy_global * c
        v_local = torch.cat([vx_local, vy_local, omega], dim=-1)
        
        # 5. Concatenate LOCAL velocities and LOCAL commands
        # x shape: (batch_size, 6)
        x = torch.cat([v_local, u_tensor], dim=-1)
        
        # 6. Prepare for GRU (Add sequence dimension)
        # x shape: (batch_size, 1, 6)
        x = x.unsqueeze(1)
        
        # Initialize hidden state if necessary
        if self.h is None or self.h.shape[1] != x.shape[0]:
            self.reset_hidden(x.shape[0], x.device)
            
        # 7. Forward pass through GRU
        # out shape: (batch_size, 1, hidden_dim)
        out, self.h = self.gru(x, self.h)
        
        # Remove sequence dimension and pass through final linear layer
        out = out.squeeze(1)
        delta_v_local = self.fc(out)  # Shape: (batch_size, 3)
        
        # Unpack the local predictions
        dvx_local = delta_v_local[..., 0:1]
        dvy_local = delta_v_local[..., 1:2]
        domega = delta_v_local[..., 2:3]
        
        # 8. Transform Predicted LOCAL Velocity Deltas back to GLOBAL Frame
        dvx_global = dvx_local * c - dvy_local * s
        dvy_global = dvx_local * s + dvy_local * c
        delta_v_global = torch.cat([dvx_global, dvy_global, domega], dim=-1)
        
        # 9. Calculate position deltas using pure kinematics (global velocity * dt)
        delta_pos = s_tensor[..., 3:6] * self.dt
        
        # 10. Reassemble the full 6D global delta tensor
        delta_tensor = torch.cat([delta_pos, delta_v_global], dim=-1)
        
        return State.from_tensor(delta_tensor)