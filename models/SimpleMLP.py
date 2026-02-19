import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command

class SimpleMLP(DynamicsBase):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # The input is 9D: 6 state variables + 3 command variables
        # The output is 6D: the change (delta) for each state variable
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 6)
        )

    def compute_delta(self, state: State, command: Command) -> State:
        """
        Calculates the change in state using the neural network.
        """
        # 1. Extract tensors from the typed inputs
        s_tensor = state.as_tensor()
        u_tensor = command.as_tensor()
        
        # 2. Concatenate along the last dimension (works for both single inputs and batches)
        # s_tensor shape: (..., 6)
        # u_tensor shape: (..., 3)
        # x shape: (..., 9)
        x = torch.cat([s_tensor, u_tensor], dim=-1)
        
        # 3. Forward pass through the MLP
        # delta_tensor shape: (..., 6)
        delta_tensor = self.net(x)
        
        # 4. Pack the output back into a State object
        return State.from_tensor(delta_tensor)