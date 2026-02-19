import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command

class GRUDynamics(DynamicsBase):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The GRU consumes the 9D input (6 state + 3 cmd) and updates its hidden state
        self.gru = nn.GRU(
            input_size=9, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # A final linear layer maps the GRU's hidden features to the 6D state delta
        self.fc = nn.Linear(hidden_dim, 6)
        
        # We will store the internal hidden state here during the step-by-step rollout
        self.h = None

    def reset_hidden(self, batch_size: int, device: torch.device):
        """
        Resets the internal hidden state. Must be called at the start of every new sequence.
        """
        self.h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def compute_delta(self, state: State, command: Command) -> State:
        """
        Calculates the change in state using the GRU.
        """
        # 1. Extract tensors (Expected shape: [batch_size, features])
        s_tensor = state.as_tensor()
        u_tensor = command.as_tensor()
        
        # 2. Concatenate into a 9D vector
        x = torch.cat([s_tensor, u_tensor], dim=-1)
        
        # 3. Add a sequence dimension for the GRU
        # GRU with batch_first=True expects shape: [batch_size, seq_length, features]
        # Since we are stepping 1 frame at a time, seq_length = 1
        x = x.unsqueeze(1) 
        
        # Initialize hidden state if it hasn't been created or if the batch size changed
        if self.h is None or self.h.shape[1] != x.shape[0]:
            self.reset_hidden(x.shape[0], x.device)
            
        # 4. Forward pass through GRU
        # out shape: [batch_size, 1, hidden_dim]
        # self.h is automatically updated and stored for the next compute_delta call!
        out, self.h = self.gru(x, self.h)
        
        # 5. Remove the sequence dimension and predict the deltas
        out = out.squeeze(1) # Shape: [batch_size, hidden_dim]
        delta_tensor = self.fc(out) # Shape: [batch_size, 6]
        
        return State.from_tensor(delta_tensor)