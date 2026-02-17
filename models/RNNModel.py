import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNRobotModel(nn.Module):
    """
    Recurrent Neural Network (GRU) for Robot Dynamics.
    Captures history-dependent physics (lag, jerk, latency).
    
    Input:  State [x, y, theta, vx_global, vy_global, omega]
    Output: Next State
    """
    def __init__(self, hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.register_buffer('dt', torch.tensor(0.016)) # 60Hz

        # --- THE RECURRENT BRAIN (GRU Cell) ---
        # Input Size: 6 (3 Body Velocities + 3 Commands)
        # Hidden Size: 64 (The "Memory" vector)
        self.rnn_cell = nn.GRUCell(input_size=6, hidden_size=hidden_size)
        
        # --- DECODER ---
        # Maps the hidden state (memory) to physical acceleration
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3) # Output: [acc_x_body, acc_y_body, acc_alpha]
        )
        
        self.to(device)

    def angle_normalize(self, theta):
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(self, initial_state, command_sequence):
        """
        Simulates the trajectory using the RNN.
        initial_state: [Batch, 6]
        command_sequence: [Batch, TimeSteps, 3]
        """
        batch_size = initial_state.shape[0]
        seq_len = command_sequence.shape[1]
        
        # Initialize Hidden State (Memory)
        # We start with zeros (no memory of past events at t=0)
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        
        # Store trajectory
        states = [initial_state]
        curr_state = initial_state
        
        # --- TIME LOOP ---
        for t in range(seq_len):
            cmd = command_sequence[:, t, :]
            
            # 1. Unpack Current State
            x, y, theta = curr_state[:, 0], curr_state[:, 1], curr_state[:, 2]
            vx_world, vy_world, omega = curr_state[:, 3], curr_state[:, 4], curr_state[:, 5]
            
            # 2. Transform World -> Body Frame
            c, s = torch.cos(theta), torch.sin(theta)
            vx_body =  vx_world * c + vy_world * s
            vy_body = -vx_world * s + vy_world * c
            
            # 3. Construct Input for RNN
            # [Body_Vel (3), Command (3)]
            rnn_input = torch.cat([vx_body.unsqueeze(1), 
                                   vy_body.unsqueeze(1), 
                                   omega.unsqueeze(1), 
                                   cmd], dim=1)
            
            # 4. Update RNN Memory
            # h_new = GRU(input, h_old)
            h_t = self.rnn_cell(rnn_input, h_t)
            
            # 5. Predict Acceleration from Memory
            body_accel = self.decoder(h_t)
            
            # 6. Integration (Dynamics Update)
            
            # Update Velocities (Body Frame)
            next_vx_body = vx_body + body_accel[:, 0] * self.dt
            next_vy_body = vy_body + body_accel[:, 1] * self.dt
            next_omega   = omega   + body_accel[:, 2] * self.dt
            
            # Update Angle
            next_theta = theta + next_omega * self.dt
            next_theta = self.angle_normalize(next_theta)
            
            # Transform Body -> World (Using NEW angle)
            c_new, s_new = torch.cos(next_theta), torch.sin(next_theta)
            next_vx_world = next_vx_body * c_new - next_vy_body * s_new
            next_vy_world = next_vx_body * s_new + next_vx_body * c_new
            
            # Update Position
            next_x = x + next_vx_world * self.dt
            next_y = y + next_vy_world * self.dt
            
            # Pack State
            curr_state = torch.stack([next_x, next_y, next_theta, next_vx_world, next_vy_world, next_omega], dim=1)
            states.append(curr_state)
            
        return torch.stack(states, dim=1)