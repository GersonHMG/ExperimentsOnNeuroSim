import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralOmniRobot(nn.Module):
    """
    A Pure Neural Network model for robot dynamics.
    Replaces all physics equations (F=ma, friction, motor models) with an MLP.
    
    Input:  State [x, y, theta, VX_GLOBAL, VY_GLOBAL, omega]
    Output: Next State
    """
    def __init__(self, hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.register_buffer('dt', torch.tensor(0.016)) # 60Hz

        # --- THE BRAIN (MLP) ---
        # Input:  6 dims (vx_body, vy_body, omega, vx_cmd, vy_cmd, w_cmd)
        # Output: 3 dims (acc_x_body, acc_y_body, acc_omega)
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),               # Tanh is smooth, good for physics
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3) # Predicts Body Acceleration
        )
        
        # Initialize final layer with small weights for stability at start
        # This ensures the robot doesn't jump randomly before training
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.constant_(self.net[-1].bias, 0)
        
        self.to(device)

    def angle_normalize(self, theta):
        """Wraps angle to [-pi, pi]"""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def dynamics_step(self, state, cmd):
        """
        Single step prediction using the Neural Network.
        """
        # Unpack State (Global Frame)
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vx_world, vy_world, omega = state[:, 3], state[:, 4], state[:, 5]
        
        # ==========================================
        # STEP 1: PREPARE INPUTS (Global -> Body)
        # ==========================================
        # The NN must learn in the Body Frame to be generalizable.
        # If we fed it World Coords, it would learn that "North is fast", 
        # rather than "Forward is fast".
        
        c, s = torch.cos(theta), torch.sin(theta)
        
        # Rotate World Velocity to Body Velocity
        vx_body =  vx_world * c + vy_world * s
        vy_body = -vx_world * s + vy_world * c
        
        # Construct Input Vector: [Body_Vel (3), Command (3)]
        # Shape: [Batch, 6]
        current_body_vel = torch.stack([vx_body, vy_body, omega], dim=1)
        nn_input = torch.cat([current_body_vel, cmd], dim=1)
        
        # ==========================================
        # STEP 2: NEURAL PREDICTION
        # ==========================================
        # The NN predicts the acceleration (change in velocity)
        # output: [ax_body, ay_body, alpha]
        body_accel = self.net(nn_input)
        
        # ==========================================
        # STEP 3: INTEGRATION
        # ==========================================
        
        # 1. Update Body Velocity
        # v_new = v_old + a * dt
        next_body_vel = current_body_vel + body_accel * self.dt
        
        next_vx_body = next_body_vel[:, 0]
        next_vy_body = next_body_vel[:, 1]
        next_omega     = next_body_vel[:, 2]
        
        # 2. Update Position & Orientation
        next_theta = theta + next_omega * self.dt
        next_theta = self.angle_normalize(next_theta)
        
        # 3. Rotate back to World Frame (for position update)
        # Using the NEW angle for better integration stability
        c_new, s_new = torch.cos(next_theta), torch.sin(next_theta)
        
        next_vx_world = next_vx_body * c_new - next_vy_body * s_new
        next_vy_world = next_vx_body * s_new + next_vx_body * c_new
        
        next_x = x + next_vx_world * self.dt
        next_y = y + next_vy_world * self.dt
        
        return torch.stack([next_x, next_y, next_theta, next_vx_world, next_vy_world, next_omega], dim=1)

    def forward(self, initial_state, command_sequence):
        states = [initial_state]
        curr = initial_state
        
        for t in range(command_sequence.shape[1]):
            cmd = command_sequence[:, t, :]
            curr = self.dynamics_step(curr, cmd)
            states.append(curr)
            
        return torch.stack(states, dim=1)