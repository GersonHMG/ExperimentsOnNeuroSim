import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualNet(nn.Module):
    """
    Predicts additive acceleration errors (residuals) that the analytical 
    physics model fails to capture.
    Input:  [vx, vy, omega, vx_cmd, vy_cmd, omega_cmd] (6 dims)
    Output: [acc_x_res, acc_y_res, alpha_res]          (3 dims)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3) 
        
        # Initialize output layer weights near zero.
        # This ensures the model behaves exactly like standard physics 
        # at the start of training (Residual = 0).
        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x)) # Tanh prevents exploding gradients
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class HybridOmniRobotSim(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- 1. NEURAL COMPONENT ---
        self.residual_net = ResidualNet()
        
        # --- 2. PHYSICS CONSTANTS ---
        self.register_buffer('mass', torch.tensor(2.8))           # kg
        self.register_buffer('dt', torch.tensor(0.016))           # 60Hz
        self.register_buffer('radius', torch.tensor(0.09))        # Wheel radius
        
        # --- 3. LEARNABLE PHYSICS PARAMETERS ---
        # Using softplus in property accessors to enforce positivity where needed
        self.com_offset = nn.Parameter(torch.tensor([-0.05, 0.0])) 
        self._inertia_param = nn.Parameter(torch.tensor(0.011))
        self._motor_gain_param = nn.Parameter(torch.tensor(25.0))
        self._tire_grip_param = nn.Parameter(torch.tensor(15.0))
        self._drag_viscous_param = nn.Parameter(torch.tensor([1.0, 5.0, 1.0]))
        self._drag_coulomb_param = nn.Parameter(torch.tensor([0.5, 3.0, 0.5]))

        # --- 4. KINEMATICS (Mecanum/Omni setup) ---
        angles_deg = torch.tensor([60., 130., 230., 300.])
        angles_rad = torch.deg2rad(angles_deg)
        
        # Force Projection Matrix (Wheel -> Body Force)
        fx_dir = -torch.sin(angles_rad)
        fy_dir = torch.cos(angles_rad)
        torque_arm = torch.full_like(angles_rad, self.radius.item())
        self.register_buffer('force_matrix', torch.stack([fx_dir, fy_dir, torque_arm]))
        
        # Inverse Kinematics Matrix (Body Velocity -> Wheel Velocity)
        row_vx = -torch.sin(angles_rad)
        row_vy = torch.cos(angles_rad)
        row_omega = torch.full_like(angles_rad, self.radius.item())
        self.register_buffer('ik_matrix', torch.stack([row_vx, row_vy, row_omega]).T)
        
        self.to(device)

    # --- SAFE PARAMETER ACCESSORS ---
    @property
    def inertia(self): return F.softplus(self._inertia_param) + 1e-4
    @property
    def motor_gain(self): return F.softplus(self._motor_gain_param)
    @property
    def tire_grip(self): return F.softplus(self._tire_grip_param)
    @property
    def drag_viscous(self): return F.softplus(self._drag_viscous_param)
    @property
    def drag_coulomb(self): return F.softplus(self._drag_coulomb_param)

    def angle_normalize(self, theta):
        """Wraps angle to [-pi, pi]"""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def get_wheel_velocities(self, body_vel):
        return body_vel @ self.ik_matrix.T

    def dynamics_step(self, state, cmd_velocity):
        """
        state: [batch, 6] -> x, y, theta, vx, vy, omega
        cmd_velocity: [batch, 3] -> vx_cmd, vy_cmd, omega_cmd
        """
        # Unpack State
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        current_body_vel = state[:, 3:6] 
        
        # ==========================================
        # PART A: ANALYTICAL PHYSICS (The "Prior")
        # ==========================================
        
        # 1. Calculate Motor Forces
        # Motors try to reach target wheel speed based on command
        target_wheel_vels = self.get_wheel_velocities(cmd_velocity)
        current_wheel_vels = self.get_wheel_velocities(current_body_vel)
        motor_force = self.motor_gain * (target_wheel_vels - current_wheel_vels)
        
        # Project wheel forces to body frame (Geometric Center)
        F_geo = motor_force @ self.force_matrix.T
        
        # 2. Add Grip/Slip Dynamics & Drag
        # Simple friction model opposing the difference between command and actual
        slip_vel = current_body_vel - cmd_velocity 
        F_geo = F_geo - (self.tire_grip * slip_vel)
        
        # Viscous drag (proportional to velocity) + Coulomb drag (constant friction)
        F_geo = F_geo - (self.drag_viscous * current_body_vel)
        F_geo = F_geo - (self.drag_coulomb * torch.sign(current_body_vel))
        
        Fx_geo, Fy_geo, Tau_geo = F_geo[:, 0], F_geo[:, 1], F_geo[:, 2]

        # 3. Center of Mass (CoM) Transformation
        # We must shift forces from Geometric Center to Center of Mass
        dx, dy = self.com_offset[0], self.com_offset[1]
        
        # Torque at CoM = Torque_geo + (r_geo_to_com x F_geo)
        # r_geo_to_com is (-dx, -dy). Cross product results in:
        Tau_com = Tau_geo - (dx * Fy_geo - dy * Fx_geo)
        
        acc_x_com = Fx_geo / self.mass
        acc_y_com = Fy_geo / self.mass
        acc_alpha = Tau_com / self.inertia
        
        # 4. Shift Acceleration back to Geometric Center (for Kinematics)
        # a_geo = a_com + alpha x r + omega x (omega x r)
        # r is vector from CoM to GeoCenter (dx, dy)
        omega = state[:, 5]
        omega_sq = omega ** 2
        
        # Tangential Component (alpha x r)
        # alpha_z cross (dx, dy) -> (-alpha*dy, alpha*dx)
        acc_tang_x = -acc_alpha * dy 
        acc_tang_y =  acc_alpha * dx
        
        # Centripetal Component (omega x (omega x r)) -> -omega^2 * r
        acc_cent_x = -omega_sq * dx
        acc_cent_y = -omega_sq * dy
        
        acc_x_phys = acc_x_com + acc_tang_x + acc_cent_x
        acc_y_phys = acc_y_com + acc_tang_y + acc_cent_y
        
        # ==========================================
        # PART B: NEURAL RESIDUAL (The "Correction")
        # ==========================================
        
        nn_input = torch.cat([current_body_vel, cmd_velocity], dim=1)
        acc_residuals = self.residual_net(nn_input)
        
        # ==========================================
        # PART C: INTEGRATION
        # ==========================================
        
        # Total Acceleration
        acc_x_total = acc_x_phys + acc_residuals[:, 0]
        acc_y_total = acc_y_phys + acc_residuals[:, 1]
        acc_alpha_total = acc_alpha + acc_residuals[:, 2]
        
        # 1. Update Velocity (Symplectic Euler Step 1)
        next_vx = current_body_vel[:, 0] + acc_x_total * self.dt
        next_vy = current_body_vel[:, 1] + acc_y_total * self.dt
        next_omega = current_body_vel[:, 2] + acc_alpha_total * self.dt
        
        # 2. Update Position (Symplectic Euler Step 2)
        # We use the NEW velocity and NEW angle for the world frame projection
        # to ensure better stability during rotation.
        
        next_theta = theta + next_omega * self.dt
        next_theta = self.angle_normalize(next_theta) # Wrap to [-pi, pi]
        
        c, s = torch.cos(next_theta), torch.sin(next_theta)
        
        # Rotate body velocity to world frame
        vx_world = c * next_vx - s * next_vy
        vy_world = s * next_vx + c * next_vy
        
        next_x = x + vx_world * self.dt
        next_y = y + vy_world * self.dt
        
        return torch.stack([next_x, next_y, next_theta, next_vx, next_vy, next_omega], dim=1)

    def forward(self, initial_state, command_sequence):
        """
        initial_state: [x,y,theta, vx,vy,omega] [batch, 6]
        command_sequence: [vx_cmd, vy_cmd, omega_cmd] [batch, seq_len, 3]
        Output: [batch, seq_len + 1, 6]
        """
        states = [initial_state]
        curr = initial_state
        
        # Iterate over time
        for t in range(command_sequence.shape[1]):
            cmd = command_sequence[:, t, :]
            curr = self.dynamics_step(curr, cmd)
            states.append(curr)
            
        return torch.stack(states, dim=1)