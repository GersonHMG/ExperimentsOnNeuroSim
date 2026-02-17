import torch
import torch.nn as nn
import torch.nn.functional as F

class OmniRobotPhysics(nn.Module):
    """
    Physics model where:
    - Input/Output State: [x, y, theta, VX_GLOBAL, VY_GLOBAL, omega]
    - Commands:           [vx_local, vy_local, omega]
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- 1. PHYSICS CONSTANTS ---
        self.register_buffer('mass', torch.tensor(2.8))           # kg
        self.register_buffer('dt', torch.tensor(0.016))           # 60Hz
        self.register_buffer('radius', torch.tensor(0.09))        # Wheel radius
        
        # --- 2. LEARNABLE PARAMETERS ---
        self.com_offset = nn.Parameter(torch.tensor([-0.05, 0.0])) 
        self._inertia_param = nn.Parameter(torch.tensor(0.011))
        self._motor_gain_param = nn.Parameter(torch.tensor(25.0))
        self._tire_grip_param = nn.Parameter(torch.tensor(15.0))
        self._drag_viscous_param = nn.Parameter(torch.tensor([1.0, 5.0, 1.0]))
        self._drag_coulomb_param = nn.Parameter(torch.tensor([0.5, 3.0, 0.5]))

        # --- 3. KINEMATICS (Mecanum Setup) ---
        angles_deg = torch.tensor([60., 130., 230., 300.])
        angles_rad = torch.deg2rad(angles_deg)
        
        # Forces: Wheel -> Body
        fx_dir = -torch.sin(angles_rad)
        fy_dir = torch.cos(angles_rad)
        torque_arm = torch.full_like(angles_rad, self.radius.item())
        self.register_buffer('force_matrix', torch.stack([fx_dir, fy_dir, torque_arm]))
        
        # IK: Body Vel -> Wheel Vel
        row_vx = -torch.sin(angles_rad)
        row_vy = torch.cos(angles_rad)
        row_omega = torch.full_like(angles_rad, self.radius.item())
        self.register_buffer('ik_matrix', torch.stack([row_vx, row_vy, row_omega]).T)
        
        self.to(device)

    # --- PROPERTIES ---
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
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def get_wheel_velocities(self, body_vel):
        return body_vel @ self.ik_matrix.T

    def dynamics_step(self, state, cmd_velocity):
        """
        state: [x, y, theta, VX_GLOBAL, VY_GLOBAL, omega]
        cmd:   [vx_local, vy_local, omega_cmd]
        """
        # Unpack State
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vx_world, vy_world, omega = state[:, 3], state[:, 4], state[:, 5]
        
        # ==========================================
        # STEP 1: TRANSFORM WORLD VEL -> BODY VEL
        # ==========================================
        # We need Body Velocity to calculate friction and motor forces properly.
        # Rotation Matrix R(-theta)
        c, s = torch.cos(theta), torch.sin(theta)
        
        vx_body =  vx_world * c + vy_world * s
        vy_body = -vx_world * s + vy_world * c
        
        current_body_vel = torch.stack([vx_body, vy_body, omega], dim=1)

        # ==========================================
        # STEP 2: CALCULATE FORCES (IN BODY FRAME)
        # ==========================================
        
        # Motor Forces
        target_wheel_vels = self.get_wheel_velocities(cmd_velocity)
        current_wheel_vels = self.get_wheel_velocities(current_body_vel)
        motor_force = self.motor_gain * (target_wheel_vels - current_wheel_vels)
        F_geo = motor_force @ self.force_matrix.T
        
        # Friction & Drag
        slip_vel = current_body_vel - cmd_velocity
        F_geo -= self.tire_grip * slip_vel
        F_geo -= self.drag_viscous * current_body_vel
        F_geo -= self.drag_coulomb * torch.sign(current_body_vel)
        
        Fx_geo, Fy_geo, Tau_geo = F_geo[:, 0], F_geo[:, 1], F_geo[:, 2]

        # Center of Mass Correction
        dx, dy = self.com_offset[0], self.com_offset[1]
        Tau_com = Tau_geo - (dx * Fy_geo - dy * Fx_geo)
        
        acc_x_com = Fx_geo / self.mass
        acc_y_com = Fy_geo / self.mass
        acc_alpha = Tau_com / self.inertia
        
        # Shift Acceleration back to Geometric Center
        omega_sq = omega ** 2
        acc_tang_x = -acc_alpha * dy 
        acc_tang_y =  acc_alpha * dx
        acc_cent_x = -omega_sq * dx
        acc_cent_y = -omega_sq * dy
        
        # These are the Accelerations in the BODY FRAME
        acc_x_body = acc_x_com + acc_tang_x + acc_cent_x
        acc_y_body = acc_y_com + acc_tang_y + acc_cent_y
        
        # ==========================================
        # STEP 3: TRANSFORM BODY ACCEL -> WORLD ACCEL
        # ==========================================
        # Rotation Matrix R(+theta)
        # acc_world = R * acc_body
        
        acc_x_world = acc_x_body * c - acc_y_body * s
        acc_y_world = acc_x_body * s + acc_y_body * c

        # ==========================================
        # STEP 4: INTEGRATION (GLOBAL FRAME)
        # ==========================================
        
        # Update Global Velocity
        next_vx_world = vx_world + acc_x_world * self.dt
        next_vy_world = vy_world + acc_y_world * self.dt
        next_omega = omega + acc_alpha * self.dt
        
        # Update Global Position
        next_x = x + next_vx_world * self.dt
        next_y = y + next_vy_world * self.dt
        
        next_theta = theta + next_omega * self.dt
        next_theta = self.angle_normalize(next_theta)
        
        return torch.stack([next_x, next_y, next_theta, next_vx_world, next_vy_world, next_omega], dim=1)

    def forward(self, initial_state, command_sequence):
        states = [initial_state]
        curr = initial_state
        for t in range(command_sequence.shape[1]):
            cmd = command_sequence[:, t, :]
            curr = self.dynamics_step(curr, cmd)
            states.append(curr)
        return torch.stack(states, dim=1)