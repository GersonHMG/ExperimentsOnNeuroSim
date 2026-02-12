import torch
import torch.nn as nn
import numpy as np

class OmniRobotDynamic(nn.Module):
    def __init__(self, dt=0.016, mass=2.8, inertia=0.2, L=0.09):
        super().__init__()
        
        # Physical Constants (Fixed)
        self.dt = dt
        self.M = mass
        self.I = inertia
        self.L = L
        
        # Wheel Configuration
        # Angles: 60, 130, 230, 300 degrees
        self.angles_deg = torch.tensor([60.0, 130.0, 230.0, 300.0])
        self.angles = torch.deg2rad(self.angles_deg)
        
        # Precompute geometry vectors for force projection
        # Force acts perpendicular to the wheel axis (+90 degrees)
        self.force_dirs_x = torch.cos(self.angles + torch.pi/2)
        self.force_dirs_y = torch.sin(self.angles + torch.pi/2)

        # --- TRAINABLE PARAMETERS ---
        # Initialize with reasonable defaults, but wrap as nn.Parameter
        # Shape: (4,) for 4 wheels
        self.motor_gains = nn.Parameter(torch.tensor([10.0, 10.0, 10.0, 10.0]))
        self.friction_coeffs = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5]))

    def _global_to_local(self, vx_g, vy_g, omega_g, theta):
        """
        Differentiable global-to-local transform.
        Args:
            theta: Current robot heading (tensor)
        """
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        vx_local =  c * vx_g + s * vy_g
        vy_local = -s * vx_g + c * vy_g
        
        # Omega is the same in both frames
        return vx_local, vy_local, omega_g

    def inverse_kinematics(self, vx_local, vy_local, omega):
        """
        Calculates wheel velocities (u_i) from robot body velocity.
        """
        # u_i = -sin(alpha)vx + cos(alpha)vy + L*omega
        # We compute this for all 4 wheels at once using broadcasting
        
        # Shape broadcasting:
        # vx_local: (Batch,) -> (Batch, 1)
        # angles: (4,) -> (1, 4)
        
        vx = vx_local.unsqueeze(-1)
        vy = vy_local.unsqueeze(-1)
        w  = omega.unsqueeze(-1)
        
        # Wheel projection terms
        term_x = -torch.sin(self.angles) * vx
        term_y =  torch.cos(self.angles) * vy
        term_w =  self.L * w
        
        return term_x + term_y + term_w

    def forward(self, current_state, commands):
        """
        Forward pass of the physics engine.
        
        Args:
            current_state: Tensor (Batch, 4) -> [vx, vy, omega, theta]
            commands:      Tensor (Batch, 3) -> [vx_cmd, vy_cmd, omega_cmd]
            
        Returns:
            next_state:    Tensor (Batch, 4) -> [vx_new, vy_new, omega_new, theta_new]
        """
        # Unpack Inputs
        vx, vy, omega, theta = current_state[:, 0], current_state[:, 1], current_state[:, 2], current_state[:, 3]
        cmd_vx_g, cmd_vy_g, cmd_omega_g = commands[:, 0], commands[:, 1], commands[:, 2]
        
        # 1. Transform Global Commands -> Local Body Frame
        # Note: We must use the 'theta' passed in the input to preserve gradients
        cmd_vx_local, cmd_vy_local, cmd_omega_local = self._global_to_local(
            cmd_vx_g, cmd_vy_g, cmd_omega_g, theta
        )

        # 2. Calculate Wheel Velocities (Target vs Current)
        target_wheel_vels = self.inverse_kinematics(cmd_vx_local, cmd_vy_local, cmd_omega_local)
        current_wheel_vels = self.inverse_kinematics(vx, vy, omega)
        
        # 3. Calculate Wheel Forces (The Learning Step)
        # Force = Gain * Error - Friction * Velocity
        velocity_errors = target_wheel_vels - current_wheel_vels
        
        # Apply parameters (Broadcasting handles Batch x 4_Wheels)
        wheel_forces = (self.motor_gains * velocity_errors) - \
                       (self.friction_coeffs * current_wheel_vels)
        
        # 4. Resolve Forces to Robot Body (Forward Dynamics)
        # Sum forces from all wheels projected onto body axes
        fx_total = torch.sum(wheel_forces * self.force_dirs_x, dim=1)
        fy_total = torch.sum(wheel_forces * self.force_dirs_y, dim=1)
        torque_total = torch.sum(wheel_forces * self.L, dim=1)
        
        # 5. Integration (Euler)
        # a = F/m
        ax = fx_total / self.M
        ay = fy_total / self.M
        alpha = torque_total / self.I
        
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        omega_new = omega + alpha * self.dt
        
        # 6. Update Heading
        theta_new = theta + omega_new * self.dt
        # Normalize angle -pi to pi
        theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        
        # Stack output
        next_state = torch.stack([vx_new, vy_new, omega_new, theta_new], dim=1)
        
        return next_state