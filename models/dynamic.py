import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command
import torch.nn.functional as F
import math

class GrSimDynamics(DynamicsBase):
    def __init__(self):
        super().__init__()
        
        # ==========================================
        # 1. Base Velocity Limits
        # ==========================================
        self._vel_absolute_max = nn.Parameter(torch.tensor(3.0))       
        self._vel_angular_max = nn.Parameter(torch.tensor(5.0))        
        
        # ==========================================
        # 2. Trainable Inertial Properties
        # ==========================================
        self._body_mass = nn.Parameter(torch.tensor(3.0))      # Equivalent to BodyMass
        self._body_inertia = nn.Parameter(torch.tensor(0.05))  # Rotational inertia
        
        # ==========================================
        # 3. Trainable Geometric Properties
        # ==========================================
        self.robot_radius = torch.tensor(0.09)
        
        # Wheel angles (alpha_1 to alpha_4) initialized to standard 45° offset (X-configuration)
        self.wheel_angles = torch.tensor([math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4])
        
        # ==========================================
        # 4. Trainable Force & Friction Constraints
        # ==========================================
        self._wheel_motor_fmax = nn.Parameter(torch.tensor(20.0))       # Equivalent to Wheel_Motor_FMax
        self._wheel_tangent_friction = nn.Parameter(torch.tensor(15.0)) # Equivalent to WheelTangentFriction

    # Enforce strictly positive values for physical quantities
    @property
    def vel_absolute_max(self): return F.softplus(self._vel_absolute_max)
    @property
    def vel_angular_max(self): return F.softplus(self._vel_angular_max)
    @property
    def body_mass(self): return F.softplus(self._body_mass)
    @property
    def body_inertia(self): return F.softplus(self._body_inertia)
    @property
    def wheel_motor_fmax(self): return F.softplus(self._wheel_motor_fmax)
    @property
    def wheel_tangent_friction(self): return F.softplus(self._wheel_tangent_friction)

    def compute_delta(self, state: State, command: Command) -> State:
        dt = self.dt

        # ==========================================
        # 1. VELOCITY LIMITS (Same as before)
        # ==========================================
        cmd_lin_vel = torch.stack([command.vx_cmd, command.vy_cmd], dim=-1)
        cmd_lin_speed = torch.linalg.norm(cmd_lin_vel, dim=-1)
        
        scale_lin = torch.clamp(self.vel_absolute_max / (cmd_lin_speed + 1e-8), max=1.0)
        vx_cmd_limited = command.vx_cmd * scale_lin
        vy_cmd_limited = command.vy_cmd * scale_lin

        omega_cmd_limited = torch.clamp(
            command.omega_cmd, 
            -self.vel_angular_max, 
            self.vel_angular_max
        )

        # ==========================================
        # 2. FRAME TRANSFORMATION (Global to Local)
        # ==========================================
        cos_theta = torch.cos(state.theta)
        sin_theta = torch.sin(state.theta)
        
        vx_curr_local = state.vx * cos_theta + state.vy * sin_theta
        vy_curr_local = -state.vx * sin_theta + state.vy * cos_theta
        
        # ==========================================
        # 3. KINEMATIC & DYNAMIC FORCE CALCULATIONS
        # ==========================================
        # Calculate the ideal required accelerations to meet the command
        a_req_x = (vx_cmd_limited - vx_curr_local) / dt
        a_req_y = (vy_cmd_limited - vy_curr_local) / dt
        a_req_w = (omega_cmd_limited - state.omega) / dt
        
        # Calculate required body forces using Newton's second law (F = ma, Torque = I * alpha)
        F_req_x = self.body_mass * a_req_x
        F_req_y = self.body_mass * a_req_y
        T_req_w = self.body_inertia * a_req_w
        
        # Stack into Body Force vector: Shape (Batch, 3)
        F_body_req = torch.stack([F_req_x, F_req_y, T_req_w], dim=-1)
        
        # Build Kinematic Coupling Matrix H from trainable wheel geometry
        sin_a = torch.sin(self.wheel_angles)
        cos_a = torch.cos(self.wheel_angles)
        R = self.robot_radius.expand(4)
        
        # H maps Wheel states to Body states. Shape: (4, 3)
        H = torch.stack([-sin_a, cos_a, R], dim=1) 
        
        # Use pseudo-inverse of H^T to distribute required body force across the 4 wheels optimally
        H_T = H.T  # Shape: (3, 4)
        H_T_pinv = torch.linalg.pinv(H_T) # Shape: (4, 3)
        
        # Batched matrix multiplication for row vectors: (Batch, 3) @ (3, 4) -> (Batch, 4) wheel forces
        F_w_req = F_body_req @ H_T_pinv.T
        
        # ==========================================
        # 4. APPLY PHYSICAL LIMITS (Motor Output & Slip)
        # ==========================================
        # A wheel cannot push harder than its motor allows, nor harder than its traction limit
        max_force = torch.min(self.wheel_motor_fmax, self.wheel_tangent_friction)
        
        # Clamp individual wheel forces
        F_w_actual = torch.clamp(F_w_req, -max_force, max_force)
        
        # Map actual clamped wheel forces back to actual body forces
        # Batched multiplication: (Batch, 4) @ (4, 3) -> (Batch, 3)
        F_body_actual = F_w_actual @ H
        
        # Calculate the true resulting accelerations of the robot chassis
        a_act_x = F_body_actual[..., 0] / self.body_mass
        a_act_y = F_body_actual[..., 1] / self.body_mass
        a_act_w = F_body_actual[..., 2] / self.body_inertia
        
        # Update velocities based on achievable physical acceleration
        vx_new_local = vx_curr_local + a_act_x * dt
        vy_new_local = vy_curr_local + a_act_y * dt
        omega_new = state.omega + a_act_w * dt

        # ==========================================
        # 5. FRAME TRANSFORMATION (Local back to Global)
        # ==========================================
        vx_new_global = vx_new_local * cos_theta - vy_new_local * sin_theta
        vy_new_global = vx_new_local * sin_theta + vy_new_local * cos_theta

        # ==========================================
        # 6. COMPUTE DELTAS
        # ==========================================
        delta_x = vx_new_global * dt
        delta_y = vy_new_global * dt
        delta_theta = omega_new * dt

        delta_vx = vx_new_global - state.vx
        delta_vy = vy_new_global - state.vy
        delta_omega = omega_new - state.omega

        return State(
            x=delta_x,
            y=delta_y,
            theta=delta_theta,
            vx=delta_vx,
            vy=delta_vy,
            omega=delta_omega
        )