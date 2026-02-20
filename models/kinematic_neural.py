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
        self._body_mass = nn.Parameter(torch.tensor(3.0))      
        self._body_inertia = nn.Parameter(torch.tensor(0.05))  
        
        # ==========================================
        # 3. Trainable Geometric Properties
        # ==========================================
        self._robot_radius = nn.Parameter(torch.tensor(0.09))  
        
        self.wheel_angles = nn.Parameter(torch.tensor([
            math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4
        ])) 
        
        # ==========================================
        # 4. Trainable Force & Friction Constraints
        # ==========================================
        self._wheel_motor_fmax = nn.Parameter(torch.tensor(20.0))       
        self._wheel_tangent_friction = nn.Parameter(torch.tensor(15.0)) 

        # ==========================================
        # 5. Neural Network for Unmodeled Dynamics
        # ==========================================
        # Input features (6): [vx_current, vy_current, omega_current, vx_cmd, vy_cmd, omega_cmd]
        # Output (3): [residual_ax, residual_ay, residual_aw]
        self.residual_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),          # Tanh is generally smoother for physical dynamics than ReLU
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )
        
        # CRITICAL TRICK: Initialize the last layer to output near zero initially.
        # This forces the model to rely entirely on the physics at the start of training,
        # only relying on the NN when the physics parameters can't explain the data.
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    @property
    def vel_absolute_max(self): return F.softplus(self._vel_absolute_max)
    @property
    def vel_angular_max(self): return F.softplus(self._vel_angular_max)
    @property
    def body_mass(self): return F.softplus(self._body_mass)
    @property
    def body_inertia(self): return F.softplus(self._body_inertia)
    @property
    def robot_radius(self): return F.softplus(self._robot_radius)
    @property
    def wheel_motor_fmax(self): return F.softplus(self._wheel_motor_fmax)
    @property
    def wheel_tangent_friction(self): return F.softplus(self._wheel_tangent_friction)

    def compute_delta(self, state: State, command: Command) -> State:
        dt = self.dt

        # ==========================================
        # 1. VELOCITY LIMITS 
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
        a_req_x = (vx_cmd_limited - vx_curr_local) / dt
        a_req_y = (vy_cmd_limited - vy_curr_local) / dt
        a_req_w = (omega_cmd_limited - state.omega) / dt
        
        F_req_x = self.body_mass * a_req_x
        F_req_y = self.body_mass * a_req_y
        T_req_w = self.body_inertia * a_req_w
        
        F_body_req = torch.stack([F_req_x, F_req_y, T_req_w], dim=-1)
        
        sin_a = torch.sin(self.wheel_angles)
        cos_a = torch.cos(self.wheel_angles)
        R = self.robot_radius.expand(4)
        
        H = torch.stack([-sin_a, cos_a, R], dim=1) 
        H_T = H.T  
        H_T_pinv = torch.linalg.pinv(H_T) 
        
        F_w_req = F_body_req @ H_T_pinv.T
        
        # ==========================================
        # 4. APPLY PHYSICAL LIMITS
        # ==========================================
        max_force = torch.min(self.wheel_motor_fmax, self.wheel_tangent_friction)
        F_w_actual = torch.clamp(F_w_req, -max_force, max_force)
        F_body_actual = F_w_actual @ H
        
        # Calculate Base Physics Accelerations
        a_act_x = F_body_actual[..., 0] / self.body_mass
        a_act_y = F_body_actual[..., 1] / self.body_mass
        a_act_w = F_body_actual[..., 2] / self.body_inertia
        
        # ==========================================
        # 5. APPLY NEURAL NETWORK RESIDUALS
        # ==========================================
        # Pack current state and target command into a single tensor for the NN
        nn_input = torch.stack([
            vx_curr_local, vy_curr_local, state.omega,
            vx_cmd_limited, vy_cmd_limited, omega_cmd_limited
        ], dim=-1)
        
        # Get learned residual accelerations
        a_residual = self.residual_net(nn_input)
        
        # Add the black-box residuals to the white-box physics
        a_act_x = a_act_x + a_residual[..., 0]
        a_act_y = a_act_y + a_residual[..., 1]
        a_act_w = a_act_w + a_residual[..., 2]

        # Update velocities based on combined hybrid acceleration
        vx_new_local = vx_curr_local + a_act_x * dt
        vy_new_local = vy_curr_local + a_act_y * dt
        omega_new = state.omega + a_act_w * dt

        # ==========================================
        # 6. FRAME TRANSFORMATION & DELTAS
        # ==========================================
        vx_new_global = vx_new_local * cos_theta - vy_new_local * sin_theta
        vy_new_global = vx_new_local * sin_theta + vy_new_local * cos_theta

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