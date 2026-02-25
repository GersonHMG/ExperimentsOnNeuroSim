import torch
import torch.nn as nn
from models.base_model import DynamicsBase, State, Command
import torch.nn.functional as F
import math
from typing import Tuple, Union


class GrSimRNNDynamics(DynamicsBase):
    """
    Hybrid Physics + Recurrent Neural Network dynamics model.

    Architecture
    ------------
    - Identical analytical physics backbone to GrSimDynamics (neural_dynamic.py).
    - The feed-forward residual network is replaced with a GRU + linear projection.
      The GRU maintains hidden state across rollout steps, allowing the correction
      term to depend on temporal context (e.g., accumulated slip, unmodelled
      friction history) rather than only the current observation.

    Hidden state
    ------------
    Call reset_hidden(batch_size, device) at the start of every new trajectory
    (the trainers and evaluator do this automatically when they detect the method).

    Input to GRU  : [vx_local, vy_local, omega, vx_cmd, vy_cmd, omega_cmd]  (6-D)
    Output of GRU : projected to 3-D → [F_x_res, F_y_res, Torque_w_res]
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ==========================================
        # 1. Velocity Limits
        # ==========================================
        self._vel_absolute_max = nn.Parameter(torch.tensor(3.0))
        self._vel_angular_max  = nn.Parameter(torch.tensor(5.0))

        # ==========================================
        # 2. Inertial Properties
        # ==========================================
        self._body_mass    = nn.Parameter(torch.tensor(3.0))
        self._body_inertia = nn.Parameter(torch.tensor(0.05))

        # ==========================================
        # 3. Geometric Properties (fixed)
        # ==========================================
        self.robot_radius = torch.tensor(0.09)
        self.wheel_angles = torch.tensor(
            [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
        )

        # ==========================================
        # 4. Force & Friction Constraints
        # ==========================================
        self._wheel_motor_fmax       = nn.Parameter(torch.tensor(20.0))
        self._wheel_tangent_friction = nn.Parameter(torch.tensor(15.0))

        # ==========================================
        # 5. Recurrent Residual Network
        # ==========================================
        # GRU input:  [vx_local, vy_local, omega, vx_cmd, vy_cmd, omega_cmd]  → 6
        # GRU output: hidden_dim → linear head → 3 (residual body forces / torque)
        self.gru = nn.GRU(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.residual_head = nn.Linear(hidden_dim, 3)

        # Near-zero init so the model starts as pure physics
        nn.init.uniform_(self.residual_head.weight, -1e-5, 1e-5)
        nn.init.uniform_(self.residual_head.bias,   -1e-5, 1e-5)

        # Hidden state buffer (populated by reset_hidden)
        self._hidden: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Constrained physical properties
    # ------------------------------------------------------------------

    @property
    def vel_absolute_max(self):    return F.softplus(self._vel_absolute_max)
    @property
    def vel_angular_max(self):     return F.softplus(self._vel_angular_max)
    @property
    def body_mass(self):           return F.softplus(self._body_mass)
    @property
    def body_inertia(self):        return F.softplus(self._body_inertia)
    @property
    def wheel_motor_fmax(self):    return F.softplus(self._wheel_motor_fmax)
    @property
    def wheel_tangent_friction(self): return F.softplus(self._wheel_tangent_friction)

    # ------------------------------------------------------------------
    # Hidden-state management
    # ------------------------------------------------------------------

    def reset_hidden(self, batch_size: int, device: Union[torch.device, str]) -> None:
        """Zeros the GRU hidden state. Call at the start of every new trajectory."""
        self._hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )

    def _ensure_hidden(self, batch_size: int, device: torch.device) -> None:
        if self._hidden is None or self._hidden.shape[1] != batch_size:
            self.reset_hidden(batch_size, device)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_delta_internal(
        self, state: State, command: Command
    ) -> Tuple[State, torch.Tensor]:
        """
        One-step physics + recurrent residual forward pass.
        Returns (state_delta, residual_forces).
        """
        dt = self.dt
        device = state.vx.device

        # ---- 1. Velocity limits ----
        cmd_lin_vel   = torch.stack([command.vx_cmd, command.vy_cmd], dim=-1)
        cmd_lin_speed = torch.linalg.norm(cmd_lin_vel, dim=-1)
        scale_lin     = torch.clamp(self.vel_absolute_max / (cmd_lin_speed + 1e-8), max=1.0)
        vx_cmd_limited    = command.vx_cmd * scale_lin
        vy_cmd_limited    = command.vy_cmd * scale_lin
        omega_cmd_limited = torch.clamp(
            command.omega_cmd, -self.vel_angular_max, self.vel_angular_max
        )

        # ---- 2. Global → Local frame ----
        cos_theta    = torch.cos(state.theta)
        sin_theta    = torch.sin(state.theta)
        vx_curr_local = state.vx * cos_theta + state.vy * sin_theta
        vy_curr_local = -state.vx * sin_theta + state.vy * cos_theta

        # ---- 3. Analytical body forces ----
        a_req_x  = (vx_cmd_limited - vx_curr_local) / dt
        a_req_y  = (vy_cmd_limited - vy_curr_local) / dt
        a_req_w  = (omega_cmd_limited - state.omega) / dt
        F_body_analytical = torch.stack(
            [self.body_mass * a_req_x,
             self.body_mass * a_req_y,
             self.body_inertia * a_req_w],
            dim=-1,
        )  # (Batch, 3)

        # ---- 4. Recurrent residual forces ----
        batch_size = state.vx.shape[0]
        self._ensure_hidden(batch_size, device)

        # GRU expects (batch, seq_len, input_size) — seq_len=1 for step-by-step rollout
        nn_inputs = torch.stack(
            [vx_curr_local, vy_curr_local, state.omega,
             vx_cmd_limited, vy_cmd_limited, omega_cmd_limited],
            dim=-1,
        ).unsqueeze(1)  # (Batch, 1, 6)

        gru_out, self._hidden = self.gru(nn_inputs, self._hidden)
        # gru_out: (Batch, 1, hidden_dim)  →  (Batch, hidden_dim)
        F_body_residual = self.residual_head(gru_out.squeeze(1))  # (Batch, 3)

        # ---- 5. Combined body force ----
        F_body_req = F_body_analytical + F_body_residual

        # ---- 6. Wheel kinematics ----
        sin_a = torch.sin(self.wheel_angles.to(device))
        cos_a = torch.cos(self.wheel_angles.to(device))
        R     = self.robot_radius.to(device).expand(4)
        H     = torch.stack([-sin_a, cos_a, R], dim=1)     # (4, 3)
        H_T   = H.T                                          # (3, 4)
        H_T_pinv = torch.linalg.pinv(H_T)                   # (4, 3)
        F_w_req  = F_body_req @ H_T_pinv.T                  # (Batch, 4)

        # ---- 7. Physical limits ----
        max_force    = torch.min(self.wheel_motor_fmax, self.wheel_tangent_friction)
        F_w_actual   = torch.clamp(F_w_req, -max_force, max_force)
        F_body_actual = F_w_actual @ H                       # (Batch, 3)

        a_act_x = F_body_actual[..., 0] / self.body_mass
        a_act_y = F_body_actual[..., 1] / self.body_mass
        a_act_w = F_body_actual[..., 2] / self.body_inertia

        vx_new_local = vx_curr_local + a_act_x * dt
        vy_new_local = vy_curr_local + a_act_y * dt
        omega_new    = state.omega    + a_act_w * dt

        # ---- 8. Local → Global frame ----
        vx_new_global = vx_new_local * cos_theta - vy_new_local * sin_theta
        vy_new_global = vx_new_local * sin_theta + vy_new_local * cos_theta

        # ---- 9. Deltas ----
        delta = State(
            x     = vx_new_global * dt,
            y     = vy_new_global * dt,
            theta = omega_new * dt,
            vx    = vx_new_global - state.vx,
            vy    = vy_new_global - state.vy,
            omega = omega_new     - state.omega,
        )
        return delta, F_body_residual

    # ------------------------------------------------------------------
    # DynamicsBase interface
    # ------------------------------------------------------------------

    def compute_delta(self, state: State, command: Command) -> State:
        """Required by DynamicsBase. Discards residual forces for standard forward()."""
        delta, _ = self._compute_delta_internal(state, command)
        return delta

    # ------------------------------------------------------------------
    # Extended interface for HybridAutoRegressiveTrainer
    # ------------------------------------------------------------------

    def forward_with_residual(
        self,
        state_tensor: torch.Tensor,
        cmd_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like forward() but also returns the GRU residual forces for the
        DynamicSMAPELoss regularisation term.

        Returns
        -------
        next_state_tensor : (Batch, 6)
        residual_forces   : (Batch, 3)
        """
        s_k = State.from_tensor(state_tensor)
        u_k = Command.from_tensor(cmd_tensor)

        delta, residual_forces = self._compute_delta_internal(s_k, u_k)

        next_state = State(
            x     = s_k.x     + delta.x,
            y     = s_k.y     + delta.y,
            theta = self.wrap_angle(s_k.theta + delta.theta),
            vx    = s_k.vx    + delta.vx,
            vy    = s_k.vy    + delta.vy,
            omega = s_k.omega + delta.omega,
        )
        return next_state.as_tensor(), residual_forces
