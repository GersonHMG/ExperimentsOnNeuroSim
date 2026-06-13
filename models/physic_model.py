import torch
import torch.nn as nn

from utils.utils import State, Command


class RobotDigitalTwin(nn.Module):
    """
    Kinematic digital twin with:

      • Slew-rate (acceleration) limits per axis (surge, sway, yaw).
      • Linear-to-angular coupling: linear acceleration in the local frame
        induces angular acceleration via an unknown CoM offset, captured by
        two parameters (k_x, k_y) acting on (Δv_x, Δv_y) in the local frame.

    All dynamics live in the local frame; the result is rotated into the
    global frame for pose integration with forward Euler at dt = 0.016 s.

        v_local_k       = R(-theta_k) @ v_global_k

        # 2. Linear velocity update (per axis): slew-rate limited tracking
        d_v_drive       = clip(u_k - v_local_k,   ±a_max · dt)
        v_local_{k+1}   = v_local_k + d_v_drive

        # 3. Angular velocity update: slew-rate tracking + CoM coupling
        d_omega_drive   = clip(omega_cmd - omega, ±a_max[2] · dt)
        d_omega_couple  = k_x · Δv_x_local + k_y · Δv_y_local
        omega_{k+1}     = omega_k + d_omega_drive + d_omega_couple

        v_global_{k+1}  = R(theta_k) @ v_local_{k+1}
        p_{k+1}         = p_k + v_global_{k+1} · dt
        theta_{k+1}     = theta_k + omega_{k+1} · dt

    Parameters
    ----------
    a_max  (3,) : max |dv/dt| per axis (surge m/s², sway m/s², yaw rad/s²)
    k_couple (2,) : (k_x, k_y), units rad/m — derives from m·y_cm/I_z and
                    −m·x_cm/I_z; either sign. Zero ⇔ CoM at geometric center.
    """

    def __init__(
        self,
        dt: float = 0.016,
        a_max_init: tuple = (3.0, 3.0, 10.0),   # surge, sway, yaw
        k_couple_init: tuple = (0.0, 0.0),      # (k_x, k_y), CoM offset coupling
    ):
        super().__init__()
        self.dt = dt

        # Positivity-constrained -> log-space.
        self.log_a_max = nn.Parameter(torch.log(torch.tensor(a_max_init, dtype=torch.float32)))
        # k_x, k_y can be either sign -> raw parameter.
        self.k_couple  = nn.Parameter(torch.tensor(k_couple_init, dtype=torch.float32))

    @property
    def a_max(self) -> torch.Tensor:
        return torch.exp(self.log_a_max)

    @staticmethod
    def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
        """Wrap an angle to (-pi, pi]."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(
        self,
        state_tensors: torch.Tensor,   # (B, W, 6)
        cmd_tensors: torch.Tensor,     # (B, W, 3) — local frame
    ) -> State:
        s_k = State.from_tensor(state_tensors[:, -1, :])
        u_k = Command.from_tensor(cmd_tensors[:, -1, :])

        cos_t = torch.cos(s_k.theta)
        sin_t = torch.sin(s_k.theta)

        # 1. Current global velocity -> local frame (R(-theta) @ v_global).
        vx_local =  cos_t * s_k.vx + sin_t * s_k.vy
        vy_local = -sin_t * s_k.vx + cos_t * s_k.vy
        # omega is identical in local and global frames.

        # 2. Linear velocity update: slew-rate limited tracking of the command.
        delta_max = self.a_max * self.dt                                              # (3,)
        d_vx = torch.clamp(u_k.vx_cmd - vx_local, min=-delta_max[0], max=delta_max[0])
        d_vy = torch.clamp(u_k.vy_cmd - vy_local, min=-delta_max[1], max=delta_max[1])

        vx_local_new = vx_local + d_vx
        vy_local_new = vy_local + d_vy

        # 3. Angular velocity update: slew-rate tracking + linear-acc coupling.
        #    Δω_couple = k_x · Δv_x_local + k_y · Δv_y_local
        #              = (k_x · a_x_local + k_y · a_y_local) · dt
        d_w_drive  = torch.clamp(u_k.omega_cmd - s_k.omega, min=-delta_max[2], max=delta_max[2])
        d_w_couple = (
            self.k_couple[0] * (vx_local_new - vx_local)
            + self.k_couple[1] * (vy_local_new - vy_local)
        )
        omega_new = s_k.omega + d_w_drive + d_w_couple

        # 4. New local velocity -> global frame (R(theta) @ v_local).
        vx_new = cos_t * vx_local_new - sin_t * vy_local_new
        vy_new = sin_t * vx_local_new + cos_t * vy_local_new

        # 5. Forward Euler integration of pose using the new velocity.
        x_new     = s_k.x + vx_new * self.dt
        y_new     = s_k.y + vy_new * self.dt
        theta_new = self.wrap_angle(s_k.theta + omega_new * self.dt)

        return State(
            x     = x_new,
            y     = y_new,
            theta = theta_new,
            vx    = vx_new,
            vy    = vy_new,
            omega = omega_new,
        )