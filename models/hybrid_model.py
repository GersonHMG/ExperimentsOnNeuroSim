import torch
from torch import nn

from utils.utils import State, Command


class HybridModel(nn.Module):
    """
    Hybrid digital twin: physics velocities + neural residual correction.

    A user-supplied `physics_model` provides a baseline next-state prediction
    (we take its vx, vy, omega). A CNN head reads the historical (velocity,
    command-in-global-frame) sequence and outputs a velocity correction
    (Δvx, Δvy, Δω) added on top of the physics output. Pose is then
    integrated with forward Euler.

    `physics_model` must accept (state_tensors, cmd_tensors) of shapes
    (B, W, 6) and (B, W, 3) and return a `State` whose vx, vy, omega
    fields are used here.
    """

    def __init__(self, physics_model: nn.Module, dt: float = 0.016):
        super().__init__()
        self.dt = dt
        self.physics_model = physics_model

        # Residual head — operates on [v_global, cmd_global] sequences.
        # 6 input channels = 3 velocity + 3 rotated command.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=6,  out_channels=32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(output_size=2),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,  out_features=3),
        )
        self.cnn = nn.Sequential(self.feature_extractor, self.regression_head)

    def wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Wrap to (-pi, pi]."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def to_global(self, theta: torch.Tensor) -> torch.Tensor:
        """R(theta): Local -> Global. Returns shape (..., 3, 3)."""
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones  = torch.ones_like(theta)
        row1 = torch.stack([c,    -s,    zeros], dim=-1)
        row2 = torch.stack([s,     c,    zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones],  dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def forward(
        self,
        state_tensors: torch.Tensor,   # (B, W, 6)
        cmd_tensors:   torch.Tensor,   # (B, W, 3) — local frame
    ) -> State:
        # 1. CNN residual: predicts a velocity correction from history.
        velocity_tensors = state_tensors[..., 3:6]                       # (B, W, 3) global
        theta_tensors    = state_tensors[..., 2]                         # (B, W)
        R_global         = self.to_global(theta_tensors)                 # (B, W, 3, 3)
        cmd_global       = torch.matmul(R_global, cmd_tensors.unsqueeze(-1)).squeeze(-1)

        feats = torch.cat([velocity_tensors, cmd_global], dim=-1)        # (B, W, 6)
        feats = feats.permute(0, 2, 1)                                   # (B, 6, W)
        out   = self.cnn(feats)                                          # (B, 3)
        res_vx, res_vy, res_omega = out[:, 0], out[:, 1], out[:, 2]

        # 2. Physics baseline: full next-state prediction; we take the velocities.
        s_physics = self.physics_model(state_tensors, cmd_tensors)

        # 3. Combine: physics velocities + neural residual.
        vx    = s_physics.vx    + res_vx
        vy    = s_physics.vy    + res_vy
        omega = s_physics.omega + res_omega

        # 4. Integrate pose from the last historical pose using combined velocities.
        s_k = State.from_tensor(state_tensors[:, -1, :])
        x_new     = s_k.x + vx * self.dt
        y_new     = s_k.y + vy * self.dt
        theta_new = self.wrap_angle(s_k.theta + omega * self.dt)

        return State(
            x     = x_new,
            y     = y_new,
            theta = theta_new,
            vx    = vx,
            vy    = vy,
            omega = omega,
        )