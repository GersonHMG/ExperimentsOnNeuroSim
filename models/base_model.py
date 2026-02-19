import torch.nn as nn
from abc import ABC, abstractmethod
from typing import NamedTuple
import torch

class State(NamedTuple):
    """Global Frame (F_G) Vector s_k in R^6"""
    x: torch.Tensor       # Global X
    y: torch.Tensor       # Global Y
    theta: torch.Tensor   # Global Heading (signed angle)
    vx: torch.Tensor      # Global Velocity X
    vy: torch.Tensor      # Global Velocity Y
    omega: torch.Tensor   # Angular Velocity

    def as_tensor(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.theta, self.vx, self.vy, self.omega], dim=-1)

    @classmethod
    def from_tensor(cls, t: torch.Tensor):
        return cls(*[t[..., i] for i in range(6)])

class Command(NamedTuple):
    """Local Frame (F_L) Vector u_k in R^3"""
    vx_cmd: torch.Tensor    # Commanded Surge
    vy_cmd: torch.Tensor    # Commanded Sway
    omega_cmd: torch.Tensor # Commanded Yaw Rate

    def as_tensor(self) -> torch.Tensor:
        return torch.stack([self.vx_cmd, self.vy_cmd, self.omega_cmd], dim=-1)

    @classmethod
    def from_tensor(cls, t: torch.Tensor):
        return cls(*[t[..., i] for i in range(3)])
    
class DynamicsBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.dt = 0.016  # Fixed sampling interval

    @abstractmethod
    def compute_delta(self, state: State, command: Command) -> State:
        """
        Calculate the change in state (derivative * dt).
        """
        pass

    def wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Normalizes to signed angle system (-pi, pi]."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(self, state_tensor: torch.Tensor, cmd_tensor: torch.Tensor) -> torch.Tensor:
        # Unpack tensors into typeclasses
        s_k = State.from_tensor(state_tensor)
        u_k = Command.from_tensor(cmd_tensor)

        # Compute transition delta
        delta = self.compute_delta(s_k, u_k)

        # Apply Euler integration: s_{k+1} = s_k + delta
        next_state = State(
            x = s_k.x + delta.x,
            y = s_k.y + delta.y,
            theta = self.wrap_angle(s_k.theta + delta.theta),
            vx = s_k.vx + delta.vx,
            vy = s_k.vy + delta.vy,
            omega = s_k.omega + delta.omega
        )

        return next_state.as_tensor()