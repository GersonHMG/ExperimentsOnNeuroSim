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