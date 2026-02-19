import torch
import torch.nn as nn

class WrappedDynamicsLoss(nn.Module):
    def __init__(self, weights=None):
        super(WrappedDynamicsLoss, self).__init__()
        
        # State Vector: [x, y, theta, vx, vy, omega]

    def wrap_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Wraps the angle difference to the (-pi, pi] range.
        This handles the circular topology of the signed angle system.
        """
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Calculate raw differences (pred - target)
        diff = predictions - targets
        
        # 2. Slice the tensor to isolate the angle
        # Using `...` allows this to work dynamically regardless of whether 
        # the input is [batch, features] or [batch, seq_len, features]
        diff_xy = diff[..., :2]
        diff_theta = self.wrap_angle(diff[..., 2:3]) 
        diff_rest = diff[..., 3:]
        
        # 3. Reassemble the cleanly wrapped differences
        diff_wrapped = torch.cat([diff_xy, diff_theta, diff_rest], dim=-1)
        
        # 4. Square the errors (Standard MSE step)
        squared_error = diff_wrapped ** 2
        
        # 5. Apply dimension weights and return the mean across the batch
        weighted_squared_error = squared_error
        return torch.mean(weighted_squared_error)