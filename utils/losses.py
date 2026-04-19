import torch
import torch.nn as nn

class WrappedAngleMSE(nn.Module):
    """
    Computes MSE for linear variables and wrapped Angular MSE for angular variables.
    Works seamlessly with both (Batch, Features) and (Batch, Sequence, Features) tensors.
    """
    def __init__(self, angle_index: int = 2):
        super().__init__()
        # In your State tuple, theta is at index 2: (x, y, theta, vx, vy, omega)
        self.angle_index = angle_index
        
        # We use reduction='none' so we can manipulate specific columns before averaging
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Compute standard squared errors for EVERYTHING
        squared_errors = self.mse(pred, target)
        
        # 2. Extract just the angle columns using ellipsis to handle any tensor dimensions
        pred_angle = pred[..., self.angle_index]
        target_angle = target[..., self.angle_index]
        
        # 3. Calculate the shortest angular difference (wrapped between -pi and pi)
        diff = target_angle - pred_angle
        wrapped_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # 4. Replace the standard angle error with the squared wrapped error
        squared_errors[..., self.angle_index] = wrapped_diff ** 2
        
        # 5. Return the mean across all batches, sequences, and features
        return torch.mean(squared_errors)