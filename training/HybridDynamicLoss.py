import torch
import torch.nn as nn

class DynamicSMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8, lambda_residual=0.01):
        super(DynamicSMAPELoss, self).__init__()
        self.epsilon = epsilon
        
        # The penalty multiplier for the neural network's intervention
        self.lambda_residual = lambda_residual

    def wrap_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Wraps the angle difference to the (-pi, pi] range.
        Handles the circular topology of the signed angle system.
        """
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, residual_forces: torch.Tensor):
        """
        predictions: Output states from the model (Batch, 6)
        targets: Ground truth states (Batch, 6)
        residual_forces: The output from the residual neural network (Batch, 3)
        """
        # ==========================================
        # 1. Trajectory Loss (Your Wrapped sMAPE)
        # ==========================================
        # Calculate raw differences (pred - target)
        diff = predictions - targets
        
        # Slice the tensor to isolate the angle (assuming [x, y, theta, vx, vy, omega])
        diff_xy = diff[..., :2]
        diff_theta = self.wrap_angle(diff[..., 2:3]) 
        diff_rest = diff[..., 3:]
        
        # Reassemble the cleanly wrapped differences
        diff_wrapped = torch.cat([diff_xy, diff_theta, diff_rest], dim=-1)
        
        # Compute sMAPE components
        numerator = 2.0 * torch.abs(diff_wrapped)
        denominator = torch.abs(predictions) + torch.abs(targets) + self.epsilon
        
        # Calculate unweighted sMAPE and get the mean across the batch
        smape_raw = numerator / denominator
        state_loss = torch.mean(smape_raw)

        # ==========================================
        # 2. Residual Penalty (The Physics Constraint)
        # ==========================================
        # L2 Regularization on the forces predicted by the neural network
        residual_penalty = torch.mean(residual_forces ** 2)

        # ==========================================
        # 3. Total Combined Loss
        # ==========================================
        total_loss = state_loss + (self.lambda_residual * residual_penalty)
        
        return total_loss, state_loss, residual_penalty