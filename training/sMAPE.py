import torch
import torch.nn as nn

class WrappedDynamicsSMAPELoss(nn.Module):
    def __init__(self, weights=None, epsilon=1e-8):
        super(WrappedDynamicsSMAPELoss, self).__init__()
        
        # State Vector: [x, y, theta, vx, vy, omega]
        # Default weights emphasize the angle (index 2) and angular velocity (index 5)
        if weights is None:
            weights = torch.tensor([1.0, 1.0, 10.0, 1.0, 1.0, 5.0], dtype=torch.float32)
        
        # Register buffer for device synchronization
        self.register_buffer('weights', weights)
        
        # Epsilon prevents division by zero when both target and prediction are 0
        self.epsilon = epsilon

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
        diff_xy = diff[..., :2]
        diff_theta = self.wrap_angle(diff[..., 2:3]) 
        diff_rest = diff[..., 3:]
        
        # 3. Reassemble the cleanly wrapped differences for the numerator
        diff_wrapped = torch.cat([diff_xy, diff_theta, diff_rest], dim=-1)
        
        # 4. Compute sMAPE components
        numerator = 2.0 * torch.abs(diff_wrapped)
        
        # Note: We use the raw predictions and targets for the denominator.
        # Ensure your theta predictions and targets are constrained to [-pi, pi] 
        # elsewhere in your pipeline so the denominator scales correctly.
        denominator = torch.abs(predictions) + torch.abs(targets) + self.epsilon
        
        # 5. Calculate unweighted sMAPE
        smape_raw = numerator / denominator
        
        # 6. Apply dimension weights and return the mean across the batch
        weighted_smape = smape_raw * self.weights
        
        return torch.mean(weighted_smape)