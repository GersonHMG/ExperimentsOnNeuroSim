import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class AutoregressiveTrainer:
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, device='cpu', dt=0.016):
        """
        Trainer for autoregressive robot dynamics model.
        
        Args:
            model: The dynamic model (e.g., OmniRobotDynamic)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            optimizer: PyTorch optimizer
            device: 'cpu' or 'cuda'
            dt: Time step for integration (must match dataset generation)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.dt = dt
        self.criterion = nn.MSELoss()
        
        self.model.to(self.device)

    def _integrate_trajectory(self, states):
        """
        Integrates velocities to get positions (x, y) and includes theta.
        Assumes starting position (0,0) for x and y.
        states: (Batch, Steps, 4) -> [vx, vy, omega, theta]
        """
        # vx, vy are at indices 0, 1
        vx = states[:, :, 0]
        vy = states[:, :, 1]
        theta = states[:, :, 3]
        
        x = torch.cumsum(vx * self.dt, dim=1)
        y = torch.cumsum(vy * self.dt, dim=1)
        
        return torch.stack([x, y, theta], dim=2)

    def _autoregressive_loop(self, input_seq, target_seq):
        """
        Runs the autoregressive loop given input and target sequences.
        Returns the predicted trajectory.
        """
        target_steps = target_seq.shape[1]
        
        # --- Initialization ---
        # 1. Get initial state from the LAST step of the input sequence
        # Indices: 0:4=state (vx, vy, omega, theta), 4:7=commands
        
        # State (vx, vy, omega)
        current_state = input_seq[:, -1, 0:4] 
        
        predicted_states = []
        
        # --- Autoregressive Loop ---
        for t in range(target_steps):
            # Get command for current target step (vx_cmd, vy_cmd, omega_cmd)
            # Indices 4:7 are commands
            current_cmd = target_seq[:, t, 4:7]
            
            # Forward pass: Predict next velocity state
            # Model signature: forward(current_state, commands)
            # Output: next_state (vx, vy, omega, theta)
            next_state = self.model(current_state, current_cmd)
            step_state = next_state
            predicted_states.append(step_state)
            
            # Update state for next iteration (Autoregression)
            current_state = next_state
            
        # Stack all predictions: (Batch, Target_Steps, 4)
        predicted_trajectory = torch.stack(predicted_states, dim=1)
        return predicted_trajectory

    def one_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # batch keys: 'input_x', 'target_x'
            # Shape: (batch, steps, 7)
            input_seq = batch['input_x'].to(self.device)
            target_seq = batch['target_x'].to(self.device)
            
            # --- Autoregressive Loop ---
            predicted_trajectory = self._autoregressive_loop(input_seq, target_seq)
            
            # --- Loss Calculation ---
            # Get Ground Truth: (Batch, Target_Steps, 4) -> indices 0:4 (vx, vy, omega, theta)
            # Since we predict velocities and theta, we compare with those directly.
            ground_truth_trajectory = target_seq[:, :, 0:4]
            
            # 1. State Loss (Velocities + Theta)
            state_loss = self.criterion(predicted_trajectory, ground_truth_trajectory)

            # 2. Position Loss (Trajectory Integration)
            pred_pos = self._integrate_trajectory(predicted_trajectory)
            gt_pos = self._integrate_trajectory(ground_truth_trajectory)
            
            position_loss = self.criterion(pred_pos, gt_pos)

            # Compute Total Loss
            loss = state_loss + position_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self):
        """Standard validation loop following the same logic."""
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_seq = batch['input_x'].to(self.device)
                target_seq = batch['target_x'].to(self.device)
                
                predicted_traj = self._autoregressive_loop(input_seq, target_seq)
                gt_traj = target_seq[:, :, 0:4]
                
                state_loss = self.criterion(predicted_traj, gt_traj)
                
                pred_pos = self._integrate_trajectory(predicted_traj)
                gt_pos = self._integrate_trajectory(gt_traj)
                
                position_loss = self.criterion(pred_pos, gt_pos)
                
                loss = state_loss + position_loss
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)