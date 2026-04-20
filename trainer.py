import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bars
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
import math

class ModelTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion,
        learning_rate: float = 1e-3, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        writer: SummaryWriter = None
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Standard regression loss. 
        # Note: Be careful with MSE on the `theta` angle due to the -pi/pi wrap.

        self.criterion = criterion
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = writer

    def train_epoch(self, current_target_length: int) -> float:
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Training (Len: {current_target_length})")
        
        for input_states, input_cmds, target_states, future_cmds in pbar:
            
            curr_state_seq = input_states.as_tensor().to(self.device)
            curr_cmd_seq = input_cmds.as_tensor().to(self.device)
            target_state_seq = target_states.as_tensor().to(self.device)
            future_cmd_seq = future_cmds.as_tensor().to(self.device)
            
            self.optimizer.zero_grad()
            predictions = []
            
            # 1. Loop only up to the CURRENT curriculum target length
            for t in range(current_target_length):
                pred_state = self.model(curr_state_seq, curr_cmd_seq)
                pred_state_tensor = pred_state.as_tensor().unsqueeze(1)
                predictions.append(pred_state_tensor)
                
                curr_state_seq = torch.cat([curr_state_seq[:, 1:, :], pred_state_tensor], dim=1)
                next_cmd = future_cmd_seq[:, t:t+1, :]
                curr_cmd_seq = torch.cat([curr_cmd_seq[:, 1:, :], next_cmd], dim=1)
                
            all_predictions = torch.cat(predictions, dim=1)
            
            # 2. SLICE the ground truth targets to match the current prediction length
            current_targets = target_state_seq[:, :current_target_length, :]
            
            # 3. Calculate loss using the shortened sequences
            loss = self.criterion(all_predictions, current_targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    def validate_epoch(self) -> float:
        self.model.eval()
        running_loss = 0.0
        
        # Disable gradient calculation for memory efficiency and speed
        with torch.no_grad():
            # Unpack the 4 sequences from the updated dataset
            for input_states, input_cmds, target_states, future_cmds in self.val_loader:
                
                # 1. Convert NamedTuples to tensors and move to device
                curr_state_seq = input_states.as_tensor().to(self.device)
                curr_cmd_seq = input_cmds.as_tensor().to(self.device)
                
                target_state_seq = target_states.as_tensor().to(self.device)
                future_cmd_seq = future_cmds.as_tensor().to(self.device)
                
                target_length = target_state_seq.shape[1] 
                predictions = []
                
                # 2. Autoregressive Loop
                for t in range(target_length):
                    # Forward pass: predict the next single step
                    pred_state = self.model(curr_state_seq, curr_cmd_seq)
                    
                    # Convert predicted State to tensor and add a sequence dimension
                    pred_state_tensor = pred_state.as_tensor().unsqueeze(1)
                    predictions.append(pred_state_tensor)
                    
                    # Roll the state window: drop oldest step, append new prediction
                    curr_state_seq = torch.cat([curr_state_seq[:, 1:, :], pred_state_tensor], dim=1)
                    
                    # Roll the command window: drop oldest step, append the upcoming known command
                    next_cmd = future_cmd_seq[:, t:t+1, :]
                    curr_cmd_seq = torch.cat([curr_cmd_seq[:, 1:, :], next_cmd], dim=1)
                    
                # 3. Concatenate all step-by-step predictions into one sequence tensor
                all_predictions = torch.cat(predictions, dim=1)
                
                # 4. Compute trajectory loss using your custom wrapper
                loss = self.criterion(all_predictions, target_state_seq)
                
                running_loss += loss.item()
                
        return running_loss / len(self.val_loader)



    def fit(self, epochs: int, curriculum_fraction: float = 1.0) -> Tuple[List[float], List[float]]:
        """
        Main training loop with adjustable curriculum learning.
        
        Args:
            epochs (int): Total number of training epochs.
            curriculum_fraction (float): Fraction of total epochs (0.0 to 1.0) 
                it takes to reach the maximum prediction horizon.
        """
        print(f"Training on device: {self.device}")
        train_losses = []
        val_losses = []
        
        # Get the maximum target length from the dataset
        max_target_length = self.train_loader.dataset.target_length
        
        for epoch in range(1, epochs + 1):
            
            # --- DYNAMIC CURRICULUM SCHEDULE ---
            # 1. Calculate how far along we are in the curriculum phase
            progress = epoch / (epochs * curriculum_fraction)
            
            # 2. Cap the progress at 1.0 so we don't overshoot the max length
            progress = min(progress, 1.0)
            
            # 3. Calculate the target length based on the capped progress
            current_target_length = math.ceil(progress * max_target_length)
            current_target_length = max(1, current_target_length) # Ensure at least 1
            
            print(f"\n--- Epoch {epoch}/{epochs} | Target Length: {current_target_length}/{max_target_length} ---")
            
            # Pass the dynamically calculated length to the training step
            train_loss = self.train_epoch(current_target_length)
            
            # Validation ALWAYS evaluates the full max_target_length for a consistent metric
            val_loss = self.validate_epoch() 
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if self.writer:
                self.writer.add_scalars('Loss/Combined', {
                    'Train': train_loss,
                    'Validation': val_loss
                }, epoch)
                
                # Also log the curriculum length to TensorBoard to visualize the schedule
                self.writer.add_scalar('Curriculum/Target_Length', current_target_length, epoch)
                
                # Plot Weights and Biases
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        self.writer.add_histogram(f'Weights/{name}', param, epoch)
                    elif 'bias' in name:
                        self.writer.add_histogram(f'Biases/{name}', param, epoch)
                        
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                        
                self.writer.flush()
                
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        return train_losses, val_losses