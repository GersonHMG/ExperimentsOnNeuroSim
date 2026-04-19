import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bars
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter

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

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        
        # Wrap loader with tqdm for a nice visual progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        
        # Unpack the 4 sequences from our updated dataset
        for input_states, input_cmds, target_states, future_cmds in pbar:
            
            # 1. Convert NamedTuples to tensors and move to device
            curr_state_seq = input_states.as_tensor().to(self.device)
            curr_cmd_seq = input_cmds.as_tensor().to(self.device)
            
            target_state_seq = target_states.as_tensor().to(self.device)
            future_cmd_seq = future_cmds.as_tensor().to(self.device)
            
            # 2. Zero the gradients
            self.optimizer.zero_grad()
            
            # Increase this while training
            target_length = target_state_seq.shape[1] # How many steps into the future
            predictions = []
            
            # 3. Autoregressive Loop
            for t in range(target_length):
                # Forward pass: predict the next single step
                # (Your CNNModel returns a `State` NamedTuple)
                pred_state = self.model(curr_state_seq, curr_cmd_seq)
                
                # Convert predicted State to tensor and add a sequence dimension
                # Shape becomes: (batch_size, 1, 6)
                pred_state_tensor = pred_state.as_tensor().unsqueeze(1)
                predictions.append(pred_state_tensor)
                
                # Roll the state window: drop oldest step (index 0), append new prediction
                curr_state_seq = torch.cat([curr_state_seq[:, 1:, :], pred_state_tensor], dim=1)
                
                # Roll the command window: drop oldest step, append the upcoming known command
                next_cmd = future_cmd_seq[:, t:t+1, :]
                curr_cmd_seq = torch.cat([curr_cmd_seq[:, 1:, :], next_cmd], dim=1)
                
            # 4. Concatenate all step-by-step predictions into one sequence tensor
            # Final Shape: (batch_size, target_length, 6)
            all_predictions = torch.cat(predictions, dim=1)
            
            # 5. Compute loss across the entire predicted trajectory 
            # (We bypass _calculate_loss here since we are comparing full sequence tensors)
            loss = self.criterion(all_predictions, target_state_seq)
            
            # 6. Backward pass and optimization
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

    def fit(self, epochs: int) -> Tuple[List[float], List[float]]:
        """
        Main training loop.
        
        Returns:
            Tuple[List[float], List[float]]: A tuple containing the training 
            and validation loss histories across all epochs.
        """
        print(f"Training on device: {self.device}")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Record the losses for this epoch
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalars('Loss/Combined', {
                    'Train': train_loss,
                    'Validation': val_loss
                }, epoch)
                self.writer.flush() # Forces it to write to disk immediately
            
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if self.writer:
            self.writer.close()   
         
        return train_losses, val_losses