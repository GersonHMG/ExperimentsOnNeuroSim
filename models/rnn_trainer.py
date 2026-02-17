import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- CHANGE 1: IMPORT THE RNN MODEL ---
# Assuming you saved the previous RNN code in 'RNNModel.py'
from RNNModel import RNNRobotModel 

# --- CONFIGURATION ---
CSV_FILE = r'dataset/processed_data/data0.csv'
BATCH_SIZE = 32
CHUNK_SIZE = 50       # Increased Chunk Size (RNNs are good at long memories)
LEARNING_RATE = 0.001 
EPOCHS = 30           

class RobotTrajectoryDataset(Dataset):
    def __init__(self, csv_file, chunk_size):
        # 1. Load Data
        df = pd.read_csv(csv_file)
        
        # 2. Extract Columns
        # State: [x, y, theta, vx, vy, omega]
        self.states = df[['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']].values.astype(np.float32)
        
        # Commands: [vx_cmd, vy_cmd, omega_cmd]
        self.commands = df[['vx_cmd', 'vy_cmd', 'omega_cmd']].values.astype(np.float32)
        
        self.chunk_size = chunk_size
        # We need enough room for the chunk + 1 extra step for the target
        self.n_samples = len(df) - chunk_size - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Initial State at t=0
        initial_state = self.states[idx]
        
        # Commands for t=0 to t=N
        cmd_seq = self.commands[idx : idx + self.chunk_size]
        
        # Ground Truth targets for t=0 to t=N+1 
        # (The model outputs initial state + N simulated steps)
        target_seq = self.states[idx : idx + self.chunk_size + 1]
        
        return torch.tensor(initial_state), torch.tensor(cmd_seq), torch.tensor(target_seq)

def train():
    # 1. Setup Data
    dataset = RobotTrajectoryDataset(CSV_FILE, CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"Loaded {len(dataset)} training sequences.")

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # --- CHANGE 2: INSTANTIATE RNN ---
    sim = RNNRobotModel(hidden_size=64, device=device)
    sim.train() 
    
    # 3. Optimizer
    optimizer = optim.Adam(sim.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_pos_loss = 0
        total_ang_loss = 0
        
        for batch_idx, (init_state, cmd_seq, target_seq) in enumerate(dataloader):
            init_state = init_state.to(device)
            cmd_seq = cmd_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            
            # --- FORWARD PASS ---
            sim_trajectory = sim(init_state, cmd_seq)
            
            # --- CALCULATE LOSS ---
            
            # 1. Position Loss
            pos_error = sim_trajectory[..., 0:2] - target_seq[..., 0:2]
            loss_pos = torch.mean(pos_error ** 2)
            
            # 2. Velocity Loss
            vel_error = sim_trajectory[..., 3:6] - target_seq[..., 3:6]
            loss_vel = torch.mean(vel_error ** 2)
            
            # 3. Angle Loss (Sine/Cosine)
            pred_theta = sim_trajectory[..., 2]
            target_theta = target_seq[..., 2]
            
            sin_error = torch.sin(pred_theta) - torch.sin(target_theta)
            cos_error = torch.cos(pred_theta) - torch.cos(target_theta)
            loss_theta = torch.mean(sin_error**2 + cos_error**2)
            
            # Combined Loss
            loss = loss_pos + (2.0 * loss_theta) + (10.0 * loss_vel)
            
            # --- BACKPROP ---
            loss.backward()
            
            # Gradient Clipping is CRITICAL for RNNs to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(sim.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_pos_loss += loss_pos.item()
            total_ang_loss += loss_theta.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_pos = total_pos_loss / len(dataloader)
        avg_ang = total_ang_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f} (Pos: {avg_pos:.4f}, Ang: {avg_ang:.4f})")
    
    # --- CHANGE 3: SAVE NAME ---
    save_path = "rnn_robot_params.pth"
    torch.save(sim.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

if __name__ == "__main__":
    train()