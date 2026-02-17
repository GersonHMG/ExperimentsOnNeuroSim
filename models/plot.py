import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from HybridModel import HybridOmniRobotSim 
#from PhysicsModel import OmniRobotPhysics
from NeuralNetworkSim import NeuralOmniRobot

# --- CONFIGURATION ---
MODEL_PATH = 'neural_network_sim.pth' 
CSV_FILE = 'dataset/processed_data/data0.csv'

def plot_full_results():
    # 1. Load Data
    # Slicing [:100] for a short segment, remove [:100] to plot everything
    #df = pd.read_csv(CSV_FILE)[400:500]
    df = pd.read_csv(CSV_FILE)[0:100]
    # Extract Ground Truth (Positions)
    real_x = df['filtered_x'].values
    real_y = df['filtered_y'].values
    
    # Extract Ground Truth (Velocities)
    real_vx = df['vx'].values
    real_vy = df['vy'].values
    real_omega = df['omega'].values
    
    # Extract Inputs (Commands)
    cmd_vx = df['vx_cmd'].values
    cmd_vy = df['vy_cmd'].values
    cmd_w  = df['omega_cmd'].values
    
    # Create Command Tensor
    commands = np.stack([cmd_vx, cmd_vy, cmd_w], axis=1)
    commands_tensor = torch.tensor(commands, dtype=torch.float32).unsqueeze(0)

    # 2. Load Simulator & Model
    device = 'cpu'
    sim = NeuralOmniRobot(device=device)
    sim.load_state_dict(torch.load(MODEL_PATH, map_location=device))


    # 3. Run Simulation
    # Initial State: [x, y, theta, vx, vy, omega]
    x0 = df.iloc[0]
    init_state = torch.tensor([[
        x0.get('filtered_x', 0.0), 
        x0.get('filtered_y', 0.0), 
        x0.get('filtered_theta', 0.0),
        x0['vx'], 
        x0['vy'], 
        x0['omega']
    ]], dtype=torch.float32)

    print(f"Simulating {len(df)} steps...")
    with torch.no_grad():
        sim_data = sim(init_state, commands_tensor)
        
    # Convert to Numpy [TimeSteps, 6]
    sim_data = sim_data.squeeze(0).numpy()
    
    # Sync lengths (Simulation returns N+1 states usually)
    if sim_data.shape[0] > len(real_vx):
        sim_data = sim_data[:len(real_vx)]
        
    # Extract Simulation Data
    sim_x = sim_data[:, 0]
    sim_y = sim_data[:, 1]
    sim_vx = sim_data[:, 3]
    sim_vy = sim_data[:, 4]
    sim_omega = sim_data[:, 5]

    # --- PLOTTING ---
    plt.figure(figsize=(14, 10))

    # PLOT 1: Trajectory (Spatial)
    plt.subplot(2, 2, 1)
    plt.title(f"Trajectory Map (X vs Y)\nModel: {MODEL_PATH}")
    plt.plot(real_x, real_y, 'b-', label='Real Robot', linewidth=2, alpha=0.6)
    plt.plot(sim_x, sim_y, 'r--', label='Simulation', linewidth=2)
    plt.plot(real_x[0], real_y[0], 'go', label='Start') # Start Point
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal') # Essential for spatial plots
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 2: Velocity X (Surge)
    plt.subplot(2, 2, 2)
    plt.title("Body Velocity X (Surge)")
    plt.plot(real_vx, 'b-', label='Real', alpha=0.6)
    plt.plot(sim_vx, 'r--', label='Sim')
    plt.plot(cmd_vx, 'g:', label='Cmd', alpha=0.4)
    plt.ylabel("m/s")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 3: Velocity Y (Sway)
    plt.subplot(2, 2, 3)
    plt.title("Body Velocity Y (Sway)")
    plt.plot(real_vy, 'b-', label='Real', alpha=0.6)
    plt.plot(sim_vy, 'r--', label='Sim')
    plt.plot(cmd_vy, 'g:', label='Cmd', alpha=0.4)
    plt.ylabel("m/s")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 4: Angular Velocity (Yaw Rate)
    plt.subplot(2, 2, 4)
    plt.title("Angular Velocity (Omega)")
    plt.plot(real_omega, 'b-', label='Real', alpha=0.6)
    plt.plot(sim_omega, 'r--', label='Sim')
    plt.plot(cmd_w, 'g:', label='Cmd', alpha=0.4)
    plt.ylabel("rad/s")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- METRICS ---
    pos_error = np.sqrt((real_x[-1] - sim_x[-1])**2 + (real_y[-1] - sim_y[-1])**2)
    rmse_vx = np.sqrt(np.mean((real_vx - sim_vx)**2))
    rmse_vy = np.sqrt(np.mean((real_vy - sim_vy)**2))
    rmse_w  = np.sqrt(np.mean((real_omega - sim_omega)**2))

    print(f"\n--- VALIDATION METRICS ---")
    print(f"Final Position Drift: {pos_error:.4f} m")
    print(f"RMSE Vx:    {rmse_vx:.4f} m/s")
    print(f"RMSE Vy:    {rmse_vy:.4f} m/s")
    print(f"RMSE Omega: {rmse_w:.4f} rad/s")

if __name__ == "__main__":
    plot_full_results()