import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime
# Your imports
from dataset.data_loader import get_trajectory_dataloaders
from models.SimpleMLP import SimpleMLP
from training.auto_regressive_trainer import AutoregressiveTrainer
from training.losses import WrappedDynamicsLoss

# Make sure to import your evaluator! Adjust the path as needed.
from evaluation.evaluator import TrajectoryEvaluator

data_path = r"C:\Projects\Robocup\NeuroSimSSL\experiments\dataset\processed_data\data3.csv"

def main():
    # 1. Define hyperparameters
    horizon_steps = 10
    batch_size = 32
    epochs = 80
    learning_rate = 1e-3

    # Initialize TensorBoard Writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"runs/losses_experiment_{timestamp}"
    writer = SummaryWriter(log_dir=run_name)

    # 2. Load the Data
    print("Loading datasets...")
    train_loader, val_loader = get_trajectory_dataloaders(
        csv_file=data_path, 
        input_length=1,           
        horizon=horizon_steps, 
        batch_size=batch_size
    )

    # 3. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 4. Instantiate Model and Loss Function
    model = SimpleMLP(hidden_dim=64)
    loss_fn = WrappedDynamicsLoss() 

    # 5. Initialize the Trainer
    trainer = AutoregressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        horizon=horizon_steps,
        learning_rate=learning_rate,
        device=device,
        writer=writer 
    )

    # 6. Run Training
    print("Starting training loop...")
    train_losses, val_losses = trainer.train(
        epochs=epochs, 
        save_path="simple_mlp_dynamics.pth"
    )
    
    # ==========================================
    # II. FULL TRAJECTORY EVALUATION
    # ==========================================
    print("Running full trajectory evaluation...")
    
    # Load raw dataframe to extract a continuous chunk
    df = pd.read_csv(data_path)
    start_idx = len(df) // 2 
    eval_length = 100 
    
    # Extract Ground Truth States
    # Vector: [x, y, theta, vx, vy, omega]
    true_states = torch.tensor(
        df[['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']].iloc[start_idx : start_idx + eval_length + 1].values, 
        dtype=torch.float32
    )
    
    # Extract Local Commands
    commands = torch.tensor(
        df[['vx_cmd', 'vy_cmd', 'omega_cmd']].iloc[start_idx : start_idx + eval_length].values, 
        dtype=torch.float32
    )
    
    evaluator = TrajectoryEvaluator(model, device=device)
    
    # Run Autoregressive Rollout: s_{k+1} = f(s_k, u_k)
    predicted_states = evaluator.run_rollout(
        initial_state=true_states[0], 
        commands=commands
    )
    
    # 1. Calculate Mean Euclidean Distance Error (MEDE)
    # distance = sqrt((x_true - x_pred)^2 + (y_true - y_pred)^2)
    euclidean_distances = torch.norm(true_states[:, :2] - predicted_states[:, :2], dim=1)
    mede = torch.mean(euclidean_distances).item()
    
    # 2. Calculate Mean Absolute Circular Error (MACE)
    # We must wrap the angle difference to (-pi, pi]
    angle_diff = predicted_states[:, 2] - true_states[:, 2]
    wrapped_angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    mace = torch.mean(torch.abs(wrapped_angle_diff)).item()
    
    print(f"MEDE: {mede:.4f}m | MACE: {mace:.4f} rad")
    
    # Generate the figure and add to TensorBoard
    # Passing the new metrics to the title for better visibility in TensorBoard
    fig = evaluator.plot_trajectory(
        true_states, 
        predicted_states, 
        title=f"Rollout (MEDE: {mede:.3f}m, MACE: {mace:.3f}rad)"
    )
    writer.add_figure('Evaluation/Trajectory_Plot', fig, global_step=epochs)
    
    plt.close(fig)
    # ==========================================

    # Hyperparams
    hparam_dict = {
        'horizon_steps': horizon_steps,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'hidden_dim': 64  # Good to include model architecture constants too
    }

    # 2. Define the metrics dictionary (using the results from your evaluation)
    metric_dict = {
        'Evaluation/MEDE': mede,
        'Evaluation/MACE': mace,
        'Loss/Final_Val': val_losses[-1]
    }
    writer.add_scalar('Evaluation/MEDE', mede, global_step=epochs)
    writer.add_scalar('Evaluation/MACE', mace, global_step=epochs)
    writer.add_scalar('Loss/Final_Val', val_losses[-1], global_step=epochs)
    # 3. Log to the HParams dashboard
    writer.add_hparams(hparam_dict, metric_dict)

    # 8. Close the writer when finished
    writer.close() 
    print("Pipeline finished successfully! Run 'tensorboard --logdir=runs' in your terminal to view logs.")

if __name__ == "__main__":
    main()