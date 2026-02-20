import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams  # <-- Bulletproof HParams fix import
from datetime import datetime

# Your imports
from dataset.data_loader import get_trajectory_dataloaders
from training.auto_regressive_trainer import AutoregressiveTrainer
from training.losses import WrappedDynamicsLoss
from evaluation.evaluator import TrajectoryEvaluator

# IMPORT THE NEW GRU MODEL
from models.rnn_model import RNNModel 

data_path = r"C:\Projects\Robocup\NeuroSimSSL\experiments\dataset\processed_data\data3.csv"

def main():
    # 1. Define hyperparameters
    horizon_steps = 10
    batch_size = 16
    epochs = 80
    learning_rate = 1e-3
    hidden_dim = 64
    num_layers = 1

    # Initialize TensorBoard Writer (Updated name for RNN)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"runs/gru_experiment_{timestamp}"
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

    # 4. Instantiate the RNN Model and Loss Function
    model = RNNModel(hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Optional: If you updated WrappedDynamicsLoss to set position weights to 0.0, 
    # it will automatically be used here.
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
    print("Starting GRU training loop...")
    train_losses, val_losses = trainer.train(
        epochs=epochs, 
        save_path="velocity_gru_dynamics.pth"  # <-- Saved with a new name
    )
    
    # ==========================================
    # II. FULL TRAJECTORY EVALUATION
    # ==========================================
    print("Running full trajectory evaluation...")
    
    df = pd.read_csv(data_path)
    start_idx = len(df) // 2 
    eval_length = 100 
    
    true_states = torch.tensor(
        df[['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']].iloc[start_idx : start_idx + eval_length + 1].values, 
        dtype=torch.float32
    )
    
    commands = torch.tensor(
        df[['vx_cmd', 'vy_cmd', 'omega_cmd']].iloc[start_idx : start_idx + eval_length].values, 
        dtype=torch.float32
    )
    
    evaluator = TrajectoryEvaluator(model, device=device)
    
    predicted_states = evaluator.run_rollout(
        initial_state=true_states[0], 
        commands=commands
    )
    
    euclidean_distances = torch.norm(true_states[:, :2] - predicted_states[:, :2], dim=1)
    mede = torch.mean(euclidean_distances).item()
    
    angle_diff = predicted_states[:, 2] - true_states[:, 2]
    wrapped_angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    mace = torch.mean(torch.abs(wrapped_angle_diff)).item()
    
    print(f"MEDE: {mede:.4f}m | MACE: {mace:.4f} rad")
    
    fig = evaluator.plot_trajectory(
        true_states, 
        predicted_states, 
        title=f"GRU Rollout (MEDE: {mede:.3f}m, MACE: {mace:.3f}rad)"
    )
    writer.add_figure('Evaluation/Trajectory_Plot', fig, global_step=epochs)
    plt.close(fig)
    
    # ==========================================
    # III. BULLETPROOF HPARAMS LOGGING
    # ==========================================
    hparam_dict = {
        'horizon_steps': horizon_steps,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'model_type': 'GRU'
    }

    metric_dict = {
        'Evaluation/MEDE': mede,
        'Evaluation/MACE': mace,
        'Loss/Final_Val': val_losses[-1]
    }
    
    # 1. Manually write the raw hparams summary directly to the file writer
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    # 2. Log the scalars so they match the table perfectly
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_step=epochs)

    # 8. Close the writer
    writer.close() 
    print("Pipeline finished successfully! Run 'tensorboard --logdir=runs' in your terminal to view logs.")

if __name__ == "__main__":
    main()