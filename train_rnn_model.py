import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset.data_loader import get_trajectory_dataloaders
from models.rnn_neural_dynamic import GrSimRNNDynamics
from training.hybrid_autoregressive_trainer import HybridAutoRegressiveTrainer
from training.HybridDynamicLoss import DynamicSMAPELoss
from evaluation.evaluator import TrajectoryEvaluator

data_path = r"C:\Projects\Robocup\NeuroSimSSL\experiments\dataset\processed_data_sim\random_trajectory.csv"
test_path = r"C:\Projects\Robocup\NeuroSimSSL\experiments\dataset\processed_data_sim\isolate_axis.csv"


def main():
    # =========================================
    # 1. Hyperparameters
    # =========================================
    horizon_steps   = 10
    batch_size      = 16
    epochs          = 50
    learning_rate   = 1e-3
    lambda_residual = 0.01   # Weight of residual-force regularisation in the loss
    hidden_dim      = 64     # GRU hidden state size
    num_layers      = 1      # Number of stacked GRU layers

    # =========================================
    # 2. TensorBoard Writer
    # =========================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f"runs/rnn_hybrid_{timestamp}"
    writer    = SummaryWriter(log_dir=run_name)

    # =========================================
    # 3. Data
    # =========================================
    print("Loading datasets...")
    train_loader, val_loader = get_trajectory_dataloaders(
        csv_file=data_path,
        input_length=1,
        horizon=horizon_steps,
        batch_size=batch_size,
    )

    # =========================================
    # 4. Device
    # =========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================
    # 5. Model & Loss
    # =========================================
    model   = GrSimRNNDynamics(hidden_dim=hidden_dim, num_layers=num_layers)
    loss_fn = DynamicSMAPELoss(lambda_residual=lambda_residual)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # =========================================
    # 6. Trainer
    # =========================================
    trainer = HybridAutoRegressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        horizon=horizon_steps,
        learning_rate=learning_rate,
        device=device,
        writer=writer,
    )

    # =========================================
    # 7. Training
    # =========================================
    print("Starting RNN hybrid training loop...")
    train_losses, val_losses = trainer.train(
        epochs=epochs,
        save_path="grsim_rnn_dynamics_network.pth",
    )

    # =========================================
    # 8. Full Trajectory Evaluation
    # =========================================
    print("Running full trajectory evaluation...")

    df          = pd.read_csv(test_path)
    start_idx   = len(df) // 2
    eval_length = 200

    true_states = torch.tensor(
        df[['filtered_x', 'filtered_y', 'filtered_theta', 'vx', 'vy', 'omega']]
            .iloc[start_idx : start_idx + eval_length + 1].values,
        dtype=torch.float32,
    )
    commands = torch.tensor(
        df[['vx_cmd', 'vy_cmd', 'omega_cmd']]
            .iloc[start_idx : start_idx + eval_length].values,
        dtype=torch.float32,
    )

    # TrajectoryEvaluator calls reset_hidden automatically when present
    evaluator        = TrajectoryEvaluator(model, device=device)
    predicted_states = evaluator.run_rollout(
        initial_state=true_states[0],
        commands=commands,
    )

    # Mean Euclidean Distance Error
    euclidean_distances = torch.norm(true_states[:, :2] - predicted_states[:, :2], dim=1)
    mede = torch.mean(euclidean_distances).item()

    # Mean Absolute Circular Error
    angle_diff         = predicted_states[:, 2] - true_states[:, 2]
    wrapped_angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    mace               = torch.mean(torch.abs(wrapped_angle_diff)).item()

    print(f"MEDE: {mede:.4f} m  |  MACE: {mace:.4f} rad")

    # Trajectory plot → TensorBoard
    fig = evaluator.plot_trajectory(
        true_states,
        predicted_states,
        title=f"RNN-Hybrid Rollout (MEDE: {mede:.3f} m, MACE: {mace:.3f} rad)",
    )
    writer.add_figure("Evaluation/Trajectory_Plot", fig, global_step=epochs)
    plt.close(fig)

    # =========================================
    # 9. Hyperparameter & Metric Logging
    # =========================================
    hparam_dict = {
        "horizon_steps":   horizon_steps,
        "batch_size":      batch_size,
        "epochs":          epochs,
        "learning_rate":   learning_rate,
        "lambda_residual": lambda_residual,
        "hidden_dim":      hidden_dim,
        "num_layers":      num_layers,
        "loss_fn":         "DynamicSMAPELoss",
        "model":           "GrSimRNNDynamics",
    }
    metric_dict = {
        "Evaluation/MEDE": mede,
        "Evaluation/MACE": mace,
    }

    writer.add_scalar("Evaluation/MEDE", mede, global_step=epochs)
    writer.add_scalar("Evaluation/MACE", mace, global_step=epochs)
    writer.add_hparams(hparam_dict, metric_dict, run_name=".")
    writer.close()

    print(
        "Pipeline finished. "
        "Run 'tensorboard --logdir=runs' to view logs."
    )


if __name__ == "__main__":
    main()
