import os
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import TrajectoryDataset
from trainer import ModelTrainer
from models.hybrid_model import HybridModel
from models.physic_model import RobotDigitalTwin
from utils.losses import WrappedAngleMSE


df      = pd.read_csv("dataset/processed_data_sim/random_trajectory.csv")
test_df = pd.read_csv("dataset/processed_data_sim/random_trajectory.csv")

hparams = {
    'window_length': 15,
    'batch_size': 64,
    'target_length': 30,
    'learning_rate': 0.001,
    'epochs': 10,
    'curriculum': 0.8,
    'stride': 1,
}
# TO DO: Try stride > 1
dt = 0.016

train_dataset = TrajectoryDataset(
    df,
    stride=hparams['stride'],
    window_length=hparams['window_length'],
    target_length=hparams['target_length'],
)
test_dataset = TrajectoryDataset(
    test_df,
    window_length=hparams['window_length'],
    target_length=hparams['target_length'],
)

train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32,                    shuffle=False)

# --- Model: physics backbone + neural residual head ---
physics_model = RobotDigitalTwin(dt=dt)
model = HybridModel(physics_model=physics_model, dt=dt)

# Freeze physics: no gradient computed, no update applied.
for p in model.physics_model.parameters():
    p.requires_grad_(False)

# Optimizer only sees the trainable (CNN) parameters.
optimizer = torch.optim.Adam(model.cnn.parameters(), lr=hparams['learning_rate'])

curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir   = f"runs/hybrid_trajectory_{curr_time}"
tb_writer = SummaryWriter(log_dir=log_dir)

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    learning_rate=hparams['learning_rate'],
    criterion=WrappedAngleMSE(),
    device="cpu",
    writer=tb_writer,
)

train_history, val_history = trainer.fit(
    epochs=hparams['epochs'],
    curriculum_fraction=hparams['curriculum'],
)

best_val_loss  = min(val_history)
final_val_loss = val_history[-1]

tb_writer.add_hparams(
    hparam_dict=hparams,
    metric_dict={
        'hparam/best_val_loss':  best_val_loss,
        'hparam/final_val_loss': final_val_loss,
    },
)
tb_writer.close()



import matplotlib.pyplot as plt
from datetime import datetime

DT = 0.016
TRAJ_IDX = 0
model.eval()


@torch.no_grad()
def rollout_windowed(model, state_window, cmd_window, future_cmds):
    """
    Open-loop rollout with a sliding (W,) window — for models like HybridModel
    that look at the whole history each step.

    Args:
        state_window: (W, 6)
        cmd_window:   (W, 3)
        future_cmds:  (T, 3)
    Returns:
        states: (T + 1, 6), starting from state_window[-1].
    """
    state_seq = state_window.unsqueeze(0)   # (1, W, 6)
    cmd_seq   = cmd_window.unsqueeze(0)     # (1, W, 3)

    trajectory = [state_window[-1].unsqueeze(0)]   # list of (1, 6)
    for t in range(future_cmds.shape[0]):
        pred = model(state_seq, cmd_seq).as_tensor()         # (1, 6)
        trajectory.append(pred)
        state_seq = torch.cat([state_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
        next_cmd  = future_cmds[t].view(1, 1, 3)
        cmd_seq   = torch.cat([cmd_seq[:, 1:, :], next_cmd], dim=1)
    return torch.cat(trajectory, dim=0)                       # (T + 1, 6)


# --- Pull one windowed sample from the dataset ---
sample = test_dataset[TRAJ_IDX]
state_window, cmd_window, target_states, target_cmds = sample

def _to_tensor(x):
    return x.as_tensor() if hasattr(x, "as_tensor") else x

state_window  = _to_tensor(state_window)     # (W, 6)
cmd_window    = _to_tensor(cmd_window)       # (W, 3)
target_states = _to_tensor(target_states)    # (T, 6)
target_cmds   = _to_tensor(target_cmds)      # (T, 3)

predicted_states = rollout_windowed(model, state_window, cmd_window, target_cmds)
true_states      = torch.cat([state_window[-1].unsqueeze(0), target_states], dim=0)
t_axis           = torch.arange(true_states.shape[0]) * DT


# --- Plot (same layout, just relabeled) ---
curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
PRED_LABEL = 'hybrid model'

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.plot(true_states[:, 0],      true_states[:, 1],      label='ground truth', lw=2)
ax.plot(predicted_states[:, 0], predicted_states[:, 1], '--', label=PRED_LABEL, lw=2)
ax.scatter(true_states[0, 0], true_states[0, 1], c='k', s=40, zorder=5, label='start')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('Trajectory (global XY)')
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True); ax.legend()

ax = axes[0, 1]
ax.plot(t_axis, true_states[:, 2],      label='ground truth')
ax.plot(t_axis, predicted_states[:, 2], '--', label=PRED_LABEL)
ax.set_xlabel('time [s]'); ax.set_ylabel('θ [rad]')
ax.set_title('Heading')
ax.grid(True); ax.legend()

ax = axes[1, 0]
ax.plot(t_axis, true_states[:, 3],      label='vx truth')
ax.plot(t_axis, predicted_states[:, 3], '--', label='vx pred')
ax.plot(t_axis, true_states[:, 4],      label='vy truth')
ax.plot(t_axis, predicted_states[:, 4], '--', label='vy pred')
ax.set_xlabel('time [s]'); ax.set_ylabel('velocity [m/s]')
ax.set_title('Global linear velocities')
ax.grid(True); ax.legend()

ax = axes[1, 1]
ax.plot(t_axis, true_states[:, 5],      label='ground truth')
ax.plot(t_axis, predicted_states[:, 5], '--', label=PRED_LABEL)
ax.set_xlabel('time [s]'); ax.set_ylabel('ω [rad/s]')
ax.set_title('Angular velocity')
ax.grid(True); ax.legend()

fig.suptitle('Hybrid model — predicted vs real trajectory', fontsize=14)
fig.tight_layout()
plt.show()

os.makedirs("saved_models", exist_ok=True)
save_path = f"saved_models/hybrid_last_{curr_time}.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved model to {save_path}")

# Quick readout: did the physics parameters actually move?
print("Final physics parameters:")
print(f"  a_max:    {physics_model.a_max.detach().tolist()}")
print(f"  k_couple: {physics_model.k_couple.detach().tolist()}")