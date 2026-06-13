import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset_loader import TrajectoryDataset
from trainer import ModelTrainer
from models.physic_model import RobotDigitalTwin
from utils.losses import WrappedAngleMSE


df = pd.read_csv("dataset/processed_data_sim/random_trajectory.csv")
test_df = pd.read_csv("dataset/processed_data_sim/yaw_movement.csv")

# Only data/window hyperparameters remain — the kinematic model has no
# learnable parameters, no optimizer, and no training schedule.
hparams = {
    'window_length': 15,
    'batch_size': 64,
    'target_length': 180,
    'stride': 15,
}

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

train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Pure-physics baseline: no parameters, dt is fixed inside the model.
model = RobotDigitalTwin()
model.eval()

# -----------------------------------------------------------------------------
# Autoregressive rollout vs ground truth — single test trajectory by index
# -----------------------------------------------------------------------------
DT = 0.016
TRAJ_IDX = 2   # which sample from test_dataset to visualize


@torch.no_grad()
def rollout(model, initial_state: torch.Tensor, cmd_sequence: torch.Tensor) -> torch.Tensor:
    """
    Open-loop rollout: feed the model its own predictions.

    Args:
        initial_state: (6,)
        cmd_sequence:  (T, 3)
    Returns:
        states: (T + 1, 6), starting from initial_state
    """
    state = initial_state.view(1, 1, 6)                       # (1, 1, 6)
    trajectory = [state.squeeze(1)]                           # list of (1, 6)
    for t in range(cmd_sequence.shape[0]):
        cmd = cmd_sequence[t].view(1, 1, 3)                   # (1, 1, 3)
        next_state = model(state, cmd).as_tensor()            # State -> (1, 6)
        trajectory.append(next_state)
        state = next_state.unsqueeze(1)                       # (1, 1, 6)
    return torch.cat(trajectory, dim=0)                       # (T + 1, 6)


sample = test_dataset[TRAJ_IDX]
state_window, cmd_window, target_states, target_cmds = sample

# Normalize: TrajectoryDataset may return tensors or State/Command NamedTuples.
def _to_tensor(x):
    return x.as_tensor() if hasattr(x, "as_tensor") else x

state_window  = _to_tensor(state_window)   # (W, 6)
cmd_window    = _to_tensor(cmd_window)     # (W, 3)
target_states = _to_tensor(target_states)  # (T, 6)
target_cmds   = _to_tensor(target_cmds)    # (T, 3)

initial_state = state_window[-1]                              # (6,)
cmd_sequence  = target_cmds                                   # (T, 3)


    
true_states = torch.cat([initial_state.unsqueeze(0), target_states], dim=0)  # (T+1, 6)
predicted_states = rollout(model, initial_state, cmd_sequence)               # (T+1, 6)

t_axis = torch.arange(true_states.shape[0]) * DT


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
curr_time = datetime.now().strftime('%b%d_%H-%M-%S')

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) XY trajectory
ax = axes[0, 0]
ax.plot(true_states[:, 0],      true_states[:, 1],      label='ground truth', lw=2)
ax.plot(predicted_states[:, 0], predicted_states[:, 1], '--', label='kinematic model', lw=2)
ax.scatter(true_states[0, 0], true_states[0, 1], c='k', s=40, zorder=5, label='start')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('Trajectory (global XY)')
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True); ax.legend()

# (b) Heading
ax = axes[0, 1]
ax.plot(t_axis, true_states[:, 2],      label='ground truth')
ax.plot(t_axis, predicted_states[:, 2], '--', label='kinematic model')
ax.set_xlabel('time [s]'); ax.set_ylabel('θ [rad]')
ax.set_title('Heading')
ax.grid(True); ax.legend()

# (c) Global linear velocities
ax = axes[1, 0]
ax.plot(t_axis, true_states[:, 3],      label='vx truth')
ax.plot(t_axis, predicted_states[:, 3], '--', label='vx pred')
ax.plot(t_axis, true_states[:, 4],      label='vy truth')
ax.plot(t_axis, predicted_states[:, 4], '--', label='vy pred')
ax.set_xlabel('time [s]'); ax.set_ylabel('velocity [m/s]')
ax.set_title('Global linear velocities')
ax.grid(True); ax.legend()

# (d) Angular velocity
ax = axes[1, 1]
ax.plot(t_axis, true_states[:, 5],      label='ground truth')
ax.plot(t_axis, predicted_states[:, 5], '--', label='kinematic model')
ax.set_xlabel('time [s]'); ax.set_ylabel('ω [rad/s]')
ax.set_title('Angular velocity')
ax.grid(True); ax.legend()

fig.suptitle('Kinematic baseline — predicted vs real trajectory', fontsize=14)
fig.tight_layout()

plt.show()