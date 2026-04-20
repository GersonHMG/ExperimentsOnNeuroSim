import pandas as pd
from torch.utils.data import DataLoader
from dataset_loader import TrajectoryDataset
from trainer import ModelTrainer
from models.cnn_model import CNNModel
from utils.losses import WrappedAngleMSE
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import torch

df = pd.read_csv("dataset/processed_data_sim/random_trajectory.csv")
test_df = pd.read_csv("dataset/processed_data_sim/isolate_axis.csv")

hparams = {
    'window_length': 15,
    'batch_size': 32,
    'target_length': 20,
    'learning_rate': 0.001,
    'epochs': 20,
    'curriculum': 0.75,
    'note': 'Kinematic Model'
}
# TO DO: Try stride > 1
dt = 0.016
train_dataset = TrajectoryDataset(
    df, 
    window_length=hparams['window_length'], 
    target_length=hparams['target_length']
)

test_dataset = TrajectoryDataset(
    test_df, 
    window_length=hparams['window_length'], 
    target_length=hparams['target_length']
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=hparams['batch_size'], 
    shuffle=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False
)

curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f"runs/cnn_trajectory_{curr_time}"
tb_writer = SummaryWriter(log_dir=log_dir)
model = CNNModel(dt=dt)
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    learning_rate=hparams['learning_rate'],
    criterion=WrappedAngleMSE(),
    device="cpu",
    writer=tb_writer
)

train_history, val_history = trainer.fit(epochs=hparams['epochs'], curriculum_fraction=hparams['curriculum'])

# Logging
best_val_loss = min(val_history)
final_val_loss = val_history[-1]

# You can now pass the entire dictionary directly!
tb_writer.add_hparams(
    hparam_dict=hparams, 
    metric_dict={
        'hparam/best_val_loss': best_val_loss,
        'hparam/final_val_loss': final_val_loss
    }
)

tb_writer.close()


os.makedirs("saved_models", exist_ok=True)
save_path = f"saved_models/cnn_last_{curr_time}.pth"
torch.save(model.state_dict(), save_path)