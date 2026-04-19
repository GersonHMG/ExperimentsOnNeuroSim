import pandas as pd
from torch.utils.data import DataLoader
from dataset_loader import TrajectoryDataset
from trainer import ModelTrainer
from models.cnn_model import CNNModel
from utils.losses import WrappedAngleMSE

df = pd.read_csv("dataset/processed_data_sim/random_trajectory.csv")
test_df = pd.read_csv("dataset/processed_data_sim/isolate_axis.csv")

window_length = 15
batch_size = 32
target_length = 5

# 1. Instantiate the Dataset with the FULL dataframe first.
# This ensures the internal 10-step sequences are still mathematically valid.
train_dataset = TrajectoryDataset(test_df, window_length=window_length, target_length=target_length)
test_dataset = TrajectoryDataset(df, window_length=window_length, target_length=target_length)


# 4. Create the DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False 
)

# 1. Initialize your specific model with the dt (time delta)
dt = 0.016  # Example timestep
model = CNNModel(dt=dt)

# 3. Initialize the trainer
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    learning_rate=0.001,
    criterion=WrappedAngleMSE(),
    device="cpu"
)

train_history, val_history = trainer.fit(epochs=5)

import matplotlib.pyplot as plt
# Plot the learning curves
plt.figure(figsize=(8, 5))
plt.plot(train_history, label="Training Loss", color="blue")
plt.plot(val_history, label="Validation Loss", color="orange")
plt.title("Model Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
