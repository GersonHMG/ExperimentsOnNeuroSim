import torch
from torch.utils.tensorboard import SummaryWriter  # <-- Add this import

# Your imports
from dataset.data_loader import get_trajectory_dataloaders
from torch.utils.data import Dataset, DataLoader, Subset
from models.SimpleMLP import SimpleMLP

from training.auto_regressive_trainer import AutoregressiveTrainer
from training.losses import WrappedDynamicsLoss

data_path = r"C:\Projects\Robocup\NeuroSimSSL\experiments\dataset\processed_data\data3.csv"

def main():
    # 1. Define hyperparameters
    horizon_steps = 10
    batch_size = 32
    epochs = 60
    learning_rate = 1e-3

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/noweight")

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

    # 5. Initialize the Trainer (Pass the writer here!)
    trainer = AutoregressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        horizon=horizon_steps,
        learning_rate=learning_rate,
        device=device,
        writer=writer  # <-- Pass the writer to the trainer
    )

    # 6. Run Training
    print("Starting training loop...")
    train_losses, val_losses = trainer.train(
        epochs=epochs, 
        save_path="simple_mlp_dynamics.pth"
    )
    
    # 7. Close the writer when finished
    writer.close()  # <-- Always close the writer!
    print("Pipeline finished successfully! Run 'tensorboard --logdir=runs' in your terminal to view logs.")

if __name__ == "__main__":
    main()