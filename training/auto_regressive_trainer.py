import torch
import torch.optim as optim

class AutoregressiveTrainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, horizon=10, learning_rate=1e-3, device='cpu', writer=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.horizon = horizon
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.writer = writer  # <-- Store the writer
        self.train_losses = []
        self.val_losses = []

    def _rollout_horizon(self, seq_s, seq_u):
        loss = 0.0
        curr_state = seq_s[:, 0, :].to(self.device)
        
        # Reset hidden state if using a recurrent model
        if hasattr(self.model, 'reset_hidden'):
            self.model.reset_hidden(batch_size=curr_state.shape[0], device=self.device)
            
        for t in range(self.horizon):
            u_t = seq_u[:, t, :].to(self.device)
            target_state = seq_s[:, t+1, :].to(self.device)
            curr_state = self.model(curr_state, u_t)
            loss += self.loss_fn(curr_state, target_state)
            
        return loss / self.horizon

    def train(self, epochs, save_path="neural_dynamics.pth"):
        print(f"Starting autoregressive training on {self.device} for {epochs} epochs (Horizon: {self.horizon})...")
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        for epoch in range(epochs):
            # -- Training Phase --
            self.model.train()
            running_train_loss = 0.0
            for seq_s, seq_u in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._rollout_horizon(seq_s, seq_u)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * seq_s.size(0)
                
            epoch_train_loss = running_train_loss / train_size
            self.train_losses.append(epoch_train_loss)

            # -- Validation Phase --
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for seq_s, seq_u in self.val_loader:
                    loss = self._rollout_horizon(seq_s, seq_u)
                    running_val_loss += loss.item() * seq_s.size(0)
                    
            epoch_val_loss = running_val_loss / val_size
            self.val_losses.append(epoch_val_loss)

            # --> LOGGING TO TENSORBOARD <--
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Training complete. Model weights saved to {save_path}")
        return self.train_losses, self.val_losses