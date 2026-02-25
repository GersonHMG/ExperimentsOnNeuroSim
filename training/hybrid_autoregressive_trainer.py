import torch
import torch.optim as optim


class HybridAutoRegressiveTrainer:
    """
    Autoregressive trainer for hybrid physics + neural dynamics models.

    Unlike the standard AutoregressiveTrainer, this trainer calls
    model.forward_with_residual() at each rollout step so that the
    residual forces produced by the neural network branch are available
    for the DynamicSMAPELoss regularisation term.

    Compatible loss: training.HybridDynamicLoss.DynamicSMAPELoss
        forward(predictions, targets, residual_forces)
        -> (total_loss, state_loss, residual_penalty)
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        horizon: int = 10,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        writer=None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.horizon = horizon
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = writer
        self.train_losses = []
        self.val_losses = []

    # ------------------------------------------------------------------
    # Internal rollout helpers
    # ------------------------------------------------------------------

    def _rollout_horizon(self, seq_s: torch.Tensor, seq_u: torch.Tensor):
        """
        Roll out the model for self.horizon steps and accumulate the
        HybridDynamicLoss across all steps.

        Returns:
            avg_total_loss:    mean total loss over the horizon
            avg_state_loss:    mean state sMAPE over the horizon  (for logging)
            avg_residual_pen:  mean residual penalty over the horizon (for logging)
        """
        total_loss = 0.0
        total_state_loss = 0.0
        total_residual_pen = 0.0

        curr_state = seq_s[:, 0, :].to(self.device)

        # Reset recurrent hidden state at the start of every horizon rollout
        if hasattr(self.model, 'reset_hidden'):
            self.model.reset_hidden(batch_size=curr_state.shape[0], device=self.device)

        for t in range(self.horizon):
            u_t = seq_u[:, t, :].to(self.device)
            target_state = seq_s[:, t + 1, :].to(self.device)

            # forward_with_residual is the key difference from the standard trainer
            curr_state, residual_forces = self.model.forward_with_residual(curr_state, u_t)

            step_total, step_state, step_res = self.loss_fn(
                curr_state, target_state, residual_forces
            )

            total_loss += step_total
            total_state_loss += step_state
            total_residual_pen += step_res

        return (
            total_loss / self.horizon,
            total_state_loss / self.horizon,
            total_residual_pen / self.horizon,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, epochs: int, save_path: str = "neural_dynamics.pth"):
        print(
            f"Starting hybrid autoregressive training on {self.device} "
            f"for {epochs} epochs (Horizon: {self.horizon})..."
        )
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        for epoch in range(epochs):
            # ---- Training phase ----
            self.model.train()
            running_train_loss = 0.0
            running_train_state = 0.0
            running_train_res = 0.0

            for seq_s, seq_u in self.train_loader:
                self.optimizer.zero_grad()
                loss, state_loss, res_pen = self._rollout_horizon(seq_s, seq_u)
                loss.backward()
                self.optimizer.step()

                batch_n = seq_s.size(0)
                running_train_loss += loss.item() * batch_n
                running_train_state += state_loss.item() * batch_n
                running_train_res += res_pen.item() * batch_n

            epoch_train_loss = running_train_loss / train_size
            epoch_train_state = running_train_state / train_size
            epoch_train_res = running_train_res / train_size
            self.train_losses.append(epoch_train_loss)

            # ---- Validation phase ----
            self.model.eval()
            running_val_loss = 0.0
            running_val_state = 0.0
            running_val_res = 0.0

            with torch.no_grad():
                for seq_s, seq_u in self.val_loader:
                    loss, state_loss, res_pen = self._rollout_horizon(seq_s, seq_u)
                    batch_n = seq_s.size(0)
                    running_val_loss += loss.item() * batch_n
                    running_val_state += state_loss.item() * batch_n
                    running_val_res += res_pen.item() * batch_n

            epoch_val_loss = running_val_loss / val_size
            epoch_val_state = running_val_state / val_size
            epoch_val_res = running_val_res / val_size
            self.val_losses.append(epoch_val_loss)

            # ---- TensorBoard logging ----
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train_Total", epoch_train_loss, epoch)
                self.writer.add_scalar("Loss/Train_State", epoch_train_state, epoch)
                self.writer.add_scalar("Loss/Train_Residual_Penalty", epoch_train_res, epoch)
                self.writer.add_scalar("Loss/Val_Total", epoch_val_loss, epoch)
                self.writer.add_scalar("Loss/Val_State", epoch_val_state, epoch)
                self.writer.add_scalar("Loss/Val_Residual_Penalty", epoch_val_res, epoch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | "
                    f"Train Total: {epoch_train_loss:.6f}  "
                    f"(State: {epoch_train_state:.6f}, ResidualPen: {epoch_train_res:.6f}) | "
                    f"Val Total: {epoch_val_loss:.6f}"
                )

        torch.save(self.model.state_dict(), save_path)
        print(f"Training complete. Weights saved to '{save_path}'.")
        return self.train_losses, self.val_losses
