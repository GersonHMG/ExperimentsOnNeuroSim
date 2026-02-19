import torch
import matplotlib.pyplot as plt
import numpy as np

class TrajectoryEvaluator:
    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def run_rollout(self, initial_state: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        # Initial setup
        curr_state = initial_state.clone().detach().to(self.device)
        if curr_state.dim() == 1:
            curr_state = curr_state.unsqueeze(0)
            
        if hasattr(self.model, 'reset_hidden'):
            self.model.reset_hidden(batch_size=curr_state.shape[0], device=self.device)
            
        predictions = [curr_state.squeeze(0).cpu().detach()]
        
        with torch.no_grad():
            for t in range(commands.shape[0]):
                u_t = commands[t].clone().detach().unsqueeze(0).to(self.device)
                # s_{k+1} = f(s_k, u_k)
                curr_state = self.model(curr_state, u_t)
                predictions.append(curr_state.squeeze(0).cpu().detach())
                
        return torch.stack(predictions)

    def plot_trajectory(self, true_states: torch.Tensor, pred_states: torch.Tensor, 
                        title="Trajectory Rollout", angle_threshold_deg=15.0):
        """
        Plots X-Y path and Global Heading. 
        Adds orientation arrows when heading changes more than threshold.
        """
        true_np = true_states.numpy()
        pred_np = pred_states.numpy()
        threshold_rad = np.radians(angle_threshold_deg)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # --- Plot 1: X-Y Path with Orientation Quivers ---
        ax1.plot(true_np[:, 0], true_np[:, 1], label="Ground Truth", color='blue', alpha=0.5)
        ax1.plot(pred_np[:, 0], pred_np[:, 1], label="Predicted Path", color='red', linestyle='--', alpha=0.5)

        def add_quivers(states, color, label):
            x = states[:, 0]
            y = states[:, 1]
            theta = states[:, 2] # Global Heading
            
            # Filter indices based on angular change
            indices = [0]
            last_theta = theta[0]
            for i in range(1, len(theta)):
                # Use atan2 for shortest circular distance
                diff = np.abs(np.arctan2(np.sin(theta[i] - last_theta), np.cos(theta[i] - last_theta)))
                if diff >= threshold_rad:
                    indices.append(i)
                    last_theta = theta[i]
            
            # Unit vectors for orientation
            u = np.cos(theta[indices])
            v = np.sin(theta[indices])
            ax1.quiver(x[indices], y[indices], u, v, color=color, 
                       scale=20, width=0.005, label=f"{label} Orientation")

        add_quivers(true_np, 'blue', 'True')
        add_quivers(pred_np, 'red', 'Pred')

        # Start/End Markers
        ax1.scatter(true_np[0, 0], true_np[0, 1], color='green', marker='o', s=100, label='Start', zorder=5)
        ax1.scatter(true_np[-1, 0], true_np[-1, 1], color='black', marker='X', s=100, label='True End', zorder=5)
        ax1.scatter(pred_np[-1, 0], pred_np[-1, 1], color='darkred', marker='X', s=100, label='Pred End', zorder=5)
        
        ax1.set_title(f"Global X-Y Path (Arrows every {angle_threshold_deg}°)")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.legend(loc='best', fontsize='small', ncol=2)
        ax1.grid(True)
        ax1.axis('equal')

        # --- Plot 2: Global Heading (Theta) over Time ---
        time_steps = np.arange(true_np.shape[0])
        ax2.plot(time_steps, true_np[:, 2], label="True \u03B8", color='blue')
        ax2.plot(time_steps, pred_np[:, 2], label="Pred \u03B8", color='red', linestyle='--')
        
        ax2.set_title("Global Heading (\u03B8) in radians")
        ax2.set_xlabel("Step (k)")
        ax2.set_ylabel("Radians")
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig