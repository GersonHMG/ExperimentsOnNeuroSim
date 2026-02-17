# Dataset description

This dataset describes the discrete-time kinematic state, control, and estimation of a mobile robotic system. The system operates with a sampling interval of $\Delta t = 0.016$ seconds. A Kalman Filter is utilized as an estimator to process noisy measurements and compute smooth global velocities and position trajectories.

## 1. Coordinate Systems



* **Global Frame ($\mathcal{F}_G$):** The fixed inertial reference frame $(X, Y)$. Position states and calculated velocities are defined in this frame.
* **Local Frame ($\mathcal{F}_L$):** The body-fixed frame attached to the robot. Control commands are defined in this frame ($x$ forward, $y$ left).
* **Angle System:** Both the raw measurement ($\theta$) and the filtered estimate ($\hat{\theta}$) utilize a **signed angle system**. This implies that orientation is typically represented within the range $(-\pi, \pi]$ radians (or $[-180^\circ, 180^\circ)$), where the sign indicates the direction of rotation from the global $X$-axis.

## 2. State and Data Vectors

Let $k$ denote the discrete time step such that $t_k = k \cdot \Delta t$.

**A. Control Input Vector ($\mathbf{u}_k \in \mathcal{F}_L$)**
The velocity commands sent to the robot controller, defined in the local frame:
$$
\mathbf{u}_k = \begin{bmatrix} v_{x,k}^{cmd} \\ v_{y,k}^{cmd} \\ \omega_{k}^{cmd} \end{bmatrix}
$$
* **Columns:** `vx_cmd`, `vy_cmd`, `omega_cmd`

**B. Estimated Global Velocity Vector ($\hat{\mathbf{v}}_k \in \mathcal{F}_G$)**
The velocities calculated by the Kalman Filter to reduce noise, defined in the global frame:
$$
\hat{\mathbf{v}}_k = \begin{bmatrix} \hat{v}_{x,k} \\ \hat{v}_{y,k} \\ \hat{\omega}_{k} \end{bmatrix} \approx \begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix}
$$
* **Columns:** `vx`, `vy`, `omega`
* **Note:** Unlike the local commands, these linear velocities describe motion along the global axes.

**C. Filtered State Vector ($\hat{\mathbf{x}}_k \in \mathcal{F}_G$)**
The smoothed pose (position and orientation) estimate output by the Kalman Filter:
$$
\hat{\mathbf{x}}_k = \begin{bmatrix} \hat{x}_k \\ \hat{y}_k \\ \hat{\theta}_k \end{bmatrix}
$$
* **Columns:** `filtered_x`, `filtered_y`, `filtered_theta`
* **Orientation:** `filtered_theta` is stored as a signed angle.

**D. Raw Measurement Vector ($\mathbf{z}_k \in \mathcal{F}_G$)**
The noisy position and orientation observations:
$$
\mathbf{z}_k = \begin{bmatrix} x_k^{meas} \\ y_k^{meas} \\ \theta_k^{meas} \end{bmatrix}
$$
* **Columns:** `x`, `y`, `theta` (where `theta` is a signed angle).

## 3. Kinematic Relationships

The dataset captures the system dynamics governed by the following relationships:

**Integration (Position Update):**
The filtered position states are the time-integral of the estimated global velocities. Because $\hat{\theta}$ is a signed angle, updates should account for modular arithmetic (wrapping) to stay within the $(-\pi, \pi]$ range:
$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k-1} + \hat{\mathbf{v}}_k \cdot \Delta t
$$
$$
\begin{bmatrix} \hat{x}_k \\ \hat{y}_k \\ \hat{\theta}_k \end{bmatrix} \approx \begin{bmatrix} \hat{x}_{k-1} \\ \hat{y}_{k-1} \\ \hat{\theta}_{k-1} \end{bmatrix} + \begin{bmatrix} \hat{v}_{x,k} \\ \hat{v}_{y,k} \\ \hat{\omega}_{k} \end{bmatrix} \cdot 0.016
$$

**Frame Transformation (Command vs. Actual):**
While $\hat{\mathbf{v}}_k$ is in the global frame, it is driven by the local command $\mathbf{u}_k$. The relationship involves the rotation matrix $R(\hat{\theta}_k)$ which transforms the local control inputs into the global frame using the signed orientation:
$$
\begin{bmatrix} \hat{v}_{x,k} \\ \hat{v}_{y,k} \end{bmatrix} \approx R(\hat{\theta}_k) \begin{bmatrix} v_{x,k}^{cmd} \\ v_{y,k}^{cmd} \end{bmatrix} = \begin{bmatrix} \cos\hat{\theta}_k & -\sin\hat{\theta}_k \\ \sin\hat{\theta}_k & \cos\hat{\theta}_k \end{bmatrix} \begin{bmatrix} v_{x,k}^{cmd} \\ v_{y,k}^{cmd} \end{bmatrix}
$$