### Problem Statement: Robot System Simulation

The objective is to define a discrete-time dynamical system model $\mathcal{M}$ that predicts the future state of the robot $\mathbf{s}_{k+1}$ given its current state $\mathbf{s}_k$ and the control inputs $\mathbf{u}_k$.

#### 1. Variables and Spaces

* **Time Step:** $\Delta t = 0.016 \text{ s}$
* **State Vector ($\mathbf{s}_k \in \mathbb{R}^6$):** Defined in the **Global Frame** $\mathcal{F}_G$.
    $$
    \mathbf{s}_k = \begin{bmatrix} x_k \\ y_k \\ \theta_k \\ v_{x,k} \\ v_{y,k} \\ \omega_k \end{bmatrix}
    \begin{aligned}
    &\leftarrow \text{Global Position } (X) \\
    &\leftarrow \text{Global Position } (Y) \\
    &\leftarrow \text{Global Heading} \\
    &\leftarrow \text{Global Velocity } (X) \\
    &\leftarrow \text{Global Velocity } (Y) \\
    &\leftarrow \text{Angular Velocity}
    \end{aligned}
    $$

* **Control Input Vector ($\mathbf{u}_k \in \mathbb{R}^3$):** Defined in the **Local Frame** $\mathcal{F}_L$.
    $$
    \mathbf{u}_k = \begin{bmatrix} v_{x,k}^{cmd} \\ v_{y,k}^{cmd} \\ \omega_k^{cmd} \end{bmatrix}
    \begin{aligned}
    &\leftarrow \text{Commanded Surge Velocity} \\
    &\leftarrow \text{Commanded Sway Velocity} \\
    &\leftarrow \text{Commanded Yaw Rate}
    \end{aligned}
    $$

#### 2. System Dynamics

The simulator must implement the transition function $f$:
$$
\mathbf{s}_{k+1} = f(\mathbf{s}_k, \mathbf{u}_k)
$$