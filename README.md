# NeuroSimSSL - Robot Dynamics Modeling

This project focuses on learning and modeling the dynamics of an omnidirectional robot for RoboCup SSL (Small Size League). It uses a hybrid approach combining a physics-based model with learnable parameters to predict robot trajectories.

## Problem Statement

Accurate prediction of robot movement is crucial for control and planning in dynamic environments like RoboCup. This project implements an **Autoregressive Dynamics Model** that predicts the future state of the robot given its current state and a sequence of control commands.

The model aims to minimize the error between the predicted trajectory (velocities and heading) and the ground truth recorded from the robot.

## Robot Description

The robot being modeled is a **4-wheeled Omnidirectional Robot**.

*   **Type**: Holonomic (Omnidirectional)
*   **Wheel Configuration**: 4 wheels positioned at angles:
    *   $60^\circ$
    *   $130^\circ$
    *   $230^\circ$
    *   $300^\circ$
*   **Physical Parameters** (Base):
    *   Mass: $2.8 \text{ kg}$
    *   Inertia: $0.2 \text{ kg} \cdot \text{m}^2$
    *   Radius ($L$): $0.2 \text{ m}$

### Physics Model
The core model (`OmniRobotDynamic`) is a differentiable physics engine. It simulates the forces acting on the robot based on motor commands and friction.
*   **Learnable Parameters**: The model learns specific physical coefficients during training:
    *   **Motor Gains**: Efficiency of signal-to-force conversion for each wheel.
    *   **Friction Coefficients**: Rotational friction acting on each wheel.

## Dataset

The data consists of recorded trajectories of the robot executing various movement patterns.

### Structure
*   **Location**: `dataset/processed_data/`
*   **Format**: CSV files.

### Features
The model uses a specific ordering of features for input and target sequences:

1.  **State Variables** (0-3):
    *   `vx`: Linear velocity in X (Global frame)
    *   `vy`: Linear velocity in Y (Global frame)
    *   `omega`: Angular velocity
    *   `theta`: Robot Heading (Orientation)

2.  **Control Commands** (4-6):
    *   `vx_cmd`: Commanded Linear X velocity
    *   `vy_cmd`: Commanded Linear Y velocity
    *   `omega_cmd`: Commanded Angular velocity
