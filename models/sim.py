import pygame
import torch
import numpy as np
import math

# --- IMPORT YOUR ROBOT ---
from PhysicsModel import OmniRobotPhysics

# --- CONFIGURATION ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 200  # Zoom level (1 meter = 200 pixels)
FPS = 60

# Control Speeds
MAX_LIN_VEL = 0.5 # m/s
MAX_ANG_VEL = 3.5  # rad/s

# Colors
COLOR_BG = (30, 30, 30)
COLOR_ROBOT = (50, 200, 255)
COLOR_WHEEL = (160, 160, 160)  # Dark Grey/Black for wheels
COLOR_FRONT = (255, 50, 50)
COLOR_GRID = (50, 50, 50)

# Wheel Configuration (Must match the physics class)
# Angles: 60, 130, 230, 300
WHEEL_ANGLES_DEG = [60, 130, 230, 300]

def world_to_screen(x_world, y_world):
    """
    Convert physics coordinates (meters) to screen coordinates (pixels).
    """
    screen_x = int(SCREEN_WIDTH / 2 + x_world * PIXELS_PER_METER)
    screen_y = int(SCREEN_HEIGHT / 2 - y_world * PIXELS_PER_METER)
    return screen_x, screen_y

def draw_grid(surface):
    """Draws a 1-meter grid for reference."""
    for x in range(-10, 10):
        sx, _ = world_to_screen(x, 0)
        pygame.draw.line(surface, COLOR_GRID, (sx, 0), (sx, SCREEN_HEIGHT))
    for y in range(-10, 10):
        _, sy = world_to_screen(0, y)
        pygame.draw.line(surface, COLOR_GRID, (0, sy), (SCREEN_WIDTH, sy))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Omni-Robot Physics Simulator")
    clock = pygame.time.Clock()

    # --- INITIALIZE ROBOT ---
    device = 'cpu'
    sim = OmniRobotPhysics(device=device)
    #sim.load_state_dict(torch.load('models/trained_robot_params.pth', map_location=device))

    sim.eval()

    # Reset State: [x, y, theta, vx, vy, omega]
    state = torch.zeros((1, 6), device=device)
    
    # Dimensions for drawing
    robot_radius_m = sim.radius.item()
    robot_radius_px = int(robot_radius_m * PIXELS_PER_METER)
    
    # Create a surface for a single wheel (Rectangle)
    # Wheel dimensions approx: 6cm long, 2cm thick
    wheel_len_px = int(0.06 * PIXELS_PER_METER)
    wheel_width_px = int(0.02 * PIXELS_PER_METER)
    base_wheel_surf = pygame.Surface((wheel_len_px, wheel_width_px), pygame.SRCALPHA)
    base_wheel_surf.fill(COLOR_WHEEL)

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Input Handling
        keys = pygame.key.get_pressed()
        
        vx_cmd = 0.0
        vy_cmd = 0.0
        omega_cmd = 0.0

        if keys[pygame.K_w]: vx_cmd += MAX_LIN_VEL
        if keys[pygame.K_s]: vx_cmd -= MAX_LIN_VEL
        if keys[pygame.K_a]: vy_cmd += MAX_LIN_VEL
        if keys[pygame.K_d]: vy_cmd -= MAX_LIN_VEL
        if keys[pygame.K_q]: omega_cmd += MAX_ANG_VEL
        if keys[pygame.K_e]: omega_cmd -= MAX_ANG_VEL

        cmd = torch.tensor([[vx_cmd, vy_cmd, omega_cmd]], device=device)

        # 3. Physics Step
        state = sim.dynamics_step(state, cmd)

        # 4. Rendering
        screen.fill(COLOR_BG)
        draw_grid(screen)

        # Extract data
        robot_data = state[0].detach().numpy()
        rx, ry, theta = robot_data[0], robot_data[1], robot_data[2]
        
        # Calculate Robot Screen Center
        screen_x, screen_y = world_to_screen(rx, ry)

        # A. Draw Robot Body
        pygame.draw.circle(screen, COLOR_ROBOT, (screen_x, screen_y), robot_radius_px, width=2)

        # B. Draw Wheels
        # We need to calculate where each wheel is in the World Frame
        for angle_deg in WHEEL_ANGLES_DEG:
            angle_rad = math.radians(angle_deg)
            
            # 1. Calculate Wheel Position relative to robot center
            # The wheel is at angle (theta + wheel_angle) relative to global X
            total_angle = theta + angle_rad
            
            w_local_x = robot_radius_m * math.cos(total_angle)
            w_local_y = robot_radius_m * math.sin(total_angle)
            
            w_world_x = rx + w_local_x
            w_world_y = ry + w_local_y
            
            wx_screen, wy_screen = world_to_screen(w_world_x, w_world_y)
            
            # 2. Rotate the wheel sprite
            # The wheel is tangent to the circle.
            # Tangent angle = Radial Angle + 90 degrees
            tangent_angle = total_angle + (math.pi / 2)
            
            # Pygame rotation is in degrees and counter-clockwise is negative (usually) 
            # or positive depending on coord system. 
            # Math degrees (CCW) -> Pygame degrees (CCW is positive in transform.rotate)
            rotation_deg = math.degrees(tangent_angle)
            
            # Rotate surface
            rotated_wheel = pygame.transform.rotate(base_wheel_surf, rotation_deg)
            
            # 3. Blit centered at the calculated position
            rect = rotated_wheel.get_rect(center=(wx_screen, wy_screen))
            screen.blit(rotated_wheel, rect)

        # C. Draw Front Indicator
        end_x_world = rx + (robot_radius_m * math.cos(theta))
        end_y_world = ry + (robot_radius_m * math.sin(theta))
        end_screen_x, end_screen_y = world_to_screen(end_x_world, end_y_world)
        pygame.draw.line(screen, COLOR_FRONT, (screen_x, screen_y), (end_screen_x, end_screen_y), 3)
        
        # D. Draw Center of Mass (Yellow Dot)
        com_local = sim.com_offset.detach().numpy()
        com_world_x = rx + (com_local[0] * math.cos(theta) - com_local[1] * math.sin(theta))
        com_world_y = ry + (com_local[0] * math.sin(theta) + com_local[1] * math.cos(theta))
        cx, cy = world_to_screen(com_world_x, com_world_y)
        pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 4)

        # Debug Text
        font = pygame.font.SysFont("monospace", 16)
        text_vel = font.render(f"Vel: {robot_data[3]:.2f}, {robot_data[4]:.2f}", True, (255, 255, 255))
        text_pos = font.render(f"Pos: {rx:.2f}, {ry:.2f}", True, (255, 255, 255))
        screen.blit(text_vel, (10, 10))
        screen.blit(text_pos, (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()