"""
Simple demo application showcasing AI Virtual Characters
This demo shows flocking behavior, pathfinding, and basic AI behaviors
"""

import pygame
import numpy as np
import sys
import time
from typing import List

# Add the src directory to Python path for imports
sys.path.append("src/python")

from common.math_utils import Vector3D, Vector2D, Environment, RandomUtils
from locomotion.steering_behaviors import FlockingAgent, SeekBehavior, FleeBehavior
from planning.pathfinding import GridWorld, AStarPathfinder, PlanningAgent


class VisualDemo:
    """Visual demonstration of AI Virtual Characters using pygame"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Virtual Characters Demo - IAPV")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # Demo mode
        self.demo_mode = "flocking"  # "flocking", "pathfinding", "seek_flee"
        
        # Initialize environment and agents
        self.environment = Environment()
        self.initialize_demo()
        
        # Font for UI
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Mouse interaction
        self.mouse_target = Vector3D()
        self.mouse_pressed = False
    
    def initialize_demo(self):
        """Initialize the current demo"""
        self.environment = Environment()
        
        if self.demo_mode == "flocking":
            self.initialize_flocking_demo()
        elif self.demo_mode == "pathfinding":
            self.initialize_pathfinding_demo()
        elif self.demo_mode == "seek_flee":
            self.initialize_seek_flee_demo()
    
    def initialize_flocking_demo(self):
        """Initialize flocking behavior demo"""
        # Create flocking agents
        for i in range(30):
            position = Vector3D(
                RandomUtils.uniform(100, self.width - 100),
                0,
                RandomUtils.uniform(100, self.height - 100)
            )
            agent = FlockingAgent(f"boid_{i}", position)
            agent.velocity = RandomUtils.random_vector3d(1.0, 3.0)
            self.environment.add_agent(agent)
    
    def initialize_pathfinding_demo(self):
        """Initialize pathfinding demo"""
        # Create grid world
        grid_width = self.width // 20
        grid_height = self.height // 20
        self.grid_world = GridWorld(grid_width, grid_height, 20.0)
        
        # Add some obstacles
        for i in range(20):
            x = RandomUtils.randint(1, grid_width - 2)
            y = RandomUtils.randint(1, grid_height - 2)
            # Create obstacle clusters
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= x + dx < grid_width and 0 <= y + dy < grid_height:
                        self.grid_world.set_walkable(x + dx, y + dy, False)
        
        # Create pathfinding agent
        self.pathfinder = AStarPathfinder(self.grid_world)
        agent = PlanningAgent("pathfinder", Vector3D(50, 0, 50))
        agent.set_pathfinder(self.pathfinder)
        self.environment.add_agent(agent)
    
    def initialize_seek_flee_demo(self):
        """Initialize seek/flee behavior demo"""
        # Create seeking agent
        seeker = FlockingAgent("seeker", Vector3D(100, 0, 100))
        seeker.steering_controller.clear_behaviors()
        seek_behavior = SeekBehavior(Vector3D(self.width//2, 0, self.height//2))
        seeker.steering_controller.add_behavior(seek_behavior)
        self.environment.add_agent(seeker)
        
        # Create fleeing agents
        for i in range(10):
            position = Vector3D(
                RandomUtils.uniform(200, self.width - 200),
                0,
                RandomUtils.uniform(200, self.height - 200)
            )
            fleer = FlockingAgent(f"fleer_{i}", position)
            fleer.steering_controller.clear_behaviors()
            flee_behavior = FleeBehavior(seeker.position)
            fleer.steering_controller.add_behavior(flee_behavior)
            self.environment.add_agent(fleer)
        
        self.seeker_agent = seeker
        self.flee_behaviors = []
        for agent in self.environment.get_all_agents():
            if agent.id.startswith("fleer_"):
                for behavior in agent.steering_controller.behaviors:
                    if isinstance(behavior, FleeBehavior):
                        self.flee_behaviors.append(behavior)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.demo_mode = "flocking"
                    self.initialize_demo()
                elif event.key == pygame.K_2:
                    self.demo_mode = "pathfinding"
                    self.initialize_demo()
                elif event.key == pygame.K_3:
                    self.demo_mode = "seek_flee"
                    self.initialize_demo()
                elif event.key == pygame.K_r:
                    self.initialize_demo()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_pressed = True
                    mouse_pos = pygame.mouse.get_pos()
                    self.mouse_target = Vector3D(mouse_pos[0], 0, mouse_pos[1])
                    self.handle_mouse_click()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    mouse_pos = pygame.mouse.get_pos()
                    self.mouse_target = Vector3D(mouse_pos[0], 0, mouse_pos[1])
                    self.handle_mouse_drag()
    
    def handle_mouse_click(self):
        """Handle mouse click based on current demo mode"""
        if self.demo_mode == "pathfinding":
            # Set new target for pathfinding agent
            agents = self.environment.get_all_agents()
            if agents:
                agent = agents[0]
                goal = Vector2D(self.mouse_target.x, self.mouse_target.z)
                if hasattr(agent, 'plan_path_to'):
                    agent.plan_path_to(goal)
        
        elif self.demo_mode == "seek_flee":
            # Update seek target
            if hasattr(self, 'seeker_agent'):
                for behavior in self.seeker_agent.steering_controller.behaviors:
                    if isinstance(behavior, SeekBehavior):
                        behavior.set_target(self.mouse_target)
    
    def handle_mouse_drag(self):
        """Handle mouse drag"""
        if self.demo_mode == "seek_flee":
            # Update seek target and flee threats
            if hasattr(self, 'seeker_agent'):
                for behavior in self.seeker_agent.steering_controller.behaviors:
                    if isinstance(behavior, SeekBehavior):
                        behavior.set_target(self.mouse_target)
                
                # Update flee behaviors
                for behavior in self.flee_behaviors:
                    behavior.set_threat(self.seeker_agent.position)
    
    def update(self, delta_time: float):
        """Update simulation"""
        if not self.paused:
            # Update flocking neighbors for flocking demo
            if self.demo_mode == "flocking":
                agents = self.environment.get_all_agents()
                for agent in agents:
                    if isinstance(agent, FlockingAgent):
                        neighbors = self.environment.get_agents_in_radius(agent.position, 50.0)
                        agent.set_neighbors(neighbors)
            
            # Update seek/flee behaviors
            elif self.demo_mode == "seek_flee":
                if hasattr(self, 'seeker_agent'):
                    for behavior in self.flee_behaviors:
                        behavior.set_threat(self.seeker_agent.position)
            
            # Update environment
            self.environment.update(delta_time)
            
            # Keep agents within bounds
            self.apply_boundary_constraints()
    
    def apply_boundary_constraints(self):
        """Keep agents within screen bounds"""
        margin = 50
        for agent in self.environment.get_all_agents():
            pos = agent.position
            vel = agent.velocity
            
            # Bounce off walls
            if pos.x < margin or pos.x > self.width - margin:
                vel.x *= -0.8
            if pos.z < margin or pos.z > self.height - margin:
                vel.z *= -0.8
            
            # Clamp position
            pos.x = max(margin, min(self.width - margin, pos.x))
            pos.z = max(margin, min(self.height - margin, pos.z))
            
            agent.set_position(pos)
            agent.set_velocity(vel)
    
    def draw(self):
        """Draw the simulation"""
        self.screen.fill(self.WHITE)
        
        # Draw based on demo mode
        if self.demo_mode == "flocking":
            self.draw_flocking()
        elif self.demo_mode == "pathfinding":
            self.draw_pathfinding()
        elif self.demo_mode == "seek_flee":
            self.draw_seek_flee()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_flocking(self):
        """Draw flocking demo"""
        agents = self.environment.get_all_agents()
        
        for agent in agents:
            pos = (int(agent.position.x), int(agent.position.z))
            
            # Draw agent as circle
            pygame.draw.circle(self.screen, self.BLUE, pos, 5)
            
            # Draw velocity vector
            if agent.velocity.magnitude() > 0:
                vel_end = (
                    int(agent.position.x + agent.velocity.x * 3),
                    int(agent.position.z + agent.velocity.z * 3)
                )
                pygame.draw.line(self.screen, self.RED, pos, vel_end, 2)
    
    def draw_pathfinding(self):
        """Draw pathfinding demo"""
        # Draw grid
        if hasattr(self, 'grid_world'):
            for y in range(self.grid_world.height):
                for x in range(self.grid_world.width):
                    rect = pygame.Rect(x * 20, y * 20, 20, 20)
                    if not self.grid_world.is_walkable(x, y):
                        pygame.draw.rect(self.screen, self.BLACK, rect)
                    pygame.draw.rect(self.screen, self.GRAY, rect, 1)
        
        # Draw agents and paths
        agents = self.environment.get_all_agents()
        for agent in agents:
            pos = (int(agent.position.x), int(agent.position.z))
            
            # Draw agent
            pygame.draw.circle(self.screen, self.GREEN, pos, 8)
            
            # Draw current plan
            if hasattr(agent, 'current_plan') and agent.current_plan:
                points = [(int(p.x), int(p.y)) for p in agent.current_plan]
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.RED, False, points, 3)
                
                # Draw waypoints
                for point in points:
                    pygame.draw.circle(self.screen, self.YELLOW, point, 3)
    
    def draw_seek_flee(self):
        """Draw seek/flee demo"""
        agents = self.environment.get_all_agents()
        
        for agent in agents:
            pos = (int(agent.position.x), int(agent.position.z))
            
            if agent.id == "seeker":
                # Draw seeker as larger green circle
                pygame.draw.circle(self.screen, self.GREEN, pos, 8)
                
                # Draw seek target
                if hasattr(self, 'mouse_target'):
                    target_pos = (int(self.mouse_target.x), int(self.mouse_target.z))
                    pygame.draw.circle(self.screen, self.YELLOW, target_pos, 6)
                    pygame.draw.line(self.screen, self.GREEN, pos, target_pos, 2)
            
            elif agent.id.startswith("fleer_"):
                # Draw fleers as red circles
                pygame.draw.circle(self.screen, self.RED, pos, 5)
                
                # Draw flee direction
                if agent.velocity.magnitude() > 0:
                    vel_end = (
                        int(agent.position.x + agent.velocity.x * 5),
                        int(agent.position.z + agent.velocity.z * 5)
                    )
                    pygame.draw.line(self.screen, self.RED, pos, vel_end, 1)
    
    def draw_ui(self):
        """Draw user interface"""
        # Title
        title = self.font.render("AI Virtual Characters Demo", True, self.BLACK)
        self.screen.blit(title, (10, 10))
        
        # Current mode
        mode_text = self.small_font.render(f"Mode: {self.demo_mode}", True, self.BLACK)
        self.screen.blit(mode_text, (10, 50))
        
        # Instructions
        instructions = [
            "1 - Flocking Demo",
            "2 - Pathfinding Demo", 
            "3 - Seek/Flee Demo",
            "R - Reset",
            "Space - Pause",
            "Mouse - Interact"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (10, 80 + i * 25))
        
        # Pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, self.RED)
            text_rect = pause_text.get_rect(center=(self.width//2, 50))
            self.screen.blit(pause_text, text_rect)
        
        # Agent count
        agent_count = len(self.environment.get_all_agents())
        count_text = self.small_font.render(f"Agents: {agent_count}", True, self.BLACK)
        self.screen.blit(count_text, (self.width - 120, 10))
    
    def run(self):
        """Main game loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Cap delta time to prevent large jumps
            delta_time = min(delta_time, 1.0 / 30.0)
            
            self.handle_events()
            self.update(delta_time)
            self.draw()
            
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()


if __name__ == "__main__":
    demo = VisualDemo()
    demo.run()