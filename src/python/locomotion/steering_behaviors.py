"""
Steering behaviors and locomotion for AI Virtual Characters in Python
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum
import random

from ..common.math_utils import Agent, Vector3D, MathUtils, RandomUtils


class SteeringBehavior(ABC):
    """Base class for steering behaviors"""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.enabled = True
    
    @abstractmethod
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        """Calculate steering force for the agent"""
        pass


class SeekBehavior(SteeringBehavior):
    """Seek behavior - move towards a target"""
    
    def __init__(self, target: Vector3D, weight: float = 1.0):
        super().__init__(weight)
        self.target = target
    
    def set_target(self, target: Vector3D) -> None:
        self.target = target
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        desired = self.target - agent.position
        desired = desired.normalized() * 10.0  # Max speed
        return desired - agent.velocity


class FleeBehavior(SteeringBehavior):
    """Flee behavior - move away from a threat"""
    
    def __init__(self, threat: Vector3D, weight: float = 1.0):
        super().__init__(weight)
        self.threat = threat
    
    def set_threat(self, threat: Vector3D) -> None:
        self.threat = threat
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        desired = agent.position - self.threat
        desired = desired.normalized() * 10.0  # Max speed
        return desired - agent.velocity


class WanderBehavior(SteeringBehavior):
    """Wander behavior - random wandering movement"""
    
    def __init__(self, radius: float = 2.0, distance: float = 5.0, jitter: float = 1.0, weight: float = 1.0):
        super().__init__(weight)
        self.wander_radius = radius
        self.wander_distance = distance
        self.wander_jitter = jitter
        self.wander_target = Vector3D(0, 0, 1)
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        # Add random jitter to wander target
        self.wander_target = self.wander_target + Vector3D(
            RandomUtils.uniform(-self.wander_jitter, self.wander_jitter),
            0,
            RandomUtils.uniform(-self.wander_jitter, self.wander_jitter)
        )
        
        # Normalize to stay on circle
        self.wander_target = self.wander_target.normalized() * self.wander_radius
        
        # Calculate wander circle center
        velocity = agent.velocity
        if velocity.magnitude() > 0.1:
            velocity = velocity.normalized()
        else:
            velocity = Vector3D(0, 0, 1)  # Default forward
        
        circle_center = agent.position + velocity * self.wander_distance
        target = circle_center + self.wander_target
        
        # Seek toward the target
        desired = target - agent.position
        desired = desired.normalized() * 5.0  # Moderate speed for wandering
        
        return desired - agent.velocity


class SeparationBehavior(SteeringBehavior):
    """Separation behavior - avoid crowding neighbors"""
    
    def __init__(self, separation_radius: float = 3.0, weight: float = 1.0):
        super().__init__(weight)
        self.separation_radius = separation_radius
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        steer = Vector3D(0, 0, 0)
        count = 0
        
        for neighbor in neighbors:
            if neighbor.id == agent.id:
                continue
            
            distance = agent.position.distance_to(neighbor.position)
            if 0 < distance < self.separation_radius:
                diff = agent.position - neighbor.position
                diff = diff.normalized()
                diff = diff * (1.0 / distance)  # Weight by distance
                steer = steer + diff
                count += 1
        
        if count > 0:
            steer = steer / count
            steer = steer.normalized() * 10.0  # Max speed
            steer = steer - agent.velocity
        
        return steer


class AlignmentBehavior(SteeringBehavior):
    """Alignment behavior - align with neighbors' velocities"""
    
    def __init__(self, alignment_radius: float = 5.0, weight: float = 1.0):
        super().__init__(weight)
        self.alignment_radius = alignment_radius
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        velocity_sum = Vector3D(0, 0, 0)
        count = 0
        
        for neighbor in neighbors:
            if neighbor.id == agent.id:
                continue
            
            distance = agent.position.distance_to(neighbor.position)
            if 0 < distance < self.alignment_radius:
                velocity_sum = velocity_sum + neighbor.velocity
                count += 1
        
        if count > 0:
            velocity_sum = velocity_sum / count
            velocity_sum = velocity_sum.normalized() * 10.0  # Max speed
            return velocity_sum - agent.velocity
        
        return Vector3D(0, 0, 0)


class CohesionBehavior(SteeringBehavior):
    """Cohesion behavior - move towards center of neighbors"""
    
    def __init__(self, cohesion_radius: float = 8.0, weight: float = 1.0):
        super().__init__(weight)
        self.cohesion_radius = cohesion_radius
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        position_sum = Vector3D(0, 0, 0)
        count = 0
        
        for neighbor in neighbors:
            if neighbor.id == agent.id:
                continue
            
            distance = agent.position.distance_to(neighbor.position)
            if 0 < distance < self.cohesion_radius:
                position_sum = position_sum + neighbor.position
                count += 1
        
        if count > 0:
            center = position_sum / count
            desired = center - agent.position
            desired = desired.normalized() * 10.0  # Max speed
            return desired - agent.velocity
        
        return Vector3D(0, 0, 0)


class AvoidanceBehavior(SteeringBehavior):
    """Obstacle avoidance behavior"""
    
    def __init__(self, avoidance_radius: float = 4.0, weight: float = 1.0):
        super().__init__(weight)
        self.avoidance_radius = avoidance_radius
        self.obstacles = []  # List of (position, radius) tuples
    
    def add_obstacle(self, position: Vector3D, radius: float) -> None:
        self.obstacles.append((position, radius))
    
    def clear_obstacles(self) -> None:
        self.obstacles.clear()
    
    def calculate(self, agent: Agent, neighbors: List[Agent]) -> Vector3D:
        steer = Vector3D(0, 0, 0)
        
        # Avoid other agents
        for neighbor in neighbors:
            if neighbor.id == agent.id:
                continue
            
            distance = agent.position.distance_to(neighbor.position)
            if 0 < distance < self.avoidance_radius:
                diff = agent.position - neighbor.position
                diff = diff.normalized()
                diff = diff * (self.avoidance_radius / distance)
                steer = steer + diff
        
        # Avoid obstacles
        for obstacle_pos, obstacle_radius in self.obstacles:
            distance = agent.position.distance_to(obstacle_pos)
            total_radius = obstacle_radius + 1.0  # Agent radius
            
            if distance < total_radius + self.avoidance_radius:
                diff = agent.position - obstacle_pos
                diff = diff.normalized()
                diff = diff * (total_radius + self.avoidance_radius) / distance
                steer = steer + diff
        
        if steer.magnitude() > 0:
            steer = steer.normalized() * 10.0  # Max speed
            steer = steer - agent.velocity
        
        return steer


class SteeringController:
    """Controller that combines multiple steering behaviors"""
    
    def __init__(self, agent: Agent, max_speed: float = 10.0, max_force: float = 5.0):
        self.agent = agent
        self.max_speed = max_speed
        self.max_force = max_force
        self.behaviors: List[SteeringBehavior] = []
    
    def add_behavior(self, behavior: SteeringBehavior) -> None:
        self.behaviors.append(behavior)
    
    def remove_behavior(self, behavior: SteeringBehavior) -> None:
        if behavior in self.behaviors:
            self.behaviors.remove(behavior)
    
    def clear_behaviors(self) -> None:
        self.behaviors.clear()
    
    def update(self, delta_time: float, neighbors: List[Agent]) -> None:
        total_force = Vector3D(0, 0, 0)
        
        # Calculate steering forces from all behaviors
        for behavior in self.behaviors:
            if behavior.enabled:
                force = behavior.calculate(self.agent, neighbors)
                total_force = total_force + force * behavior.weight
        
        # Limit the steering force
        total_force = self._truncate(total_force, self.max_force)
        
        # Apply force to velocity
        new_velocity = self.agent.velocity + total_force * delta_time
        new_velocity = self._truncate(new_velocity, self.max_speed)
        
        self.agent.set_velocity(new_velocity)
        
        # Update position
        new_position = self.agent.position + new_velocity * delta_time
        self.agent.set_position(new_position)
    
    def _truncate(self, vector: Vector3D, max_length: float) -> Vector3D:
        magnitude = vector.magnitude()
        if magnitude > max_length:
            return vector.normalized() * max_length
        return vector


class AnimationType(Enum):
    """Animation types for movement"""
    WALK = "walk"
    RUN = "run"
    IDLE = "idle"
    TURN = "turn"


class MovementAnimator:
    """Handles animation state based on movement"""
    
    def __init__(self):
        self.current_animation = AnimationType.IDLE
        self.animation_time = 0.0
    
    def update(self, agent: Agent, delta_time: float) -> None:
        self.animation_time += delta_time
        
        new_animation = self._determine_animation(agent.velocity)
        
        if new_animation != self.current_animation:
            self.current_animation = new_animation
            self.animation_time = 0.0
            
            # Store animation state in agent memory
            agent.set_memory("animation_type", self.current_animation.value)
            agent.set_memory("animation_time", self.animation_time)
        
        agent.set_memory("animation_time", self.animation_time)
    
    def _determine_animation(self, velocity: Vector3D) -> AnimationType:
        speed = velocity.magnitude()
        
        if speed < 0.1:
            return AnimationType.IDLE
        elif speed < 5.0:
            return AnimationType.WALK
        else:
            return AnimationType.RUN
    
    def get_current_animation(self) -> AnimationType:
        return self.current_animation


class PathFollowing:
    """Utility for following a path of waypoints"""
    
    def __init__(self, waypoints: List[Vector3D], loop: bool = False):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.loop = loop
        self.arrival_radius = 1.0
    
    def get_steering_force(self, agent: Agent) -> Vector3D:
        if not self.waypoints or self.current_waypoint >= len(self.waypoints):
            return Vector3D(0, 0, 0)
        
        target = self.waypoints[self.current_waypoint]
        distance = agent.position.distance_to(target)
        
        # Check if we've reached the current waypoint
        if distance < self.arrival_radius:
            self.current_waypoint += 1
            
            # Handle looping
            if self.current_waypoint >= len(self.waypoints):
                if self.loop:
                    self.current_waypoint = 0
                else:
                    return Vector3D(0, 0, 0)  # Path completed
        
        # Seek towards current waypoint
        if self.current_waypoint < len(self.waypoints):
            target = self.waypoints[self.current_waypoint]
            desired = target - agent.position
            desired = desired.normalized() * 10.0  # Max speed
            return desired - agent.velocity
        
        return Vector3D(0, 0, 0)
    
    def is_complete(self) -> bool:
        return not self.loop and self.current_waypoint >= len(self.waypoints)
    
    def reset(self) -> None:
        self.current_waypoint = 0


class FlockingAgent(Agent):
    """Agent with built-in flocking behavior"""
    
    def __init__(self, agent_id: str, position: Vector3D = None):
        super().__init__(agent_id, position)
        self.steering_controller = SteeringController(self)
        
        # Add flocking behaviors
        self.steering_controller.add_behavior(SeparationBehavior(weight=1.5))
        self.steering_controller.add_behavior(AlignmentBehavior(weight=1.0))
        self.steering_controller.add_behavior(CohesionBehavior(weight=1.0))
        self.steering_controller.add_behavior(WanderBehavior(weight=0.1))
        
        self.animator = MovementAnimator()
    
    def update(self, delta_time: float) -> None:
        # This will be updated when neighbors are provided
        neighbors = self.get_memory("neighbors", [])
        self.steering_controller.update(delta_time, neighbors)
        self.animator.update(self, delta_time)
    
    def set_neighbors(self, neighbors: List[Agent]) -> None:
        self.set_memory("neighbors", neighbors)
    
    def add_steering_behavior(self, behavior: SteeringBehavior) -> None:
        self.steering_controller.add_behavior(behavior)
    
    def remove_steering_behavior(self, behavior: SteeringBehavior) -> None:
        self.steering_controller.remove_behavior(behavior)