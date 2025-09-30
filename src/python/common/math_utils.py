"""
Common utilities and base classes for AI Virtual Characters in Python
"""

import numpy as np
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class Vector2D:
    """2D vector with basic operations"""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalized(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector2D(0, 0)
    
    def dot(self, other: 'Vector2D') -> float:
        return self.x * other.x + self.y * other.y
    
    def distance_to(self, other: 'Vector2D') -> float:
        return (self - other).magnitude()
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Vector3D:
    """3D vector with basic operations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def normalized(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector3D(0, 0, 0)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def distance_to(self, other: 'Vector3D') -> float:
        return (self - other).magnitude()
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Agent(ABC):
    """Base class for all virtual characters/agents"""
    
    def __init__(self, agent_id: str, position: Vector3D = None):
        self.id = agent_id
        self.position = position or Vector3D()
        self.velocity = Vector3D()
        self.health = 100.0
        self.energy = 100.0
        self.memory: Dict[str, Any] = {}
        self.creation_time = time.time()
    
    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update agent state - must be implemented by subclasses"""
        pass
    
    def set_position(self, position: Vector3D) -> None:
        self.position = position
    
    def set_velocity(self, velocity: Vector3D) -> None:
        self.velocity = velocity
    
    def set_health(self, health: float) -> None:
        self.health = max(0.0, min(100.0, health))
    
    def set_energy(self, energy: float) -> None:
        self.energy = max(0.0, min(100.0, energy))
    
    def set_memory(self, key: str, value: Any) -> None:
        self.memory[key] = value
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)
    
    def has_memory(self, key: str) -> bool:
        return key in self.memory
    
    def get_age(self) -> float:
        """Get agent age in seconds"""
        return time.time() - self.creation_time


class Environment:
    """Environment class for managing world state"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.global_time = 0.0
    
    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[Agent]:
        return list(self.agents.values())
    
    def get_agents_in_radius(self, center: Vector3D, radius: float) -> List[Agent]:
        """Get all agents within a certain radius of a point"""
        nearby_agents = []
        for agent in self.agents.values():
            if agent.position.distance_to(center) <= radius:
                nearby_agents.append(agent)
        return nearby_agents
    
    def update(self, delta_time: float) -> None:
        self.global_time += delta_time
        for agent in self.agents.values():
            agent.update(delta_time)


class RandomUtils:
    """Utility class for random number generation"""
    
    @staticmethod
    def uniform(min_val: float, max_val: float) -> float:
        return random.uniform(min_val, max_val)
    
    @staticmethod
    def randint(min_val: int, max_val: int) -> int:
        return random.randint(min_val, max_val)
    
    @staticmethod
    def choice(items: List[Any]) -> Any:
        return random.choice(items)
    
    @staticmethod
    def random_vector2d(min_magnitude: float = 0.0, max_magnitude: float = 1.0) -> Vector2D:
        angle = random.uniform(0, 2 * np.pi)
        magnitude = random.uniform(min_magnitude, max_magnitude)
        return Vector2D(magnitude * np.cos(angle), magnitude * np.sin(angle))
    
    @staticmethod
    def random_vector3d(min_magnitude: float = 0.0, max_magnitude: float = 1.0) -> Vector3D:
        # Generate random direction using spherical coordinates
        theta = random.uniform(0, 2 * np.pi)  # azimuthal angle
        phi = random.uniform(0, np.pi)        # polar angle
        magnitude = random.uniform(min_magnitude, max_magnitude)
        
        sin_phi = np.sin(phi)
        return Vector3D(
            magnitude * sin_phi * np.cos(theta),
            magnitude * sin_phi * np.sin(theta),
            magnitude * np.cos(phi)
        )


class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b"""
        return a + (b - a) * t
    
    @staticmethod
    def lerp_vector3d(a: Vector3D, b: Vector3D, t: float) -> Vector3D:
        """Linear interpolation between two 3D vectors"""
        return Vector3D(
            MathUtils.lerp(a.x, b.x, t),
            MathUtils.lerp(a.y, b.y, t),
            MathUtils.lerp(a.z, b.z, t)
        )
    
    @staticmethod
    def angle_between_vectors(v1: Vector3D, v2: Vector3D) -> float:
        """Calculate angle between two vectors in radians"""
        dot_product = v1.normalized().dot(v2.normalized())
        dot_product = MathUtils.clamp(dot_product, -1.0, 1.0)
        return np.arccos(dot_product)
    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class Timer:
    """Simple timer utility"""
    
    def __init__(self, duration: float):
        self.duration = duration
        self.time_remaining = duration
        self.is_running = False
    
    def start(self) -> None:
        self.is_running = True
        self.time_remaining = self.duration
    
    def stop(self) -> None:
        self.is_running = False
    
    def reset(self) -> None:
        self.time_remaining = self.duration
    
    def update(self, delta_time: float) -> bool:
        """Update timer and return True if finished"""
        if self.is_running:
            self.time_remaining -= delta_time
            if self.time_remaining <= 0:
                self.is_running = False
                return True
        return False
    
    def is_finished(self) -> bool:
        return not self.is_running and self.time_remaining <= 0
    
    def get_progress(self) -> float:
        """Get timer progress from 0.0 to 1.0"""
        return 1.0 - (self.time_remaining / self.duration)


# Global utility functions
def distance_2d(a: Vector2D, b: Vector2D) -> float:
    return a.distance_to(b)

def distance_3d(a: Vector3D, b: Vector3D) -> float:
    return a.distance_to(b)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2*pi] range"""
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle