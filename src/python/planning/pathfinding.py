"""
Pathfinding and planning algorithms for AI Virtual Characters in Python
"""

import heapq
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Callable, Any
from enum import Enum
from dataclasses import dataclass

from ..common.math_utils import Vector2D, Vector3D, Agent


@dataclass
class Node:
    """Node for pathfinding algorithms"""
    position: Vector2D
    g_cost: float = 0.0  # Distance from start
    h_cost: float = 0.0  # Heuristic distance to goal
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


class GridWorld:
    """Simple grid-based world representation"""
    
    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.walkable = [[True for _ in range(width)] for _ in range(height)]
    
    def is_walkable(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.walkable[y][x]
        return False
    
    def set_walkable(self, x: int, y: int, walkable: bool) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.walkable[y][x] = walkable
    
    def grid_to_world(self, x: int, y: int) -> Vector2D:
        return Vector2D(x * self.cell_size, y * self.cell_size)
    
    def world_to_grid(self, position: Vector2D) -> Tuple[int, int]:
        return (int(position.x / self.cell_size), int(position.y / self.cell_size))
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-directional)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_walkable(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors


class AStarPathfinder:
    """A* pathfinding algorithm implementation"""
    
    def __init__(self, world: GridWorld):
        self.world = world
    
    def find_path(self, start: Vector2D, goal: Vector2D) -> List[Vector2D]:
        """Find path from start to goal using A* algorithm"""
        start_grid = self.world.world_to_grid(start)
        goal_grid = self.world.world_to_grid(goal)
        
        if not self.world.is_walkable(*start_grid) or not self.world.is_walkable(*goal_grid):
            return []  # No path possible
        
        open_set = []
        closed_set = set()
        nodes = {}
        
        # Create start node
        start_node = Node(self.world.grid_to_world(*start_grid))
        start_node.g_cost = 0
        start_node.h_cost = self._heuristic(start_node.position, self.world.grid_to_world(*goal_grid))
        
        heapq.heappush(open_set, start_node)
        nodes[start_grid] = start_node
        
        while open_set:
            current = heapq.heappop(open_set)
            current_grid = self.world.world_to_grid(current.position)
            
            if current_grid in closed_set:
                continue
            
            closed_set.add(current_grid)
            
            # Check if we reached the goal
            if current_grid == goal_grid:
                return self._reconstruct_path(current)
            
            # Check neighbors
            for nx, ny in self.world.get_neighbors(*current_grid):
                if (nx, ny) in closed_set:
                    continue
                
                # Create or get neighbor node
                if (nx, ny) not in nodes:
                    nodes[(nx, ny)] = Node(self.world.grid_to_world(nx, ny))
                
                neighbor = nodes[(nx, ny)]
                
                # Calculate tentative g cost
                tentative_g = current.g_cost + current.position.distance_to(neighbor.position)
                
                if neighbor.parent is None or tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self._heuristic(neighbor.position, self.world.grid_to_world(*goal_grid))
                    heapq.heappush(open_set, neighbor)
        
        return []  # No path found
    
    def _heuristic(self, a: Vector2D, b: Vector2D) -> float:
        """Euclidean distance heuristic"""
        return a.distance_to(b)
    
    def _reconstruct_path(self, node: Node) -> List[Vector2D]:
        """Reconstruct path from goal node to start"""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return list(reversed(path))


class BehaviorStatus(Enum):
    """Status of behavior tree nodes"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Base class for behavior tree nodes"""
    
    @abstractmethod
    def execute(self) -> BehaviorStatus:
        pass
    
    def reset(self) -> None:
        pass


class ActionNode(BehaviorNode):
    """Leaf node that executes an action"""
    
    def __init__(self, action: Callable[[], BehaviorStatus]):
        self.action = action
    
    def execute(self) -> BehaviorStatus:
        return self.action()


class ConditionNode(BehaviorNode):
    """Leaf node that checks a condition"""
    
    def __init__(self, condition: Callable[[], bool]):
        self.condition = condition
    
    def execute(self) -> BehaviorStatus:
        return BehaviorStatus.SUCCESS if self.condition() else BehaviorStatus.FAILURE


class SequenceNode(BehaviorNode):
    """Composite node that executes children in sequence"""
    
    def __init__(self):
        self.children: List[BehaviorNode] = []
        self.current_child = 0
    
    def add_child(self, child: BehaviorNode) -> None:
        self.children.append(child)
    
    def execute(self) -> BehaviorStatus:
        while self.current_child < len(self.children):
            status = self.children[self.current_child].execute()
            
            if status == BehaviorStatus.FAILURE:
                self.reset()
                return BehaviorStatus.FAILURE
            elif status == BehaviorStatus.RUNNING:
                return BehaviorStatus.RUNNING
            
            # Success, move to next child
            self.current_child += 1
        
        # All children succeeded
        self.reset()
        return BehaviorStatus.SUCCESS
    
    def reset(self) -> None:
        self.current_child = 0
        for child in self.children:
            child.reset()


class SelectorNode(BehaviorNode):
    """Composite node that tries children until one succeeds"""
    
    def __init__(self):
        self.children: List[BehaviorNode] = []
        self.current_child = 0
    
    def add_child(self, child: BehaviorNode) -> None:
        self.children.append(child)
    
    def execute(self) -> BehaviorStatus:
        while self.current_child < len(self.children):
            status = self.children[self.current_child].execute()
            
            if status == BehaviorStatus.SUCCESS:
                self.reset()
                return BehaviorStatus.SUCCESS
            elif status == BehaviorStatus.RUNNING:
                return BehaviorStatus.RUNNING
            
            # Failure, try next child
            self.current_child += 1
        
        # All children failed
        self.reset()
        return BehaviorStatus.FAILURE
    
    def reset(self) -> None:
        self.current_child = 0
        for child in self.children:
            child.reset()


class ParallelNode(BehaviorNode):
    """Composite node that executes all children simultaneously"""
    
    def __init__(self, success_policy: int = 1, failure_policy: int = 1):
        self.children: List[BehaviorNode] = []
        self.success_policy = success_policy  # How many children must succeed
        self.failure_policy = failure_policy  # How many children must fail
    
    def add_child(self, child: BehaviorNode) -> None:
        self.children.append(child)
    
    def execute(self) -> BehaviorStatus:
        success_count = 0
        failure_count = 0
        running_count = 0
        
        for child in self.children:
            status = child.execute()
            if status == BehaviorStatus.SUCCESS:
                success_count += 1
            elif status == BehaviorStatus.FAILURE:
                failure_count += 1
            else:
                running_count += 1
        
        if success_count >= self.success_policy:
            return BehaviorStatus.SUCCESS
        elif failure_count >= self.failure_policy:
            return BehaviorStatus.FAILURE
        else:
            return BehaviorStatus.RUNNING


class BehaviorTree:
    """Behavior tree for AI decision making"""
    
    def __init__(self, root: BehaviorNode):
        self.root = root
    
    def update(self) -> BehaviorStatus:
        return self.root.execute()
    
    def reset(self) -> None:
        self.root.reset()


class PlanningAgent(Agent):
    """Agent that uses behavior trees for planning"""
    
    def __init__(self, agent_id: str, position: Vector3D = None):
        super().__init__(agent_id, position)
        self.behavior_tree: Optional[BehaviorTree] = None
        self.current_plan: List[Vector2D] = []
        self.pathfinder: Optional[AStarPathfinder] = None
    
    def set_behavior_tree(self, tree: BehaviorTree) -> None:
        self.behavior_tree = tree
    
    def set_pathfinder(self, pathfinder: AStarPathfinder) -> None:
        self.pathfinder = pathfinder
    
    def plan_path_to(self, goal: Vector2D) -> bool:
        """Plan a path to the given goal"""
        if self.pathfinder is None:
            return False
        
        start = Vector2D(self.position.x, self.position.z)  # Use x,z for 2D planning
        path = self.pathfinder.find_path(start, goal)
        
        if path:
            self.current_plan = path
            self.set_memory("current_plan", path)
            return True
        
        return False
    
    def follow_plan(self, delta_time: float) -> None:
        """Follow the current plan"""
        if not self.current_plan:
            return
        
        # Simple plan following - move towards next waypoint
        target = self.current_plan[0]
        current_2d = Vector2D(self.position.x, self.position.z)
        
        if current_2d.distance_to(target) < 1.0:  # Reached waypoint
            self.current_plan.pop(0)
            if not self.current_plan:
                self.set_memory("plan_complete", True)
                return
        
        # Move towards target
        direction = target - current_2d
        if direction.magnitude() > 0:
            direction = direction.normalized()
            speed = 5.0
            
            # Update velocity (convert back to 3D)
            self.velocity = Vector3D(direction.x * speed, 0, direction.y * speed)
            
            # Update position
            new_position = self.position + self.velocity * delta_time
            self.set_position(new_position)
    
    def update(self, delta_time: float) -> None:
        # Execute behavior tree if available
        if self.behavior_tree:
            self.behavior_tree.update()
        
        # Follow current plan
        self.follow_plan(delta_time)


# Example behavior tree components for common AI behaviors
class PatrolBehavior:
    """Utility class for creating patrol behavior trees"""
    
    @staticmethod
    def create_patrol_tree(agent: Agent, waypoints: List[Vector2D]) -> BehaviorTree:
        """Create a behavior tree for patrolling waypoints"""
        
        def move_to_next_waypoint():
            current_waypoint = agent.get_memory("current_waypoint", 0)
            if current_waypoint < len(waypoints):
                target = waypoints[current_waypoint]
                current_pos = Vector2D(agent.position.x, agent.position.z)
                
                if current_pos.distance_to(target) < 1.0:
                    # Reached waypoint, move to next
                    next_waypoint = (current_waypoint + 1) % len(waypoints)
                    agent.set_memory("current_waypoint", next_waypoint)
                    return BehaviorStatus.SUCCESS
                else:
                    # Move towards waypoint
                    direction = target - current_pos
                    if direction.magnitude() > 0:
                        direction = direction.normalized()
                        agent.velocity = Vector3D(direction.x * 3.0, 0, direction.y * 3.0)
                    return BehaviorStatus.RUNNING
            
            return BehaviorStatus.FAILURE
        
        root = ActionNode(move_to_next_waypoint)
        return BehaviorTree(root)


class SearchBehavior:
    """Utility class for creating search behavior trees"""
    
    @staticmethod
    def create_search_tree(agent: Agent, search_area: Vector2D, radius: float) -> BehaviorTree:
        """Create a behavior tree for searching an area"""
        
        def search_randomly():
            # Generate random search points within radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius)
            
            search_point = Vector2D(
                search_area.x + distance * np.cos(angle),
                search_area.y + distance * np.sin(angle)
            )
            
            agent.set_memory("search_target", search_point)
            return BehaviorStatus.SUCCESS
        
        def move_to_search_point():
            target = agent.get_memory("search_target")
            if target is None:
                return BehaviorStatus.FAILURE
            
            current_pos = Vector2D(agent.position.x, agent.position.z)
            
            if current_pos.distance_to(target) < 1.0:
                return BehaviorStatus.SUCCESS
            
            direction = target - current_pos
            if direction.magnitude() > 0:
                direction = direction.normalized()
                agent.velocity = Vector3D(direction.x * 4.0, 0, direction.y * 4.0)
            
            return BehaviorStatus.RUNNING
        
        def search_complete():
            # Simple search completion check
            search_time = agent.get_memory("search_time", 0.0)
            search_time += 0.1  # Approximate delta time
            agent.set_memory("search_time", search_time)
            
            return BehaviorStatus.SUCCESS if search_time > 10.0 else BehaviorStatus.FAILURE
        
        # Create behavior tree structure
        sequence = SequenceNode()
        sequence.add_child(ActionNode(search_randomly))
        sequence.add_child(ActionNode(move_to_search_point))
        
        selector = SelectorNode()
        selector.add_child(ConditionNode(search_complete))
        selector.add_child(sequence)
        
        return BehaviorTree(selector)