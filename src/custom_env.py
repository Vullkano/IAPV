import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8, n_walls=10, random_start=True):
        self.size = size
        self.n_walls = n_walls
        self.window_size = 512
        self.random_start = random_start
        self.fixed_start = np.array([0, 0])  # Fixed spawn position

        # Observation Space (8 values):
        # 0: Agent Row
        # 1: Agent Col
        # 2: Wall Up (0 or 1)
        # 3: Wall Down (0 or 1)
        # 4: Wall Left (0 or 1)
        # 5: Wall Right (0 or 1)
        # 6: Target Row - Agent Row (Relative Row)
        # 7: Target Col - Agent Col (Relative Col)
        
        # Ranges:
        # Agent Pos: [0, size-1]
        # Walls: [0, 1]
        # Rel Goal: [-(size-1), (size-1)]
        
        # Using Box for simplicity in standard RL libraries, though MultiDiscrete is stricter.
        # Box is safer for PPO MlpPolicy usually.
        low = np.array([0, 0, 0, 0, 0, 0, -size + 1, -size + 1])
        high = np.array([size - 1, size - 1, 1, 1, 1, 1, size - 1, size - 1])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right (Standard mapping attempt, but let's stick to vectors)
        # Let's use the fixed vectors from previous fix to ensure keys work:
        # Right (0) -> [0, 1]
        # Up (1) -> [-1, 0]
        # Left (2) -> [0, -1]
        # Down (3) -> [1, 0]
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),  # Right
            1: np.array([-1, 0]), # Up
            2: np.array([0, -1]), # Left
            3: np.array([1, 0]),  # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self._agent_location = None
        self._target_location = None
        self._obstacles = []

    def _get_obs(self):
        # Calculate Wall presence
        # Directions: Up, Down, Left, Right
        # We need to check if neighbor is out of bounds OR is an obstacle
        
        # Up (-1, 0)
        u_pos = self._agent_location + np.array([-1, 0])
        wall_u = 1 if not self._is_valid(u_pos) else 0
        
        # Down (1, 0)
        d_pos = self._agent_location + np.array([1, 0])
        wall_d = 1 if not self._is_valid(d_pos) else 0
        
        # Left (0, -1)
        l_pos = self._agent_location + np.array([0, -1])
        wall_l = 1 if not self._is_valid(l_pos) else 0
        
        # Right (0, 1)
        r_pos = self._agent_location + np.array([0, 1])
        wall_r = 1 if not self._is_valid(r_pos) else 0
        
        rel_target = self._target_location - self._agent_location
        
        obs = np.array([
            self._agent_location[0],
            self._agent_location[1],
            wall_u,
            wall_d,
            wall_l,
            wall_r,
            rel_target[0],
            rel_target[1]
        ], dtype=np.float32)
        
        return obs
    
    def _is_valid(self, pos):
        # Check bounds
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return False
        # Check obstacles
        if tuple(pos) in self._obstacles:
            return False
        return True

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate random target
        self._target_location = self.np_random.integers(0, self.size, size=2)
        
        # Generate random obstacles
        possible_locs = [(r, c) for r in range(self.size) for c in range(self.size)]
        possible_locs.remove(tuple(self._target_location))
        
        # Pick walls
        n_walls = min(self.n_walls, len(possible_locs) - 1) # Ensure at least 1 spot for agent
        wall_indices = self.np_random.choice(len(possible_locs), n_walls, replace=False)
        self._obstacles = set([possible_locs[i] for i in wall_indices]) # Set for fast lookup
        
        # Place Agent (FIXED or RANDOM based on mode)
        if self.random_start:
            # Random spawn: choose from valid positions
            valid_starts = [loc for loc in possible_locs if loc not in self._obstacles]
            if not valid_starts:
                # Should not happen with proper n_walls checks but safety fallback
                self._obstacles = set()
                valid_starts = [loc for loc in possible_locs]
                
            start_idx = self.np_random.choice(len(valid_starts))
            self._agent_location = np.array(valid_starts[start_idx])
        else:
            # Fixed spawn: always use fixed_start position
            # Ensure fixed position is not blocked by obstacle or target
            if tuple(self.fixed_start) in self._obstacles:
                self._obstacles.discard(tuple(self.fixed_start))
            if np.array_equal(self.fixed_start, self._target_location):
                # Regenerate target if it conflicts with fixed start
                valid_targets = [loc for loc in possible_locs if loc != tuple(self.fixed_start) and loc not in self._obstacles]
                if valid_targets:
                    self._target_location = np.array(valid_targets[self.np_random.choice(len(valid_targets))])
            
            self._agent_location = self.fixed_start.copy()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Cast action to int if numpy array
        if isinstance(action, np.ndarray):
            action = int(action)
            
        direction = self._action_to_direction[action]
        proposed_location = self._agent_location + direction
        
        if self._is_valid(proposed_location):
            self._agent_location = proposed_location
            
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else -0.05
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            grid = np.full((self.size, self.size), " . ", dtype=object)
            
            for obs in self._obstacles:
                grid[obs[0], obs[1]] = " # "
                
            grid[self._target_location[0], self._target_location[1]] = " G "
            grid[self._agent_location[0], self._agent_location[1]] = " A "
            
            print("\n" + "\n".join(["".join(row) for row in grid]))
            print("-" * (3 * self.size))

if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human", size=5, n_walls=3)
    obs, _ = env.reset()
    env.render()
    print("Obs:", obs)

