"""
Pac-Man Simulation Game for RL Training Prep
Grid: configurable via config/game initialization
- Pac-Man: Heuristic AI (collect pellets, escape ghosts)
- Ghosts: Random movement (block each other)
- No user control - pure simulation
- Multiple auto-generated connected maps
"""

import pygame
import random
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict, Any

# ============================================================================
# MAP GENERATOR
# ============================================================================
class MapGenerator:
    """Generates random connected mazes for Pac-Man."""
    
    @staticmethod
    def generate_random_maze(grid_size: int = 30, 
                              wall_density: float = 0.15,
                              seed: int = None) -> Dict[str, Any]:
        """
        Generate a random connected maze.
        Uses: place random walls + ensure connectivity via BFS cleanup.
        """
        if seed is not None:
            random.seed(seed)
        
        walls = set()
        
        # Border walls
        for i in range(grid_size):
            walls.add((i, 0))
            walls.add((i, grid_size - 1))
            walls.add((0, i))
            walls.add((grid_size - 1, i))
        
        # Add random wall cells (but not border)
        num_walls = int((grid_size - 2) ** 2 * wall_density)
        attempts = 0
        while len(walls) < num_walls + 4 * grid_size - 4 and attempts < num_walls * 3:
            x = random.randint(1, grid_size - 2)
            y = random.randint(1, grid_size - 2)
            if (x, y) not in walls:
                walls.add((x, y))
            attempts += 1
        
        # Ensure connectivity - remove walls that block paths
        walls = MapGenerator._ensure_connectivity(walls, grid_size)
        
        # Generate a name
        seed_str = str(seed) if seed else "rand" + str(random.randint(1000, 9999))
        name = f"Map_{seed_str}"
        
        return {
            "name": name,
            "grid_size": grid_size,
            "walls": [list(w) for w in sorted(walls)],
            "seed": seed
        }
    
    @staticmethod
    def generate_maze_recursive(grid_size: int = 30, seed: int = None) -> Dict[str, Any]:
        """
        Generate maze using recursive backtracker (perfect maze - one path).
        More structured like classic games.
        """
        if seed is not None:
            random.seed(seed)
        
        # Start with all cells as walls
        walls = set()
        for x in range(grid_size):
            for y in range(grid_size):
                walls.add((x, y))
        
        # Pick random starting cell (must be odd coordinates for grid)
        start_x = random.randint(1, grid_size - 2)
        start_y = random.randint(1, grid_size - 2)
        
        # Make start open
        if (start_x, start_y) in walls:
            walls.remove((start_x, start_y))
        
        # Stack for backtracking
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        
        while stack:
            current = stack[-1]
            cx, cy = current
            
            # Find unvisited neighbors (2 steps away)
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if (0 < nx < grid_size - 1 and 0 < ny < grid_size - 1 and
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                # Open the wall between current and neighbor
                walls.discard((cx + dx // 2, cy + dy // 2))
                walls.discard((nx, ny))
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Add some extra random openings for variety (make it less perfect/more open)
        for _ in range(grid_size // 2):
            x = random.randint(1, grid_size - 2)
            y = random.randint(1, grid_size - 2)
            if (x, y) in walls:
                walls.discard((x, y))
        
        name = f"Maze_{seed}" if seed else f"Maze_{random.randint(1000, 9999)}"
        
        return {
            "name": name,
            "grid_size": grid_size,
            "walls": [list(w) for w in sorted(walls)],
            "seed": seed,
            "type": "recursive"
        }
    
    @staticmethod
    def generate_classic_style(grid_size: int = 30, seed: int = None) -> Dict[str, Any]:
        """Generate a Pac-Man classic style maze with corridors and obstacles."""
        if seed is not None:
            random.seed(seed)
        
        walls = set()
        
        # Border
        for i in range(grid_size):
            walls.add((i, 0))
            walls.add((i, grid_size - 1))
            walls.add((0, i))
            walls.add((grid_size - 1, i))
        
        center = grid_size // 2
        
        # Horizontal corridor dividers (with center gap)
        gap = 6
        for y in [grid_size // 4, grid_size * 3 // 4]:
            for x in range(2, center - gap // 2):
                walls.add((x, y))
                walls.add((grid_size - 1 - x, y))
        
        # Vertical corridor dividers
        for x in [grid_size // 4, grid_size * 3 // 4]:
            for y in range(2, center - gap // 2):
                walls.add((x, y))
                walls.add((x, grid_size - 1 - y))
        
        # Center blocks (4 small blocks around center)
        for offset in [-4, -3, 3, 4]:
            for y_offset in [-4, -3, 3, 4]:
                if abs(offset) == 4 or abs(y_offset) == 4:
                    walls.add((center + offset, center + y_offset))
        
        # Scattered small obstacles
        for _ in range(20):
            x = random.randint(3, grid_size - 4)
            y = random.randint(3, grid_size - 4)
            if (x, y) not in walls:
                walls.add((x, y))
        
        # Ensure connectivity
        walls = MapGenerator._ensure_connectivity(walls, grid_size)
        
        name = f"Classic_{seed}" if seed else f"Classic_{random.randint(1000, 9999)}"
        return {
            "name": name,
            "grid_size": grid_size,
            "walls": [list(w) for w in sorted(walls)],
            "seed": seed,
            "type": "classic"
        }
    
    @staticmethod
    def _ensure_connectivity(walls: Set[Tuple[int, int]], grid_size: int) -> Set[Tuple[int, int]]:
        """Ensure all non-wall cells are connected."""
        # Find starting cell
        start = None
        for x in range(1, grid_size - 1):
            for y in range(1, grid_size - 1):
                if (x, y) not in walls:
                    start = (x, y)
                    break
            if start:
                break
        
        if not start:
            return walls
        
        # BFS to find reachable
        reachable = set()
        queue = deque([start])
        reachable.add(start)
        
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (0 < nx < grid_size - 1 and 0 < ny < grid_size - 1 and
                    (nx, ny) not in walls and (nx, ny) not in reachable):
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
        
        # Clear paths to unreachable cells
        for x in range(1, grid_size - 1):
            for y in range(1, grid_size - 1):
                if (x, y) in walls or (x, y) in reachable:
                    continue
                # Clear this cell and nearby
                walls.discard((x, y))
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 < nx < grid_size - 1 and 0 < ny < grid_size - 1:
                        walls.discard((nx, ny))
        
        return walls
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> Set[Tuple[int, int]]:
        """Load walls set from map dict."""
        return {tuple(w) for w in data["walls"]}
    
    @staticmethod
    def save_map(map_data: Dict[str, Any], filepath: str):
        """Save map to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
    
    @staticmethod
    def load_map(filepath: str) -> Dict[str, Any]:
        """Load map from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_GRID_SIZE = 30
BASE_WINDOW_SIZE = 750
MIN_CELL_SIZE = 8
MAX_CELL_SIZE = 25

GRID_SIZE = DEFAULT_GRID_SIZE
CELL_SIZE = MAX_CELL_SIZE
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
POPUP_WIDTH = 400
POPUP_HEIGHT = 200


def configure_grid(grid_size: int):
    """Configure module-level grid metrics for training and rendering."""
    global GRID_SIZE, CELL_SIZE, WINDOW_SIZE

    grid_size = int(grid_size)
    if grid_size < 10:
        raise ValueError("grid_size must be at least 10")

    GRID_SIZE = grid_size
    CELL_SIZE = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, BASE_WINDOW_SIZE // GRID_SIZE))
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
DARK_BLUE = (0, 0, 40)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PINK = (255, 105, 180)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
PELLET_COLOR = (255, 200, 100)

# Game settings
FPS = 10  # Slow enough to watch
NUM_GHOSTS = 4
GHOST_ESCAPE_DISTANCE = 5  # Manhattan distance to trigger escape
GHOST_DANGER_DISTANCE = 3  # Very close - emergency escape

# Win conditions (configurable)
DEFAULT_MAX_PELLETS = 100  # Default pellet count
DEFAULT_TIME_LIMIT = 20.0  # Seconds for Pac-Man to win by time

# Movement directions
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Down, Up, Right, Left


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_valid_position(pos: Tuple[int, int], walls: Set[Tuple[int, int]], grid_size: int = None) -> bool:
    """Check if position is valid (in grid and not a wall)."""
    x, y = pos
    if grid_size is None:
        grid_size = GRID_SIZE
    return (0 <= x < grid_size and 
            0 <= y < grid_size and 
            pos not in walls)


def bfs_path(start: Tuple[int, int], 
             goal: Tuple[int, int], 
             walls: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """BFS to find shortest path from start to goal. Returns path or None."""
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for dx, dy in DIRECTIONS:
            next_pos = (current[0] + dx, current[1] + dy)
            
            if next_pos == goal:
                return path + [next_pos]
            
            if next_pos not in visited and is_valid_position(next_pos, walls):
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    
    return None  # No path found


def get_escape_direction(pacman_pos: Tuple[int, int],
                         ghost_positions: List[Tuple[int, int]],
                         walls: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Find best escape direction from ghosts using vector repulsion."""
    if not ghost_positions:
        return None
    
    # Calculate escape vector (away from all ghosts)
    escape_x, escape_y = 0, 0
    
    for gx, gy in ghost_positions:
        dx = pacman_pos[0] - gx
        dy = pacman_pos[1] - gy
        dist = max(1, manhattan_distance(pacman_pos, (gx, gy)))
        
        # Inverse square law for repulsion strength
        strength = 1.0 / (dist * dist)
        escape_x += dx * strength
        escape_y += dy * strength
    
    # Try directions, prefer ones that move along escape vector
    best_dir = None
    best_score = float('-inf')
    
    for dx, dy in DIRECTIONS:
        next_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)
        
        if not is_valid_position(next_pos, walls):
            continue
        
        # Score based on alignment with escape vector
        score = dx * escape_x + dy * escape_y
        
        # Bonus for increasing distance from closest ghost
        min_ghost_dist = min(manhattan_distance(next_pos, g) for g in ghost_positions)
        score += min_ghost_dist * 0.5
        
        if score > best_score:
            best_score = score
            best_dir = (dx, dy)
    
    return best_dir


# ============================================================================
# GAME OBJECTS
# ============================================================================
@dataclass
class Pellet:
    """A pellet that Pac-Man can collect."""
    pos: Tuple[int, int]
    active: bool = True


class Ghost:
    """Ghost enemy that moves randomly."""
    
    COLORS = [RED, CYAN, ORANGE, PINK]
    
    def __init__(self, pos: Tuple[int, int], ghost_id: int, walls: Set[Tuple[int, int]], grid_size: int = None):
        self.pos = pos
        self.id = ghost_id
        self.color = self.COLORS[ghost_id % len(self.COLORS)]
        self.walls = walls
        self.grid_size = grid_size if grid_size is not None else GRID_SIZE
        self.direction = random.choice(DIRECTIONS)
        self.move_cooldown = 0
    
    def move(self, other_ghost_positions: Set[Tuple[int, int]]):
        """Move ghost randomly, blocking other ghosts."""
        # Randomly change direction sometimes
        if random.random() < 0.2:
            self.direction = random.choice(DIRECTIONS)
        
        # Try to move in current direction
        new_pos = (self.pos[0] + self.direction[0], 
                   self.pos[1] + self.direction[1])
        
        if is_valid_position(new_pos, self.walls, self.grid_size) and new_pos not in other_ghost_positions:
            self.pos = new_pos
        else:
            # Hit wall or other ghost, pick new random direction
            valid_directions = []
            for dx, dy in DIRECTIONS:
                test_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if is_valid_position(test_pos, self.walls, self.grid_size) and test_pos not in other_ghost_positions:
                    valid_directions.append((dx, dy))
            
            if valid_directions:
                self.direction = random.choice(valid_directions)
                new_pos = (self.pos[0] + self.direction[0], 
                           self.pos[1] + self.direction[1])
                self.pos = new_pos
    
    def draw(self, surface: pygame.Surface):
        """Draw ghost on surface."""
        center_x = self.pos[0] * CELL_SIZE + CELL_SIZE // 2
        center_y = self.pos[1] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        # Body (circle)
        pygame.draw.circle(surface, self.color, (center_x, center_y), radius)
        
        # Eyes (white)
        eye_offset = radius // 3
        pygame.draw.circle(surface, WHITE, 
                          (center_x - eye_offset, center_y - eye_offset), 
                          radius // 4)
        pygame.draw.circle(surface, WHITE, 
                          (center_x + eye_offset, center_y - eye_offset), 
                          radius // 4)
        
        # Pupils
        pygame.draw.circle(surface, BLACK, 
                          (center_x - eye_offset, center_y - eye_offset), 
                          radius // 8)
        pygame.draw.circle(surface, BLACK, 
                          (center_x + eye_offset, center_y - eye_offset), 
                          radius // 8)


class PacMan:
    """Pac-Man character with heuristic AI."""
    
    def __init__(self, pos: Tuple[int, int], walls: Set[Tuple[int, int]], grid_size: int = None):
        self.pos = pos
        self.walls = walls
        self.grid_size = grid_size if grid_size is not None else GRID_SIZE
        self.score = 0
        self.pellets_collected = 0
    
    def get_closest_pellet(self, pellets: List[Pellet]) -> Optional[Pellet]:
        """Find the closest active pellet."""
        active_pellets = [p for p in pellets if p.active]
        if not active_pellets:
            return None
        
        return min(active_pellets, 
                   key=lambda p: manhattan_distance(self.pos, p.pos))
    
    def get_nearby_ghosts(self, ghosts: List[Ghost], 
                          max_dist: int = GHOST_ESCAPE_DISTANCE) -> List[Tuple[int, int]]:
        """Get positions of ghosts within range."""
        nearby = []
        for ghost in ghosts:
            if manhattan_distance(self.pos, ghost.pos) <= max_dist:
                nearby.append(ghost.pos)
        return nearby
    
    def get_next_move(self, pellets: List[Pellet], 
                      ghosts: List[Ghost]) -> Optional[Tuple[int, int]]:
        """
        Decide next move using heuristic:
        1. If danger close (GHOST_DANGER_DISTANCE), emergency escape
        2. If ghosts nearby, escape mode
        3. Otherwise, navigate to nearest pellet
        """
        ghost_positions = [g.pos for g in ghosts]
        
        # Check for danger
        danger_ghosts = self.get_nearby_ghosts(ghosts, GHOST_DANGER_DISTANCE)
        nearby_ghosts = self.get_nearby_ghosts(ghosts, GHOST_ESCAPE_DISTANCE)
        
        # Priority 1: Emergency escape (ghost very close)
        if danger_ghosts:
            escape_dir = get_escape_direction(self.pos, danger_ghosts, self.walls)
            if escape_dir:
                new_pos = (self.pos[0] + escape_dir[0], 
                           self.pos[1] + escape_dir[1])
                if is_valid_position(new_pos, self.walls, self.grid_size):
                    return new_pos
        
        # Priority 2: Escape mode
        if nearby_ghosts:
            escape_dir = get_escape_direction(self.pos, nearby_ghosts, self.walls)
            if escape_dir:
                new_pos = (self.pos[0] + escape_dir[0], 
                           self.pos[1] + escape_dir[1])
                if is_valid_position(new_pos, self.walls, self.grid_size):
                    return new_pos
        
        # Priority 3: Collect pellets
        closest_pellet = self.get_closest_pellet(pellets)
        if closest_pellet:
            path = bfs_path(self.pos, closest_pellet.pos, self.walls)
            if path and len(path) > 1:
                return path[1]  # Next step in path
        
        # No pellets left or no path, try random move
        valid_moves = []
        for dx, dy in DIRECTIONS:
            new_pos = (self.pos[0] + dx, self.pos[1] + dy)
            if is_valid_position(new_pos, self.walls, self.grid_size):
                valid_moves.append(new_pos)
        
        if valid_moves:
            return random.choice(valid_moves)
        
        return None
    
    def move(self, new_pos: Tuple[int, int]):
        """Move to new position."""
        self.pos = new_pos
    
    def collect_pellet(self, pellet: Pellet):
        """Collect a pellet."""
        if pellet.active:
            pellet.active = False
            self.pellets_collected += 1
            self.score += 10
    
    def draw(self, surface: pygame.Surface):
        """Draw Pac-Man on surface."""
        center_x = self.pos[0] * CELL_SIZE + CELL_SIZE // 2
        center_y = self.pos[1] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        # Pac-Man body (yellow circle with mouth)
        pygame.draw.circle(surface, YELLOW, (center_x, center_y), radius)
        
        # Simple mouth (arc)
        pygame.draw.polygon(surface, BLACK, [
            (center_x, center_y),
            (center_x + radius, center_y - radius // 2),
            (center_x + radius, center_y + radius // 2)
        ])


# ============================================================================
# GAME CLASS
# ============================================================================
class PacManGame:
    """Main game class managing all game state."""
    
    def __init__(self, max_pellets: int = None, time_limit: float = None, grid_size: int = None):
        self.grid_size = int(grid_size) if grid_size is not None else GRID_SIZE
        configure_grid(self.grid_size)
        self.cell_size = CELL_SIZE
        self.window_size = WINDOW_SIZE

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 40))
        pygame.display.set_caption("Pac-Man RL Training Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 48)
        
        # Win condition settings
        self.max_pellets = max_pellets if max_pellets is not None else DEFAULT_MAX_PELLETS
        self.time_limit = time_limit if time_limit is not None else DEFAULT_TIME_LIMIT
        
        # Generate multiple maps
        self.maps = []
        self._generate_maps()
        self.current_map_idx = 0
        
        # Menu state
        self.in_menu = True
        
        # Initialize game (will be set after map selection)
        self.walls = set()
        self.pacman = None
        self.ghosts = []
        self.pellets = []
        self.game_over = False
        self.win = False
        self.frame = 0
        self.score = 0
        self.start_time = 0  # For timer (ms)
        self.elapsed_time = 0.0  # Seconds
    
    def _generate_maps(self, num_maps: int = 6):
        """Generate a variety of maps."""
        print("Generating maps...")
        # Mix of map types
        generators = [
            ("Random", lambda s: MapGenerator.generate_random_maze(self.grid_size, 0.12, s)),
            ("Maze", lambda s: MapGenerator.generate_maze_recursive(self.grid_size, s)),
            ("Classic", lambda s: MapGenerator.generate_classic_style(self.grid_size, s)),
        ]
        
        for i in range(num_maps):
            gen_type, gen_fn = generators[i % len(generators)]
            seed = i * 100 + random.randint(1, 99)
            map_data = gen_fn(seed)
            map_data["display_name"] = f"{i+1}. {gen_type} #{i//3 + 1}"
            self.maps.append(map_data)
        
        print(f"Generated {len(self.maps)} maps!")
    
    def _load_map(self, map_idx: int):
        """Load a specific map by index."""
        if 0 <= map_idx < len(self.maps):
            map_data = self.maps[map_idx]
            self.walls = MapGenerator.load_from_dict(map_data)
            self.current_map_idx = map_idx
            print(f"Loaded: {map_data['display_name']}")
    
    def _draw_menu(self):
        """Draw map selection menu."""
        self.screen.fill(BLACK)
        
        # Title
        title = self.title_font.render("PAC-MAN RL SIMULATION", True, YELLOW)
        title_rect = title.get_rect(center=(self.window_size // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        sub = self.font.render("Select a Map:", True, WHITE)
        sub_rect = sub.get_rect(center=(self.window_size // 2, 140))
        self.screen.blit(sub, sub_rect)
        
        # Map list
        start_y = 180
        for i, map_data in enumerate(self.maps):
            # Highlight current
            color = CYAN if i == self.current_map_idx else WHITE
            prefix = "► " if i == self.current_map_idx else "  "
            text = self.font.render(f"{prefix}{map_data['display_name']}", True, color)
            self.screen.blit(text, (60, start_y + i * 35))
        
        # Instructions
        instr = self.font.render("Press 1-9 to select | ENTER to start | ESC to quit", True, GRAY)
        instr_rect = instr.get_rect(center=(self.window_size // 2, self.window_size + 10))
        self.screen.blit(instr, instr_rect)
        
        pygame.display.flip()
    
    def _generate_walls(self) -> Set[Tuple[int, int]]:
        """Generate wall positions. Simple open design - no closed areas."""
        walls = set()
        
        # Add border walls only - simple open field with scatter obstacles
        for i in range(self.grid_size):
            walls.add((i, 0))
            walls.add((i, self.grid_size - 1))
            walls.add((0, i))
            walls.add((self.grid_size - 1, i))
        
        # Simple, guaranteed-open obstacles (no enclosed rooms possible)
        # Just scattered short lines and small L-shapes that can't close
        
        # Horizontal corridor dividers (short lines with gaps)
        for x in range(4, 12):
            walls.add((x, 8))
            walls.add((x, 21))
        for x in range(18, 26):
            walls.add((x, 8))
            walls.add((x, 21))
        
        # Vertical corridor dividers (short lines with gaps)
        for y in range(4, 10):
            walls.add((8, y))
            walls.add((21, y))
        for y in range(19, 25):
            walls.add((8, y))
            walls.add((21, y))
        
        # Center area - just small blocks, no enclosure
        # Top center
        walls.add((14, 4))
        walls.add((15, 4))
        walls.add((14, 5))
        walls.add((15, 5))
        
        # Bottom center
        walls.add((14, 24))
        walls.add((15, 24))
        walls.add((14, 25))
        walls.add((15, 25))
        
        # Left center
        walls.add((4, 14))
        walls.add((4, 15))
        walls.add((5, 14))
        walls.add((5, 15))
        
        # Right center
        walls.add((24, 14))
        walls.add((24, 15))
        walls.add((25, 14))
        walls.add((25, 15))
        
        # Corner L-shapes (small, open)
        for (base_x, base_y) in [(3, 3), (3, 25), (25, 3), (25, 25)]:
            walls.add((base_x, base_y))
            walls.add((base_x + 1, base_y))
            walls.add((base_x, base_y + 1))
        
        # Small T-shapes scattered (all pointing outward, never enclosed)
        for (cx, cy, dx, dy) in [
            (6, 6, 0, 1),    # T pointing down
            (23, 6, 0, 1),
            (6, 23, 0, -1),  # T pointing up
            (23, 23, 0, -1),
            (6, 12, 1, 0),   # T pointing right
            (23, 12, -1, 0), # T pointing left
            (6, 17, 1, 0),
            (23, 17, -1, 0),
        ]:
            walls.add((cx, cy))
            walls.add((cx + dx, cy + dy))
            walls.add((cx - dx, cy - dy))
            walls.add((cx + (dy if dx == 0 else 0), cy + (dx if dy == 0 else 0)))
        
        return walls
    
    def _generate_pellets(self, max_count: int = None) -> List[Pellet]:
        """Generate pellet positions randomly - only on reachable cells."""
        if max_count is None:
            max_count = self.max_pellets
        
        # First, find all reachable cells (BFS from center-ish starting point)
        start = None
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if (x, y) not in self.walls:
                    start = (x, y)
                    break
            if start:
                break
        
        if not start:
            return []
        
        reachable = set()
        queue = deque([start])
        reachable.add(start)
        
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (0 < nx < self.grid_size - 1 and 0 < ny < self.grid_size - 1 and
                    (nx, ny) not in self.walls and (nx, ny) not in reachable):
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
        
        # Shuffle reachable cells for true randomness
        reachable_list = list(reachable)
        random.shuffle(reachable_list)
        
        # Take up to max_count pellets
        pellets = []
        for pos in reachable_list[:max_count]:
            pellets.append(Pellet(pos))
        
        return pellets
    
    def reset(self, new_map: bool = False):
        """Reset game state. Optionally switch to next map."""
        # If new map requested, cycle to next
        if new_map:
            self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
        
        # Load current map's walls
        self._load_map(self.current_map_idx)
        
        # Find valid spawn positions
        valid_positions = [
            (x, y) 
            for x in range(1, self.grid_size - 1) 
            for y in range(1, self.grid_size - 1)
            if (x, y) not in self.walls
        ]
        
        if not valid_positions:
            print("ERROR: No valid positions! Regenerating map...")
            self.maps[self.current_map_idx] = MapGenerator.generate_random_maze(self.grid_size, 0.1)
            self._load_map(self.current_map_idx)
            valid_positions = [
                (x, y)
                for x in range(1, self.grid_size - 1)
                for y in range(1, self.grid_size - 1)
                if (x, y) not in self.walls
            ]
        
        # Spawn Pac-Man
        pacman_spawn = random.choice(valid_positions)
        self.pacman = PacMan(pacman_spawn, self.walls, self.grid_size)
        
        # Spawn ghosts
        self.ghosts = []
        num_ghosts = min(NUM_GHOSTS, len(valid_positions) - 1)
        if num_ghosts > 0:
            ghost_spawns = random.sample(valid_positions, num_ghosts)
            for i, pos in enumerate(ghost_spawns):
                self.ghosts.append(Ghost(pos, i, self.walls, self.grid_size))
        
        # Generate pellets (only on reachable cells, up to max_pellets)
        self.pellets = self._generate_pellets()
        
        # Game state
        self.game_over = False
        self.win = False
        self.frame = 0
        self.score = 0  # Reset score on new map
        self.start_time = pygame.time.get_ticks()  # Start timer
        self.elapsed_time = 0.0
        self.time_expired = False  # Reset time flag
    
    def update(self):
        """Update game state for one frame."""
        if self.game_over or self.win:
            return
        
        self.frame += 1
        
        # Update timer
        self.elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        
        # Store old positions for collision detection
        old_ghost_positions = [g.pos for g in self.ghosts]
        old_pacman_pos = self.pacman.pos
        
        # Move Pac-Man first (store new pos before ghosts move)
        next_pos = self.pacman.get_next_move(self.pellets, self.ghosts)
        if next_pos:
            self.pacman.move(next_pos)
            
            # Check pellet collection
            for pellet in self.pellets:
                if pellet.active and pellet.pos == self.pacman.pos:
                    self.pacman.collect_pellet(pellet)
                    break
        
        # Move ghosts (blocking each other)
        ghost_positions = {g.pos for g in self.ghosts}
        for i, ghost in enumerate(self.ghosts):
            # Get positions of other ghosts (excluding self)
            other_positions = ghost_positions - {ghost.pos}
            ghost.move(other_positions)
        
        # FIXED: Collision detection AFTER both move (catches position swaps)
        for i, ghost in enumerate(self.ghosts):
            if ghost.pos == self.pacman.pos:  # Same cell
                self.game_over = True
                break
            if ghost.pos == old_pacman_pos:  # Ghost caught Pac-Man
                self.game_over = True
                break
            if self.pacman.pos == old_ghost_positions[i]:  # Pac-Man ran into ghost
                self.game_over = True
                break
        
        # Check win conditions: all pellets OR time limit
        if all(not p.active for p in self.pellets):
            self.win = True
        elif self.elapsed_time >= self.time_limit:
            # Time's up - Pac-Man wins by surviving
            self.win = True
            self.time_expired = True
    
    def draw_grid(self):
        """Draw the grid background."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                if (x, y) in self.walls:
                    pygame.draw.rect(self.screen, GRAY, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 1)
                else:
                    pygame.draw.rect(self.screen, DARK_BLUE, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)
    
    def draw_pellets(self):
        """Draw all active pellets."""
        for pellet in self.pellets:
            if pellet.active:
                center_x = pellet.pos[0] * self.cell_size + self.cell_size // 2
                center_y = pellet.pos[1] * self.cell_size + self.cell_size // 2
                pygame.draw.circle(self.screen, PELLET_COLOR, 
                                  (center_x, center_y), 4)
    
    def draw(self):
        """Draw entire game state."""
        # Clear
        self.screen.fill(BLACK)
        
        # Draw grid and walls
        self.draw_grid()
        
        # Draw pellets
        self.draw_pellets()
        
        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        
        # Draw Pac-Man
        self.pacman.draw(self.screen)
        
        # Draw UI
        self._draw_ui()
        
        # Draw popup if game ended
        self._draw_popup()
        
        pygame.display.flip()
    
    def _draw_ui(self):
        """Draw score and status info."""
        # Map name (left)
        map_name = self.maps[self.current_map_idx]["display_name"] if self.maps else "Unknown"
        map_text = self.font.render(f"Map: {map_name}", True, CYAN)
        self.screen.blit(map_text, (10, self.window_size + 10))
        
        # Timer (center-left) - show countdown (compute fresh from start_time)
        current_elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0 if self.start_time else 0
        remaining = max(0, self.time_limit - current_elapsed)
        timer_color = RED if remaining < 5 else YELLOW if remaining < 10 else WHITE
        timer_text = self.font.render(f"⏱ {remaining:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.window_size // 4, self.window_size + 10))
        
        # Score (center)
        score_text = self.font.render(
            f"Score: {self.pacman.score} | Pellets: {self.pacman.pellets_collected}/{len(self.pellets)}", 
            True, WHITE)
        score_rect = score_text.get_rect(center=(self.window_size // 2, self.window_size + 20))
        self.screen.blit(score_text, score_rect)
        
        # Ghost danger indicator (right)
        nearby = self.pacman.get_nearby_ghosts(self.ghosts, GHOST_ESCAPE_DISTANCE)
        if nearby:
            danger = self.font.render(f"⚠ {len(nearby)} ghost(s)!", True, RED)
            self.screen.blit(danger, (self.window_size - 120, self.window_size + 10))
    
    def _draw_popup(self):
        """Draw WIN or LOSE popup overlay."""
        if not self.game_over and not self.win:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size + 40))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Popup box
        popup_x = (self.window_size - POPUP_WIDTH) // 2
        popup_y = (self.window_size - POPUP_HEIGHT) // 2
        popup_rect = pygame.Rect(popup_x, popup_y, POPUP_WIDTH, POPUP_HEIGHT)
        
        # Draw popup background
        pygame.draw.rect(self.screen, WHITE, popup_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, popup_rect, 4, border_radius=10)
        
        # Title
        if self.win:
            if getattr(self, 'time_expired', False):
                title_text = "🎉 PAC-MAN WINS! 🎉"
                title_color = YELLOW
                message = f"Survived {self.time_limit}s! ({self.pacman.pellets_collected}/{len(self.pellets)} pellets)"
            else:
                title_text = "🎉 YOU WIN! 🎉"
                title_color = YELLOW
                message = "All pellets collected!"
        else:
            title_text = "💀 GAME OVER 💀"
            title_color = RED
            message = "Caught by a ghost!"
        
        # Render title
        title_font = pygame.font.Font(None, 48)
        title_surface = title_font.render(title_text, True, title_color)
        title_rect = title_surface.get_rect(center=(self.window_size // 2, popup_y + 60))
        self.screen.blit(title_surface, title_rect)
        
        # Render message
        msg_font = pygame.font.Font(None, 32)
        msg_surface = msg_font.render(message, True, BLACK)
        msg_rect = msg_surface.get_rect(center=(self.window_size // 2, popup_y + 110))
        self.screen.blit(msg_surface, msg_rect)
        
        # Render stats
        stats_font = pygame.font.Font(None, 24)
        stats_text = f"Score: {self.pacman.score} | Pellets: {self.pacman.pellets_collected}/{len(self.pellets)}"
        stats_surface = stats_font.render(stats_text, True, GRAY)
        stats_rect = stats_surface.get_rect(center=(self.window_size // 2, popup_y + 150))
        self.screen.blit(stats_surface, stats_rect)
        
        # Reset instruction
        reset_font = pygame.font.Font(None, 20)
        reset_surface = reset_font.render("R: Restart | N: Next Map | Auto in 5s...", True, GRAY)
        reset_rect = reset_surface.get_rect(center=(self.window_size // 2, popup_y + 175))
        self.screen.blit(reset_surface, reset_rect)
    
    def run(self):
        """Main game loop."""
        running = True
        game_ended_time = None  # Track when game ended
        AUTO_RESET_DELAY = 5000  # 5 seconds auto reset
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Menu state
            if self.in_menu:
                self._draw_menu()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif pygame.K_1 <= event.key <= pygame.K_9:
                            # Select map 1-9
                            idx = event.key - pygame.K_1
                            if idx < len(self.maps):
                                self.current_map_idx = idx
                        elif event.key == pygame.K_RETURN:
                            # Start game with selected map
                            self._load_map(self.current_map_idx)
                            self.reset()
                            self.in_menu = False
                            game_ended_time = None
                        elif event.key == pygame.K_UP:
                            self.current_map_idx = (self.current_map_idx - 1) % len(self.maps)
                        elif event.key == pygame.K_DOWN:
                            self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
                
                self.clock.tick(FPS)
                continue
            
            # Game state
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # R always resets (same map)
                        self.reset(new_map=False)
                        game_ended_time = None
                    elif event.key == pygame.K_n:
                        # N for next map
                        self.reset(new_map=True)
                        game_ended_time = None
                    elif event.key == pygame.K_m:
                        # M for map menu
                        self.in_menu = True
                        game_ended_time = None
            
            # Check game end state
            if (self.game_over or self.win) and game_ended_time is None:
                game_ended_time = current_time
            
            # Auto reset after delay (only if not already reset by R/N)
            if game_ended_time and (current_time - game_ended_time > AUTO_RESET_DELAY):
                self.reset(new_map=False)
                game_ended_time = None
            
            # Update only if game not ended
            if not self.game_over and not self.win:
                self.update()
            
            # Draw
            self.draw()
            
            # FPS limit
            self.clock.tick(FPS)
        
        pygame.quit()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PAC-MAN RL TRAINING SIMULATION")
    print("=" * 60)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Ghosts: {NUM_GHOSTS}")
    print(f"Escape distance: {GHOST_ESCAPE_DISTANCE} cells")
    print("\nFEATURES:")
    print("  - Auto-generated connected mazes (no closed areas)")
    print("  - Multiple map types: Random, Recursive Maze, Classic-style")
    print("  - Map selection menu at startup")
    print("  - Ghosts block each other (can't occupy same cell)")
    print("  - WIN/LOSE popup with 5s auto-reset")
    print("\nCONTROLS:")
    print("  Menu: 1-9 select map | UP/DOWN arrows | ENTER to start")
    print("  Game: ESC quit | R restart | N next map | M menu")
    print("\nPAC-MAN AI:")
    print("  - BFS pathfinding to nearest pellet")
    print("  - Escape mode within 5 cells of ghost")
    print("  - Emergency escape within 3 cells")
    print("=" * 60)
    
    game = PacManGame()
    game.run()
