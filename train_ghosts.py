"""
Q-Learning Training Script for Ghost Agents
Trains ghosts to catch Pac-Man using Q-learning.

Usage:
    # Basic training (4 ghosts, map 0)
    python train_ghosts.py --episodes 1000 --save-dir checkpoints/
    
    # Train on specific map (index 0-5)
    python train_ghosts.py --episodes 1000 --train-map 2 --save-dir checkpoints_map2/
    
    # Train on one map, test on another
    python train_ghosts.py --episodes 1000 --train-map 0 --test-map 3 --save-dir checkpoints_m0_t3/
    
    # Train with 2 ghosts
    python train_ghosts.py --ghosts 2 --episodes 1000 --save-dir checkpoints_2g/
    
    # Continue training from saved models
    python train_ghosts.py --load-dir checkpoints/ --episodes 500
    
    # Watch training live
    python train_ghosts.py --render --episodes 100
    
    # Render every 10th episode
    python train_ghosts.py --render-every 10 --episodes 500
    
    # Use custom config file
    python train_ghosts.py --episodes 1000 --config my_config.json --save-dir checkpoints/
"""

import argparse
import os
import numpy as np
import sys
from collections import deque
from typing import Optional

# Import game components
from pacman_game import (
    PacManGame, NUM_GHOSTS,
    PacMan, Ghost, Pellet, DIRECTIONS, is_valid_position,
    bfs_path, manhattan_distance
)

# Import shared RL utilities
from rl_utils import (
    encode_state, execute_action,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTIONS,
    ACTION_NAMES, REWARD_COLLISION, REWARD_PELLETS_EATEN,
    REWARD_CLOSER, REWARD_FARTHER, REWARD_INVALID, REWARD_REPEAT,
    REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, DIST_BINS,
    QLearningGhost,
)

# Import model implementations
from models import (
    MultiGhostQLearning,
    MultiGhostMADDPG,
    MultiGhostDQN,
    MultiGhostPPO,
    MultiGhostQMIX,
)

import pygame
import json
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG LOADER
# ============================================================================
DEFAULT_CONFIG_PATH = "config.json"

def load_config(config_path: str = None) -> dict:
    """Load hyperparameters from config.json file.
    
    Args:
        config_path: Path to config file. Uses default if None.
    
    Returns:
        Dict with q_learning, rewards, state, training sections.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config = {
        # Note: model type is set via --model CLI arg, not config file
        "q_learning": {
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "use_noop": False,
        },
        "maddpg": {
            "actor_lr": 0.0005,
            "critic_lr": 0.001,
            "gamma": 0.95,
            "tau": 0.01,
            "batch_size": 128,
            "buffer_size": 100000,
            "update_every": 1,
            "epsilon_start": 0.2,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "hidden_sizes": [128, 128],
        },
        "dqn": {
            "lr": 0.0005,
            "gamma": 0.95,
            "batch_size": 128,
            "buffer_size": 100000,
            "update_every": 4,
            "tau": 0.001,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "hidden_sizes": [128, 128],
        },
        "ppo": {
            "lr": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "update_epochs": 4,
            "minibatch_size": 256,
            "max_grad_norm": 0.5,
            "hidden_sizes": [128, 128]
        },
        "qmix": {
            "lr": 0.0005,
            "gamma": 0.95,
            "batch_size": 128,
            "buffer_size": 100000,
            "update_every": 4,
            "tau": 0.005,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "hidden_sizes": [128, 128],
            "mixing_hidden": 64,
        },
        "rewards": {
            "collision": 100,
            "pellets_done": -100,
            "closer": 1,
            "farther": -1,
            "invalid": -0.5,
            "repeat": -0.25,
        },
        "repeat_detection": {
            "window": 4,
            "threshold": 2,
        },
        "state": {
            "dist_bins": [0, 2, 4, 8, 16, 30],
            "grid_size": 30,
        },
        "observation": {
            "range": 30,
            "shared_pacman": False,
        },
        "training": {
            "max_steps_per_episode": 500,
            "shared_collision_reward": True,
        },
        "coordination": {
            "enabled": True,
            "vision_range": 8,
            "surround": True,
            "search_target_refresh": 12,
            "search_history_limit": 6,
        },
        "team_rewards": {
            "mobility": 1.5,
            "adjacent_block": 0.75,
            "spread": 0.5,
            "cluster_penalty": -0.6,
            "line_penalty": -1.2,
            "overlap_penalty": -0.4,
            "cluster_distance": 1
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Deep merge user config into defaults
            for section in ["q_learning", "maddpg", "dqn", "ppo", "qmix", "rewards", "repeat_detection", "state", "observation", "training", "coordination", "team_rewards"]:
                if section in user_config:
                    config[section].update(user_config[section])
            
            logger.info(f"✓ Loaded config from: {config_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠ Failed to parse config.json: {e}. Using defaults.")
        except Exception as e:
            logger.warning(f"⚠ Error loading config: {e}. Using defaults.")
    else:
        logger.info(f"ℹ Config file not found at {config_path}. Using defaults.")
    
    return config


def log_config(config: dict, train_map: int, test_map: int, 
              train_map_name: str, test_map_name: str, 
              num_ghosts: int, episodes: int, model_type: str = "qlearning",
              train_maps: Optional[list] = None, train_grid_sizes: Optional[list] = None):
    """Log all loaded configuration parameters."""
    ql = config.get("q_learning", {})
    rw = config.get("rewards", {})
    st = config.get("state", {})
    tr = config.get("training", {})
    coord = config.get("coordination", {})
    team_rw = config.get("team_rewards", {})
    
    logger.info("=" * 60)
    logger.info("CONFIGURATION LOADED")
    logger.info("=" * 60)
    
    # Map configuration
    if train_maps:
        logger.info(f"[MAP] Train Maps: {train_maps}")
    else:
        logger.info(f"[MAP] Train Map: {train_map_name} (index {train_map})")
    logger.info(f"[MAP] Test Map:  {test_map_name} (index {test_map})")
    
    # Game settings
    logger.info(f"[GAME] Ghosts: {num_ghosts}")
    logger.info(f"[GAME] Episodes: {episodes}")
    if train_grid_sizes:
        logger.info(f"[GAME] Train Grid Sizes: {train_grid_sizes}")
    else:
        logger.info(f"[GAME] Grid Size: {st.get('grid_size', 30)}x{st.get('grid_size', 30)}")
    logger.info(f"[GAME] Max Steps/Episode: {tr.get('max_steps_per_episode', 500)}")
    
    # Model selection (from CLI arg)
    logger.info(f"[MODEL] Type: {model_type}")
    
    # Q-learning hyperparameters
    logger.info(f"[QL] Alpha (α): {ql.get('alpha', 0.1)}")
    logger.info(f"[QL] Gamma (γ): {ql.get('gamma', 0.95)}")
    logger.info(f"[QL] Epsilon: start={ql.get('epsilon_start', 1.0)}, min={ql.get('epsilon_min', 0.05)}, decay={ql.get('epsilon_decay', 0.995)}")
    logger.info(f"[QL] Use NOOP: {ql.get('use_noop', False)}")
    
    # MADDPG hyperparameters
    maddpg = config.get("maddpg", {})
    logger.info(f"[MADDPG] Actor LR: {maddpg.get('actor_lr', 0.0005)}")
    logger.info(f"[MADDPG] Critic LR: {maddpg.get('critic_lr', 0.001)}")
    logger.info(f"[MADDPG] Gamma: {maddpg.get('gamma', 0.95)}")
    logger.info(f"[MADDPG] Tau: {maddpg.get('tau', 0.01)}")
    logger.info(f"[MADDPG] Batch Size: {maddpg.get('batch_size', 128)}")
    logger.info(f"[MADDPG] Buffer Size: {maddpg.get('buffer_size', 100000)}")
    logger.info(f"[MADDPG] Update Every: {maddpg.get('update_every', 1)}")
    logger.info(f"[MADDPG] Epsilon: start={maddpg.get('epsilon_start', 0.2)}, min={maddpg.get('epsilon_min', 0.02)}, decay={maddpg.get('epsilon_decay', 0.995)}")
    logger.info(f"[MADDPG] Hidden Sizes: {maddpg.get('hidden_sizes', [128, 128])}")

    # DQN hyperparameters
    dqn = config.get("dqn", {})
    logger.info(f"[DQN] LR: {dqn.get('lr', 0.0005)}")
    logger.info(f"[DQN] Gamma: {dqn.get('gamma', 0.95)}")
    logger.info(f"[DQN] Batch Size: {dqn.get('batch_size', 128)}")
    logger.info(f"[DQN] Buffer Size: {dqn.get('buffer_size', 100000)}")
    logger.info(f"[DQN] Update Every: {dqn.get('update_every', 4)}")
    logger.info(f"[DQN] Tau: {dqn.get('tau', 0.001)}")
    logger.info(f"[DQN] Epsilon: start={dqn.get('epsilon_start', 1.0)}, min={dqn.get('epsilon_min', 0.05)}, decay={dqn.get('epsilon_decay', 0.995)}")
    logger.info(f"[DQN] Hidden Sizes: {dqn.get('hidden_sizes', [128, 128])}")

    # PPO hyperparameters
    ppo = config.get("ppo", {})
    logger.info(f"[PPO] LR: {ppo.get('lr', 0.0003)}")
    logger.info(f"[PPO] Gamma: {ppo.get('gamma', 0.99)}")
    logger.info(f"[PPO] GAE Lambda: {ppo.get('gae_lambda', 0.95)}")
    logger.info(f"[PPO] Clip Epsilon: {ppo.get('clip_epsilon', 0.2)}")
    logger.info(f"[PPO] Value Coef: {ppo.get('value_coef', 0.5)}")
    logger.info(f"[PPO] Entropy Coef: {ppo.get('entropy_coef', 0.01)}")
    logger.info(f"[PPO] Update Epochs: {ppo.get('update_epochs', 4)}")
    logger.info(f"[PPO] Minibatch Size: {ppo.get('minibatch_size', 256)}")
    logger.info(f"[PPO] Max Grad Norm: {ppo.get('max_grad_norm', 0.5)}")
    logger.info(f"[PPO] Hidden Sizes: {ppo.get('hidden_sizes', [128, 128])}")

    # QMIX hyperparameters
    qmix = config.get("qmix", {})
    logger.info(f"[QMIX] LR: {qmix.get('lr', 0.0005)}")
    logger.info(f"[QMIX] Gamma: {qmix.get('gamma', 0.95)}")
    logger.info(f"[QMIX] Batch Size: {qmix.get('batch_size', 128)}")
    logger.info(f"[QMIX] Buffer Size: {qmix.get('buffer_size', 100000)}")
    logger.info(f"[QMIX] Update Every: {qmix.get('update_every', 4)}")
    logger.info(f"[QMIX] Tau: {qmix.get('tau', 0.005)}")
    logger.info(f"[QMIX] Epsilon: start={qmix.get('epsilon_start', 1.0)}, min={qmix.get('epsilon_min', 0.05)}, decay={qmix.get('epsilon_decay', 0.995)}")
    logger.info(f"[QMIX] Hidden Sizes: {qmix.get('hidden_sizes', [128, 128])}")
    logger.info(f"[QMIX] Mixing Hidden: {qmix.get('mixing_hidden', 64)}")

    
    # Rewards
    logger.info(f"[REWARD] Collision: {rw.get('collision', 100)}")
    logger.info(f"[REWARD] Pellets Done: {rw.get('pellets_done', -100)}")
    logger.info(f"[REWARD] Closer/Farther: {rw.get('closer', 1)}/{rw.get('farther', -1)}")
    logger.info(f"[REWARD] Invalid: {rw.get('invalid', -0.5)}")
    logger.info(f"[REWARD] Repeat: {rw.get('repeat', -0.25)}")
    
    # Repeat detection
    repeat_cfg = config.get("repeat_detection", {})
    logger.info(f"[REPEAT] Window: {repeat_cfg.get('window', 4)} | Threshold: {repeat_cfg.get('threshold', 2)}")
    
    # State space
    logger.info(f"[STATE] Dist Bins: {st.get('dist_bins', [0,2,4,8,16,30])}")
    logger.info(f"[STATE] Grid Size: {st.get('grid_size', 30)}")
    
    # Observation / partial observability
    obs = config.get("observation", {})
    logger.info(f"[OBS] Range: {obs.get('range', 30)} (full obs if >= grid size)")
    logger.info(f"[OBS] Shared Pac-Man: {obs.get('shared_pacman', True)}")
    
    # Coordination / communication
    logger.info(f"[COORD] Enabled: {coord.get('enabled', True)}")
    logger.info(f"[COORD] Vision Range: {coord.get('vision_range', 8)}")
    logger.info(f"[COORD] Surround: {coord.get('surround', True)}")
    logger.info(f"[COORD] Search Target Refresh: {coord.get('search_target_refresh', 12)}")
    logger.info(f"[COORD] Search History Limit: {coord.get('search_history_limit', 6)}")

    # Team rewards / penalties
    logger.info(f"[TEAM] Mobility Reward: {team_rw.get('mobility', 1.5)}")
    logger.info(f"[TEAM] Adjacent Block Reward: {team_rw.get('adjacent_block', 0.75)}")
    logger.info(f"[TEAM] Spread Reward: {team_rw.get('spread', 0.5)}")
    logger.info(f"[TEAM] Cluster Penalty: {team_rw.get('cluster_penalty', -0.6)}")
    logger.info(f"[TEAM] Line Penalty: {team_rw.get('line_penalty', -1.2)}")
    logger.info(f"[TEAM] Overlap Penalty: {team_rw.get('overlap_penalty', -0.4)}")
    logger.info(f"[TEAM] Cluster Distance: {team_rw.get('cluster_distance', 1)}")
    
    logger.info("=" * 60)


class RLTrainingEnvironment:
    """Wraps PacManGame for RL training. Set render=True to watch training."""
    
    def __init__(self, render: bool = False, fps: int = 10, num_ghosts: int = 4,
                 train_map: int = 0, test_map: Optional[int] = None,
                 config: dict = None, config_path: str = None,
                 model_type: str = "qlearning", log_moves: bool = False,
                 train_random_map: bool = False,
                 train_maps: Optional[list] = None,
                 train_grid_sizes: Optional[list] = None):
        """
        Initialize training environment.
        
        Args:
            render: Enable visual rendering
            fps: FPS when rendering
            num_ghosts: Number of ghosts (2 or 4)
            train_map: Map index for training (0-5)
            test_map: Map index for testing (defaults to train_map)
            config: Pre-loaded config dict (from load_config())
            config_path: Path to config.json (if config not provided)
            model_type: 'qlearning', 'maddpg', 'dqn', 'ppo', or 'qmix'
            log_moves: Enable logging of moves to file
            train_random_map: If True, pick random training map each episode
            train_maps: Specific map indices to sample from each episode
            train_grid_sizes: Grid sizes to sample from each episode
        """
        self.render_mode = render
        self.fps = fps
        self.num_ghosts = num_ghosts
        self.train_map = train_map
        self.test_map = test_map if test_map is not None else train_map
        self.train_random_map = train_random_map
        self.train_maps = train_maps[:] if train_maps else (list(range(6)) if train_random_map else [train_map])
        self.train_grid_sizes = [int(gs) for gs in train_grid_sizes] if train_grid_sizes else None
        self.log_moves_enabled = log_moves
        self.log_file = "evaluation_moves.txt" if log_moves else None
        
        if self.log_moves_enabled:
            with open(self.log_file, "w") as f:
                f.write(f"Evaluation Log - Map: {self.test_map}, Ghosts: {self.num_ghosts}, Model: {model_type}\n")
                f.write("Episode | Step | PacMan | Ghost 0 | Ghost 1 | Ghost 2 | Ghost 3 | Event\n")
                f.write("-" * 100 + "\n")
        
        # Load config if not provided
        if config is None:
            config = load_config(config_path)
        self.config = config
        self.base_grid_size = int(self.config.get("state", {}).get("grid_size", 30))
        if self.train_grid_sizes is None:
            self.train_grid_sizes = [self.base_grid_size]
        self.grid_size = self.base_grid_size
        
        # Model type from CLI arg (not config)
        self.model_type = model_type.lower()
        if self.model_type not in ("qlearning", "maddpg", "dqn", "ppo", "qmix"):
            logger.warning(f"Unknown model type '{model_type}', defaulting to qlearning")
            self.model_type = "qlearning"
        logger.info(f"[INIT] Model type: {self.model_type}")
        
        # Extract hyperparameters from config
        ql = config.get("q_learning", {})
        self.alpha = ql.get("alpha", 0.1)
        self.gamma = ql.get("gamma", 0.95)
        self.epsilon = ql.get("epsilon_start", 1.0)
        self.epsilon_min = ql.get("epsilon_min", 0.05)
        self.epsilon_decay = ql.get("epsilon_decay", 0.995)
        self.use_noop = ql.get("use_noop", False)
        
        # Repetition config
        repeat_cfg = config.get("repeat_detection", {})
        self.repeat_window = repeat_cfg.get("window", 4)
        self.repeat_threshold = repeat_cfg.get("threshold", 2)
        
        reward_cfg = config.get("rewards", {})
        self.repeat_penalty = reward_cfg.get("repeat", -0.25)
        
        # Store for reporting
        if self.model_type == "ppo":
            ppo_cfg = config.get("ppo", {})
            self.hyperparams = {
                "lr": ppo_cfg.get("lr", 0.0003),
                "gamma": ppo_cfg.get("gamma", 0.99),
                "gae_lambda": ppo_cfg.get("gae_lambda", 0.95),
                "clip_epsilon": ppo_cfg.get("clip_epsilon", 0.2),
                "value_coef": ppo_cfg.get("value_coef", 0.5),
                "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
                "update_epochs": ppo_cfg.get("update_epochs", 4),
                "minibatch_size": ppo_cfg.get("minibatch_size", 256),
                "max_grad_norm": ppo_cfg.get("max_grad_norm", 0.5),
                "hidden_sizes": ppo_cfg.get("hidden_sizes", [128, 128]),
                "use_noop": self.use_noop,
                "repeat_window": self.repeat_window,
                "repeat_threshold": self.repeat_threshold,
                "repeat_penalty": self.repeat_penalty,
            }
        else:
            self.hyperparams = {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "use_noop": self.use_noop,
                "repeat_window": self.repeat_window,
                "repeat_threshold": self.repeat_threshold,
                "repeat_penalty": self.repeat_penalty,
            }
        
        # Coordination / communication settings
        coord_cfg = config.get("coordination", {})
        self.coordination_enabled = coord_cfg.get("enabled", True)
        self.coordination_vision = coord_cfg.get("vision_range", 8)
        self.coordination_surround = coord_cfg.get("surround", True)
        self.search_target_refresh = coord_cfg.get("search_target_refresh", 12)
        self.search_history_limit = coord_cfg.get("search_history_limit", 6)
        self.search_targets = [None for _ in range(self.num_ghosts)]
        self.search_steps = [0 for _ in range(self.num_ghosts)]
        self.search_visited = [deque(maxlen=self.search_history_limit) for _ in range(self.num_ghosts)]
        self.current_train_map = self.train_map
        self.current_train_map_name = None
        self.episode_grid_sizes = []
        self.episode_map_indices = []
        self.episode_map_names = []

        # Team reward shaping settings
        team_cfg = config.get("team_rewards", {})
        self.team_reward_cfg = {
            "mobility": team_cfg.get("mobility", 1.5),
            "adjacent_block": team_cfg.get("adjacent_block", 0.75),
            "spread": team_cfg.get("spread", 0.5),
            "cluster_penalty": team_cfg.get("cluster_penalty", -0.6),
            "line_penalty": team_cfg.get("line_penalty", -1.2),
            "overlap_penalty": team_cfg.get("overlap_penalty", -0.4),
            "cluster_distance": team_cfg.get("cluster_distance", 1)
        }
        
        # Rewards from config (for info)
        self.rewards_config = reward_cfg
        
        # Update global NUM_GHOSTS for this session
        import pacman_game
        pacman_game.NUM_GHOSTS = num_ghosts
        
        if not render:
            # Headless mode - no window
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        pygame.init()
        
        self.game = PacManGame(grid_size=self.grid_size)
        self.game.in_menu = False  # Skip menu
        
        # FIX: Set current_map_idx BEFORE loading to prevent reset() from overriding
        self.game.current_map_idx = self.train_map
        logger.info(f"[INIT] Setting initial map to index {self.train_map}")
        self.game._load_map(self.train_map)
        self.game.reset()  # This will now use correct current_map_idx
        self.current_train_map = self.train_map
        self.current_train_map_name = self._get_map_name(self.train_map)
        
        # Clock for rendering
        self.clock = pygame.time.Clock()
        
        if self.model_type == "qlearning":
            reward_cfg = self.rewards_config
            self.agents = MultiGhostQLearning(
                num_ghosts=num_ghosts,
                alpha=self.alpha,
                gamma=self.gamma,
                epsilon=self.epsilon,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                use_noop=self.use_noop,
                repeat_window=self.repeat_window,
                repeat_threshold=self.repeat_threshold,
                repeat_penalty=self.repeat_penalty,
                reward_collision=reward_cfg.get("collision", 100),
                reward_pellets=reward_cfg.get("pellets_done", -100),
                reward_closer=reward_cfg.get("closer", 1),
                reward_farther=reward_cfg.get("farther", -1),
                reward_invalid=reward_cfg.get("invalid", -0.5)
            )
        elif self.model_type == "maddpg":
            # MADDPG uses continuous actor outputs mapped to discrete actions
            state_dim = len(encode_state((0, 0), (0, 0), [], set(), [], self.grid_size))
            action_dim = len(ACTIONS) + (1 if self.use_noop else 0)
            
            self.agents = MultiGhostMADDPG(
                num_ghosts=num_ghosts,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                repeat_window=self.repeat_window,
                repeat_threshold=self.repeat_threshold,
                repeat_penalty=self.repeat_penalty
            )
        elif self.model_type == "dqn":
            state_dim = len(encode_state((0, 0), (0, 0), [], set(), [], self.grid_size)) + num_ghosts
            action_dim = len(ACTIONS) + (1 if self.use_noop else 0)
            
            self.agents = MultiGhostDQN(
                num_ghosts=num_ghosts,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                repeat_window=self.repeat_window,
                repeat_threshold=self.repeat_threshold,
                repeat_penalty=self.repeat_penalty
            )
        elif self.model_type == "ppo":
            state_dim = len(encode_state((0, 0), (0, 0), [], set(), [], self.grid_size)) + num_ghosts
            action_dim = len(ACTIONS) + (1 if self.use_noop else 0)

            self.agents = MultiGhostPPO(
                num_ghosts=num_ghosts,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                repeat_window=self.repeat_window,
                repeat_threshold=self.repeat_threshold,
                repeat_penalty=self.repeat_penalty
            )
        elif self.model_type == "qmix":
            state_dim = len(encode_state((0, 0), (0, 0), [], set(), [], self.grid_size))
            action_dim = len(ACTIONS) + (1 if self.use_noop else 0)
            
            self.agents = MultiGhostQMIX(
                num_ghosts=num_ghosts,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                repeat_window=self.repeat_window,
                repeat_threshold=self.repeat_threshold,
                repeat_penalty=self.repeat_penalty
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Episode tracking (training)
        self.episode = 0
        self.total_episodes = 0
        self.wins_pacman = 0
        self.wins_ghosts = 0
        self.episode_lengths = []
        self.episode_rewards_history = []
        self.repeat_positions = [deque(maxlen=self.repeat_window) for _ in range(self.num_ghosts)]
        self.repeat_strike_count = [0 for _ in range(self.num_ghosts)]
        self.repeat_strike_limit = 8
        
        # Episode tracking (evaluation)
        self.eval_wins_pacman = 0
        self.eval_wins_ghosts = 0
        self.eval_episode_lengths = []
        self.eval_rewards_history = []
        
        # Store map names for reporting (with logging)
        self.train_map_name = self._get_map_name(self.train_map)
        self.test_map_name = self._get_map_name(self.test_map)
        train_map_label = "Random" if self.train_random_map else self.train_map_name
        logger.info(f"[INIT] Train map: {train_map_label}")
        logger.info(f"[INIT] Train map candidates: {self.train_maps}")
        logger.info(f"[INIT] Train grid sizes: {self.train_grid_sizes}")
        logger.info(f"[INIT] Test map:  {self.test_map_name}")
    
    def _get_map_name(self, map_idx: int) -> str:
        """Get display name for a map index."""
        if 0 <= map_idx < len(self.game.maps):
            return self.game.maps[map_idx].get("display_name", f"Map {map_idx}")
        return f"Map {map_idx}"

    def _choose_training_map(self) -> int:
        """Choose map index for the next training episode."""
        if self.train_random_map or len(self.train_maps) > 1:
            return int(np.random.choice(self.train_maps))
        return self.train_map

    def _choose_training_grid_size(self) -> int:
        """Choose grid size for the next training episode."""
        if len(self.train_grid_sizes) > 1:
            return int(np.random.choice(self.train_grid_sizes))
        return int(self.train_grid_sizes[0])

    def _rebuild_game_for_grid_size(self, grid_size: int):
        """Recreate the game instance when the grid size changes."""
        if int(grid_size) == int(self.grid_size):
            return
        self.grid_size = int(grid_size)
        self.config.setdefault("state", {})["grid_size"] = self.grid_size
        self.game = PacManGame(grid_size=self.grid_size)
        self.game.in_menu = False
        self.game.max_pellets = getattr(self.game, "max_pellets", self.config.get("training", {}).get("max_pellets", 100))
        self.game.time_limit = getattr(self.game, "time_limit", self.config.get("training", {}).get("time_limit", 20.0))

    def _set_training_episode_context(self):
        """Pick and apply training map and grid size for the current episode."""
        selected_grid_size = self._choose_training_grid_size()
        previous_max_pellets = getattr(self.game, "max_pellets", None) if hasattr(self, "game") else None
        previous_time_limit = getattr(self.game, "time_limit", None) if hasattr(self, "game") else None
        self._rebuild_game_for_grid_size(selected_grid_size)
        if previous_max_pellets is not None:
            self.game.max_pellets = previous_max_pellets
        if previous_time_limit is not None:
            self.game.time_limit = previous_time_limit

        self.current_train_map = self._choose_training_map()
        self.current_train_map_name = self._get_map_name(self.current_train_map)
        self.game.current_map_idx = self.current_train_map

    def reset(self, new_map: bool = False):
        """Reset for new episode, optionally changing map and grid size."""
        if new_map:
            self.game.current_map_idx = (self.game.current_map_idx + 1) % len(self.game.maps)
            self.current_train_map = self.game.current_map_idx
            self.current_train_map_name = self._get_map_name(self.current_train_map)
            logger.debug(f"[RESET] Cycling to map index {self.game.current_map_idx}")
        else:
            self._set_training_episode_context()

        self.game._load_map(self.game.current_map_idx)
        self.game.reset()
        self.agents.reset_all()
        self.repeat_positions = [deque(maxlen=self.repeat_window) for _ in range(self.num_ghosts)]
        self.repeat_strike_count = [0 for _ in range(self.num_ghosts)]
        self.search_targets = [None for _ in range(self.num_ghosts)]
        self.search_steps = [0 for _ in range(self.num_ghosts)]
        self.search_visited = [deque(maxlen=self.search_history_limit) for _ in range(self.num_ghosts)]

        self.episode_grid_sizes.append(self.grid_size)
        self.episode_map_indices.append(self.game.current_map_idx)
        self.episode_map_names.append(self.current_train_map_name)
        logger.info(
            f"[EPISODE] Ep {self.episode + 1:5d} | Grid: {self.grid_size}x{self.grid_size} | "
            f"Map: {self.current_train_map_name} (index {self.game.current_map_idx})"
        )
        
        if self.episode == 0:
            logger.debug(f"[RESET] Episode reset complete on map: {self._get_map_name(self.game.current_map_idx)}")
        
        return self._get_states()
    
    def reset_for_test(self, map_idx: Optional[int] = None):
        """Reset for testing on a specific map (cross-map evaluation)."""
        test_idx = map_idx if map_idx is not None else self.test_map
        self.game.current_map_idx = test_idx
        logger.info(f"[TEST] Resetting for test on map {test_idx}: {self._get_map_name(test_idx)}")
        self.game._load_map(test_idx)
        self.game.reset()
        self.agents.reset_all()
        self.repeat_positions = [deque(maxlen=self.repeat_window) for _ in range(self.num_ghosts)]
        self.repeat_strike_count = [0 for _ in range(self.num_ghosts)]
        self.search_targets = [None for _ in range(self.num_ghosts)]
        self.search_steps = [0 for _ in range(self.num_ghosts)]
        self.search_visited = [deque(maxlen=self.search_history_limit) for _ in range(self.num_ghosts)]
        return self._get_states()
    
    def _get_states(self):
        """Get current states for all ghosts with optional partial observability."""
        obs_cfg = self.config.get("observation", {})
        obs_range = obs_cfg.get("range", 30)
        shared_pacman = obs_cfg.get("shared_pacman", True)
        
        # Full observability if range >= grid_size
        if obs_range >= self.grid_size:
            obs_range = None
        
        # Check shared visibility: does ANY ghost see PM within range?
        team_sees_pm = False
        if shared_pacman and obs_range is not None:
            for ghost in self.game.ghosts:
                if manhattan_distance(ghost.pos, self.game.pacman.pos) <= obs_range:
                    team_sees_pm = True
                    break
        
        states = []
        for i, ghost in enumerate(self.game.ghosts):
            other_ghosts = [g.pos for j, g in enumerate(self.game.ghosts) if j != i]
            # Shared mode: if team sees PM, all get full obs; else all unknown
            # Individual mode: each ghost's own range
            if shared_pacman:
                effective_range = None if team_sees_pm else obs_range
            else:
                effective_range = obs_range
            
            state = encode_state(
                ghost.pos,
                self.game.pacman.pos,
                other_ghosts,
                self.game.walls,
                self.game.pellets,
                self.grid_size,
                observation_range=effective_range
            )
            states.append(state)
        return states

    def _ghost_can_see_pacman(self, ghost_pos):
        """Check if Pac-Man is within coordination vision range."""
        if not self.coordination_enabled:
            return False
        distance = manhattan_distance(ghost_pos, self.game.pacman.pos)
        return distance <= self.coordination_vision

    def _pick_search_target(self, ghost_idx: int, blocked_targets: set):
        """Assign a search waypoint so ghosts spread out when Pac-Man is unseen."""
        ghost = self.game.ghosts[ghost_idx]
        candidates = []
        step = max(3, self.grid_size // 5)
        center = self.grid_size // 2

        for x in range(1, self.grid_size - 1, step):
            for y in range(1, self.grid_size - 1, step):
                pos = (x, y)
                if pos in self.game.walls or pos in blocked_targets:
                    continue
                if pos in self.search_visited[ghost_idx]:
                    continue
                path = bfs_path(ghost.pos, pos, self.game.walls)
                if not path:
                    continue
                spread_score = min(
                    [manhattan_distance(pos, other.pos) for j, other in enumerate(self.game.ghosts) if j != ghost_idx] or [0]
                )
                center_score = manhattan_distance(pos, (center, center))
                score = spread_score + 0.2 * center_score + 0.05 * len(path)
                candidates.append((score, pos))

        if not candidates:
            for x in range(1, self.grid_size - 1):
                for y in range(1, self.grid_size - 1):
                    pos = (x, y)
                    if pos in self.game.walls or pos in blocked_targets:
                        continue
                    path = bfs_path(ghost.pos, pos, self.game.walls)
                    if path:
                        candidates.append((len(path), pos))

        if not candidates:
            return ghost.pos

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _search_actions(self, base_actions: list):
        """Coordinate exploratory search when Pac-Man is outside team vision."""
        coordinated_actions = list(base_actions)
        blocked_targets = set()

        for i, ghost in enumerate(self.game.ghosts):
            current_target = self.search_targets[i]
            target_reached = current_target is not None and ghost.pos == current_target
            target_blocked = current_target is not None and current_target in self.game.walls
            needs_new_target = (
                current_target is None or
                self.search_steps[i] <= 0 or
                target_reached or
                target_blocked or
                current_target in blocked_targets
            )

            if needs_new_target:
                current_target = self._pick_search_target(i, blocked_targets)
                self.search_targets[i] = current_target
                self.search_steps[i] = self.search_target_refresh

            blocked_targets.add(current_target)
            self.search_visited[i].append(current_target)
            path = bfs_path(ghost.pos, current_target, self.game.walls)

            if path and len(path) > 1:
                next_pos = path[1]
                dx, dy = next_pos[0] - ghost.pos[0], next_pos[1] - ghost.pos[1]
                if dx == 1:
                    coordinated_actions[i] = ACTION_RIGHT
                elif dx == -1:
                    coordinated_actions[i] = ACTION_LEFT
                elif dy == 1:
                    coordinated_actions[i] = ACTION_DOWN
                elif dy == -1:
                    coordinated_actions[i] = ACTION_UP
            self.search_steps[i] = max(0, self.search_steps[i] - 1)

        return coordinated_actions

    def _communication_actions(self, states: list, base_actions: list, training: bool = True):
        """Coordinate trapping when Pac-Man is seen and distributed search otherwise."""
        if not self.coordination_enabled:
            return base_actions

        pm = self.game.pacman.pos
        ghosts = self.game.ghosts
        walls = self.game.walls
        num = len(ghosts)

        # Partial observability: if no ghost sees Pac-Man, spread out and search.
        obs_cfg = self.config.get("observation", {})
        obs_range = obs_cfg.get("range", 30)
        can_see_pm = True
        if obs_range < self.grid_size:
            can_see_pm = any(
                manhattan_distance(g.pos, pm) <= obs_range for g in ghosts
            )
            if not can_see_pm:
                return self._search_actions(base_actions)

        if num == 0:
            return base_actions

        self.search_targets = [None for _ in range(self.num_ghosts)]
        self.search_steps = [0 for _ in range(self.num_ghosts)]

        # Adaptive radius: shrink as team gets closer to PM
        avg_dist = sum(manhattan_distance(g.pos, pm) for g in ghosts) / num
        radius = max(1, min(int(avg_dist * 0.7), 5))

        # Cardinal offsets for each ghost slot
        if num == 4:
            cardinals = [(0, -radius), (radius, 0), (0, radius), (-radius, 0)]  # N,E,S,W
        elif num == 2:
            cardinals = [(-radius, 0), (radius, 0)]  # left, right
        else:
            cardinals = []
            for i in range(num):
                if i % 2 == 0:
                    cardinals.append((radius if i < num//2 else -radius, 0))
                else:
                    cardinals.append((0, radius if i < num//2 else -radius))

        coordinated_actions = list(base_actions)

        for i, ghost in enumerate(ghosts):
            ox, oy = cardinals[i % len(cardinals)]
            target = (pm[0] + ox, pm[1] + oy)

            target = (max(0, min(self.grid_size - 1, target[0])),
                     max(0, min(self.grid_size - 1, target[1])))
            if target in walls:
                for r in range(1, radius + 1):
                    tx = pm[0] + (ox // max(abs(ox), 1) * max(0, abs(ox) - r))
                    ty = pm[1] + (oy // max(abs(oy), 1) * max(0, abs(oy) - r))
                    tx = max(0, min(self.grid_size - 1, tx))
                    ty = max(0, min(self.grid_size - 1, ty))
                    if (tx, ty) not in walls:
                        target = (tx, ty)
                        break

            path = bfs_path(ghost.pos, target, walls)
            if path and len(path) > 1:
                next_pos = path[1]
                dx, dy = next_pos[0] - ghost.pos[0], next_pos[1] - ghost.pos[1]
            else:
                path = bfs_path(ghost.pos, pm, walls)
                if path and len(path) > 1:
                    next_pos = path[1]
                    dx, dy = next_pos[0] - ghost.pos[0], next_pos[1] - ghost.pos[1]
                else:
                    dx, dy = 0, 0

            if dx == 1:
                coordinated_actions[i] = ACTION_RIGHT
            elif dx == -1:
                coordinated_actions[i] = ACTION_LEFT
            elif dy == 1:
                coordinated_actions[i] = ACTION_DOWN
            elif dy == -1:
                coordinated_actions[i] = ACTION_UP
            else:
                coordinated_actions[i] = base_actions[i]

        return coordinated_actions
    
    def step(self, actions: list):
        """
        Execute one step with given ghost actions.
        Returns: (next_states, rewards, done, info)
        """
        # Update timer (fixes timer not updating in training)
        self.game.elapsed_time = (pygame.time.get_ticks() - self.game.start_time) / 1000.0
        
        # Store old positions for collision detection and rewards
        old_ghost_positions = [g.pos for g in self.game.ghosts]
        old_pacman_pos = self.game.pacman.pos
        old_pacman_neighbors = self._pacman_neighbors(old_pacman_pos)
        old_team_dist = self._average_team_distance(old_pacman_pos, old_ghost_positions)
        old_cluster_score = self._cluster_score(old_ghost_positions)
        
        # Move ghosts with RL actions
        valid_moves = []
        occupied_positions = set(g.pos for g in self.game.ghosts)
        prev_positions = list(old_ghost_positions)
        line_axis = None
        xs = [pos[0] for pos in prev_positions]
        ys = [pos[1] for pos in prev_positions]
        if len(prev_positions) >= 3:
            if len(set(xs)) == 1:
                line_axis = "x"
            elif len(set(ys)) == 1:
                line_axis = "y"
        
        for i, (ghost, action) in enumerate(zip(self.game.ghosts, actions)):
            # Temporarily remove current ghost from occupied set
            occupied_positions.remove(ghost.pos)
            
            new_pos, valid = execute_action(ghost.pos, action, self.game.walls, self.grid_size)
            
            line_locked = False
            if line_axis == "x" and new_pos[0] == ghost.pos[0]:
                line_locked = True
            elif line_axis == "y" and new_pos[1] == ghost.pos[1]:
                line_locked = True
            
            occupied_blocked = new_pos in occupied_positions
            repeat_triggered = len(self.repeat_positions[i]) == self.repeat_window and new_pos in self.repeat_positions[i]
            
            if repeat_triggered:
                self.repeat_strike_count[i] += 1
            else:
                self.repeat_strike_count[i] = 0
            
            needs_alternative = repeat_triggered or occupied_blocked or line_locked
            
            if needs_alternative:
                if repeat_triggered and self.repeat_strike_count[i] >= self.repeat_strike_limit:
                    new_pos = ghost.pos
                    valid = False
                else:
                    alternative_actions = [a for a in ACTIONS if a != action]
                    np.random.shuffle(alternative_actions)
                    found_alternative = False
                    for alt_action in alternative_actions:
                        candidate_pos, candidate_valid = execute_action(ghost.pos, alt_action, self.game.walls, self.grid_size)
                        
                        if line_axis == "x" and candidate_pos[0] == ghost.pos[0]:
                            candidate_valid = False
                        elif line_axis == "y" and candidate_pos[1] == ghost.pos[1]:
                            candidate_valid = False
                        
                        # Check collision with other ghosts for alternative move too
                        if candidate_pos in occupied_positions:
                            candidate_valid = False
                            
                        if candidate_valid and candidate_pos not in self.repeat_positions[i]:
                            new_pos, valid = candidate_pos, candidate_valid
                            found_alternative = True
                            break
                    
                    if not found_alternative:
                        if line_locked and not repeat_triggered and not occupied_blocked:
                            valid = True
                        else:
                            new_pos = ghost.pos
                            valid = False
            
            self.repeat_positions[i].append(new_pos)
            ghost.pos = new_pos
            occupied_positions.add(new_pos)
            valid_moves.append(valid)
        
        # Move Pac-Man (heuristic AI)
        next_pm_pos = self.game.pacman.get_next_move(self.game.pellets, self.game.ghosts)
        if next_pm_pos:
            self.game.pacman.move(next_pm_pos)
            
        # Log moves if enabled
        if self.log_moves_enabled and self.log_file:
            try:
                # Capture event status BEFORE checking collision/done flags, but we need
                # to know if collision happened in this step. The collision logic is below.
                # So we defer writing until we calculate collision.
                pass 
            except Exception as e:
                logger.error(f"Failed to write log: {e}")
        
        # FIXED: Collision detection AFTER both move (catches position swaps)
        # Check 3 cases:
        # 1. Same cell now (ghost.pos == pacman.pos)
        # 2. Ghost landed on Pac-Man's old position
        # 3. Pac-Man landed on ghost's old position
        collision = False
        for i, ghost in enumerate(self.game.ghosts):
            if ghost.pos == self.game.pacman.pos:  # Same cell
                collision = True
                break
            if ghost.pos == old_pacman_pos:  # Ghost caught Pac-Man
                collision = True
                break
            if self.game.pacman.pos == old_ghost_positions[i]:  # Pac-Man ran into ghost
                collision = True
                break
        
        # Check pellet collection
        for pellet in self.game.pellets:
            if pellet.active and pellet.pos == self.game.pacman.pos:
                pellet.active = False
                self.game.pacman.pellets_collected += 1
        
        # Check win/lose (including time limit)
        pellets_done = all(not p.active for p in self.game.pellets)
        time_expired = self.game.elapsed_time >= self.game.time_limit
        
        # Log moves NOW that we know the outcome
        if self.log_moves_enabled and self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    ghost_pos_str = " | ".join([f"{str(g.pos):<8}" for g in self.game.ghosts])
                    pm_pos = f"{str(self.game.pacman.pos):<8}"
                    event = ""
                    if collision: event = "COLLISION"
                    elif pellets_done: event = "PACMAN_WIN"
                    elif time_expired: event = "TIMEOUT"
                    
                    line = f"{self.episode:7d} | {int(self.game.elapsed_time*10):4d} | {pm_pos} | {ghost_pos_str} | {event}\n"
                    f.write(line)
            except Exception as e:
                logger.error(f"Failed to write log: {e}")

        if time_expired:
            self.game.time_expired = True  # Mark for UI/popup
        done = collision or pellets_done or time_expired
        
        # Compute rewards
        # Time expired = Pac-Man survived = ghosts lose (negative reward)
        pellets_failed = pellets_done or time_expired
        rewards = self.agents.compute_rewards(
            old_ghost_positions,
            [g.pos for g in self.game.ghosts],
            self.game.pacman.pos,
            collision,
            pellets_failed,
            valid_moves
        )
        
        # Team shaping rewards to avoid line-following + encourage trapping
        rewards = self._apply_team_rewards(
            rewards=rewards,
            old_positions=old_ghost_positions,
            new_positions=[g.pos for g in self.game.ghosts],
            old_pm_pos=old_pacman_pos,
            new_pm_pos=self.game.pacman.pos,
            old_neighbors=old_pacman_neighbors,
            old_team_dist=old_team_dist,
            old_cluster_score=old_cluster_score
        )
        
        # Get next states
        next_states = self._get_states()
        
        # Info
        info = {
            "collision": collision,
            "pellets_done": pellets_done,
            "time_expired": time_expired,
            "pacman_won": pellets_done or time_expired,
            "ghosts_won": collision,
        }
        
        # Render if enabled
        if self.render_mode:
            self.game.draw()
            self.clock.tick(self.fps)
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
        
        return next_states, rewards, done, info
    
    def _pacman_neighbors(self, pacman_pos):
        """Count valid moves Pac-Man has from position."""
        neighbors = 0
        for dx, dy in DIRECTIONS:
            nx, ny = pacman_pos[0] + dx, pacman_pos[1] + dy
            if is_valid_position((nx, ny), self.game.walls):
                neighbors += 1
        return neighbors

    def _average_team_distance(self, pacman_pos, ghost_positions):
        if not ghost_positions:
            return 0
        return sum(manhattan_distance(p, pacman_pos) for p in ghost_positions) / len(ghost_positions)

    def _cluster_score(self, ghost_positions):
        if not ghost_positions:
            return 0
        # Higher score means more clustered
        total = 0
        count = 0
        for i in range(len(ghost_positions)):
            for j in range(i + 1, len(ghost_positions)):
                total += manhattan_distance(ghost_positions[i], ghost_positions[j])
                count += 1
        if count == 0:
            return 0
        avg_dist = total / count
        return avg_dist

    def _line_penalty(self, ghost_positions):
        if len(ghost_positions) < 3:
            return 0
        xs = [p[0] for p in ghost_positions]
        ys = [p[1] for p in ghost_positions]
        if len(set(xs)) == 1 or len(set(ys)) == 1:
            return 1
        return 0

    def _mobility_reduction(self, old_neighbors, new_neighbors):
        return max(0, old_neighbors - new_neighbors)

    def _apply_team_rewards(self, rewards, old_positions, new_positions,
                             old_pm_pos, new_pm_pos, old_neighbors,
                             old_team_dist, old_cluster_score):
        cfg = self.team_reward_cfg
        num = len(new_positions)
        if num == 0:
            return rewards

        # Mobility reduction reward: fewer escape moves for Pac-Man
        new_neighbors = self._pacman_neighbors(new_pm_pos)
        mobility_delta = self._mobility_reduction(old_neighbors, new_neighbors)
        if mobility_delta > 0:
            bonus = cfg.get("mobility", 1.5) * mobility_delta
            rewards = [r + bonus for r in rewards]

        # Adjacent block reward: ghosts adjacent to Pac-Man
        adj_bonus = cfg.get("adjacent_block", 0.75)
        for i, pos in enumerate(new_positions):
            if manhattan_distance(pos, new_pm_pos) == 1:
                rewards[i] += adj_bonus

        # Spread reward: encourage coverage (avg distance between ghosts)
        cluster_score = self._cluster_score(new_positions)
        spread_bonus = cfg.get("spread", 0.5)
        if cluster_score > old_cluster_score:
            rewards = [r + spread_bonus for r in rewards]

        # Cluster penalty: if ghosts too close together
        cluster_dist = cfg.get("cluster_distance", 1)
        cluster_penalty = cfg.get("cluster_penalty", -0.6)
        for i in range(len(new_positions)):
            for j in range(i + 1, len(new_positions)):
                if manhattan_distance(new_positions[i], new_positions[j]) <= cluster_dist:
                    rewards[i] += cluster_penalty
                    rewards[j] += cluster_penalty

        # Line penalty: discourage straight-line formation
        if self._line_penalty(new_positions):
            line_penalty = cfg.get("line_penalty", -1.2)
            rewards = [r + line_penalty for r in rewards]

            # If ghosts line up, nudge them to break the line
            xs = [p[0] for p in new_positions]
            ys = [p[1] for p in new_positions]
            line_axis = "x" if len(set(xs)) == 1 else "y"
            for i, pos in enumerate(new_positions):
                if line_axis == "x":
                    if pos[0] > new_pm_pos[0] and ACTION_RIGHT in ACTIONS:
                        rewards[i] += line_penalty * 0.5
                    elif pos[0] < new_pm_pos[0] and ACTION_LEFT in ACTIONS:
                        rewards[i] += line_penalty * 0.5
                else:
                    if pos[1] > new_pm_pos[1] and ACTION_DOWN in ACTIONS:
                        rewards[i] += line_penalty * 0.5
                    elif pos[1] < new_pm_pos[1] and ACTION_UP in ACTIONS:
                        rewards[i] += line_penalty * 0.5

        # Overlap penalty: two ghosts in same cell
        overlap_penalty = cfg.get("overlap_penalty", -0.4)
        seen = {}
        for i, pos in enumerate(new_positions):
            if pos in seen:
                rewards[i] += overlap_penalty
                rewards[seen[pos]] += overlap_penalty
            else:
                seen[pos] = i

        return rewards

    def render(self):
        """Manually render current frame."""
        if self.render_mode:
            self.game.draw()
            self.clock.tick(self.fps)
    
    def train_episode(self, max_steps: int = None):
        """Run one training episode."""
        if max_steps is None:
            max_steps = self.config.get("training", {}).get("max_steps_per_episode", 500)
        
        states = self.reset()
        total_rewards = [0] * self.num_ghosts
        
        for step in range(max_steps):
            # Get actions for all ghosts
            actions = self.agents.get_actions(states, training=True)
            actions = self._communication_actions(states, actions, training=True)
            
            # Step environment
            next_states, rewards, done, info = self.step(actions)
            
            # Update model
            self.agents.update_all(states, actions, rewards, next_states, done)
            
            # Track rewards
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            
            states = next_states
            
            if done:
                break
        
        # Finalize on-policy updates for truncated episodes if supported
        if hasattr(self.agents, "finish_episode"):
            self.agents.finish_episode()

        # Decay exploration for algorithms that support it
        if hasattr(self.agents, "decay_all"):
            self.agents.decay_all()
        
        # Update stats
        if info["ghosts_won"]:
            self.wins_ghosts += 1
        if info["pacman_won"]:
            self.wins_pacman += 1
        
        # Track episode data
        self.episode_lengths.append(step + 1)
        self.episode_rewards_history.append(sum(total_rewards))
        
        return {
            "total_reward": sum(total_rewards),
            "steps": step + 1,
            "ghosts_won": info["ghosts_won"],
            "pacman_won": info["pacman_won"],
            "collision": info["collision"],
        }
    
    def evaluate_episode(self, max_steps: int = None, map_idx: Optional[int] = None):
        """Run one evaluation episode on test map (no learning)."""
        if max_steps is None:
            max_steps = self.config.get("training", {}).get("max_steps_per_episode", 500)
        
        states = self.reset_for_test(map_idx)
        total_rewards = [0] * self.num_ghosts
        
        for step in range(max_steps):
            # Get actions for all ghosts (no training)
            actions = self.agents.get_actions(states, training=False)
            actions = self._communication_actions(states, actions, training=False)
            
            # Step environment
            next_states, rewards, done, info = self.step(actions)
            
            # Track rewards
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            
            states = next_states
            
            if done:
                break
        
        # Update evaluation stats
        if info["ghosts_won"]:
            self.eval_wins_ghosts += 1
        if info["pacman_won"]:
            self.eval_wins_pacman += 1
        
        self.eval_episode_lengths.append(step + 1)
        self.eval_rewards_history.append(sum(total_rewards))
        
        return {
            "total_reward": sum(total_rewards),
            "steps": step + 1,
            "ghosts_won": info["ghosts_won"],
            "pacman_won": info["pacman_won"],
            "collision": info["collision"],
        }
    
    def run_evaluation(self, episodes: int = 100, map_idx: Optional[int] = None,
                       render_every: int = 0):
        """Run evaluation-only mode on test map (no learning)."""
        test_idx = map_idx if map_idx is not None else self.test_map
        test_name = self._get_map_name(test_idx)
        
        logger.info("=" * 60)
        logger.info("EVALUATION MODE")
        logger.info("=" * 60)
        logger.info(f"[EVAL] Episodes: {episodes}")
        logger.info(f"[EVAL] Test Map: {test_name} (index {test_idx})")
        logger.info(f"[EVAL] Ghosts: {self.num_ghosts}")
        logger.info("=" * 60)
        
        # Ensure models are loaded before eval (no training)
        if self.agents is None:
            logger.warning("[EVAL] Agents not initialized.")
            return
        
        # Run evaluation loop
        for ep in range(episodes):
            self.episode = ep
            
            # Enable render for this episode?
            should_render = render_every > 0 and (ep % render_every == 0 or render_every == 1)
            was_rendering = self.render_mode
            self.render_mode = should_render
            
            stats = self.evaluate_episode(map_idx=test_idx)
            
            # Restore render mode
            self.render_mode = was_rendering
            
            if (ep + 1) % 50 == 0:
                avg_reward = np.mean(self.eval_rewards_history[-50:]) if self.eval_rewards_history else 0
                win_rate = self.eval_wins_ghosts / (self.eval_wins_ghosts + self.eval_wins_pacman + 1) * 100
                logger.info(f"[EVAL] Ep {ep+1:5d} | AvgR: {avg_reward:8.1f} | GhostWin: {win_rate:5.1f}%")
        
        # Summary
        total_games = self.eval_wins_ghosts + self.eval_wins_pacman
        ghost_win_rate = (self.eval_wins_ghosts / total_games * 100) if total_games > 0 else 0
        pacman_win_rate = (self.eval_wins_pacman / total_games * 100) if total_games > 0 else 0
        
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"[EVAL] Ghost Wins: {self.eval_wins_ghosts} ({ghost_win_rate:.1f}%)")
        logger.info(f"[EVAL] Pac-Man Wins: {self.eval_wins_pacman} ({pacman_win_rate:.1f}%)")
        logger.info(f"[EVAL] Total Games: {total_games}")
        
        if self.eval_episode_lengths:
            avg_length = np.mean(self.eval_episode_lengths)
            logger.info(f"[EVAL] Avg Episode Length: {avg_length:.1f} steps")
        
        if self.eval_rewards_history:
            avg_reward = np.mean(self.eval_rewards_history)
            logger.info(f"[EVAL] Avg Reward: {avg_reward:.1f}")
        
        logger.info("=" * 60)
        return {
            "ghost_wins": self.eval_wins_ghosts,
            "pacman_wins": self.eval_wins_pacman,
            "total_games": total_games,
            "ghost_win_rate": ghost_win_rate,
            "pacman_win_rate": pacman_win_rate,
        }
    
    def run(self, episodes: int = 1000, save_every: int = 100, 
            save_dir: str = "models/", load_dir: Optional[str] = None,
            render_every: int = 0):
        """
        Run full training.
        render_every: 0=never, 1=always, N=every N episodes
        """
        self.save_dir = save_dir  # Store for summary
        
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        if len(self.train_maps) > 1 or self.train_random_map:
            logger.info(f"[RUN] Training Maps: {self.train_maps}")
        else:
            logger.info(f"[RUN] Training Map:  {self.train_map_name} (index {self.train_map})")
        logger.info(f"[RUN] Train Grids:   {self.train_grid_sizes}")
        logger.info(f"[RUN] Test Map:      {self.test_map_name} (index {self.test_map})")
        logger.info(f"[RUN] Ghosts:        {self.num_ghosts}")
        logger.info(f"[RUN] Episodes:      {episodes}")
        logger.info(f"[RUN] Save every:   {save_every} episodes")
        logger.info(f"[RUN] Save dir:     {save_dir}")
        if render_every > 0:
            logger.info(f"[RUN] Rendering:    every {render_every} episode(s)")
        else:
            logger.info("[RUN] Rendering:    disabled (headless)")
        logger.info("=" * 60)
        
        # Load existing models if provided
        if load_dir and os.path.exists(load_dir):
            logger.info(f"[RUN] Loading existing models from: {load_dir}")
            self.agents.load_all(load_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for ep in range(episodes):
            self.episode = ep
            
            # Enable render for this episode?
            should_render = render_every > 0 and (ep % render_every == 0 or render_every == 1)
            was_rendering = self.render_mode
            self.render_mode = should_render
            
            stats = self.train_episode()
            
            # Restore render mode
            self.render_mode = was_rendering
            
            # Progress logging (every 50 episodes)
            if (ep + 1) % 50 == 0:
                recent_rewards = self.episode_rewards_history[-50:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                win_rate = self.wins_ghosts / (self.wins_ghosts + self.wins_pacman + 1) * 100
                
                # Build progress message based on model type
                base_msg = f"[PROGRESS] Ep {ep+1:5d} | AvgR: {avg_reward:8.1f} | GhostWin: {win_rate:5.1f}%"
                
                # Check if model uses epsilon for exploration
                if hasattr(self.agents, "uses_epsilon") and self.agents.uses_epsilon():
                    epsilon_val = self.agents.get_epsilon()
                    logger.info(f"{base_msg} | ε: {epsilon_val:.3f}")
                elif self.model_type == "ppo" and hasattr(self.agents, "get_training_stats"):
                    # Show PPO-specific stats
                    stats = self.agents.get_training_stats()
                    logger.info(f"{base_msg} | PLoss: {stats['policy_loss']:.3f} | VLoss: {stats['value_loss']:.3f} | Ent: {stats['entropy']:.3f}")
                else:
                    logger.info(base_msg)
            
            # Save checkpoint
            if (ep + 1) % save_every == 0:
                self.agents.save_all(save_dir)
                logger.info(f"[CHECKPOINT] Saved at episode {ep+1}")
        
        # Final save
        self.agents.save_all(save_dir)
        logger.info(f"[SAVE] Final models saved to: {save_dir}")
        
        # Print comprehensive summary report
        self._print_summary(episodes)
    
    def _print_summary(self, total_episodes: int):
        """Print comprehensive training summary with all parameters and model info."""
        from rl_utils import (REWARD_COLLISION, REWARD_PELLETS_EATEN, 
                              REWARD_CLOSER, REWARD_FARTHER, REWARD_INVALID,
                              REWARD_REPEAT, ACTIONS, ACTION_NAMES,
                              REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD)
        
        total_games = self.wins_ghosts + self.wins_pacman
        ghost_win_rate = (self.wins_ghosts / total_games * 100) if total_games > 0 else 0
        pacman_win_rate = (self.wins_pacman / total_games * 100) if total_games > 0 else 0
        
        # Calculate final Q-table stats (Q-learning only)
        total_states = 0
        avg_states = 0
        if hasattr(self.agents, "agents") and self.model_type == "qlearning":
            total_states = sum(len(a.q_table) for a in self.agents.agents)
            avg_states = total_states / len(self.agents.agents) if self.agents.agents else 0
        
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY REPORT")
        print(f"{'='*70}")
        
        # =====================================================================
        # TRAINING CONFIGURATION
        # =====================================================================
        print(f"\n┌─ TRAINING CONFIGURATION {'─'*44}┐")
        training_map_label = str(self.train_maps) if (self.train_random_map or len(self.train_maps) > 1) else self.train_map_name
        print(f"│  Training Map(s):  {training_map_label:<40}│")
        print(f"│  Test Map:         {self.test_map_name:<40}│")
        print(f"│  Grid Size(s):     {str(self.train_grid_sizes):<40}│")
        print(f"│  Total Episodes:   {total_episodes:<40}│")
        print(f"│  Number of Ghosts: {self.num_ghosts:<40}│")
        print(f"│  Max Steps/Ep:     {500:<40}│")
        print(f"│  Save Directory:   {self.save_dir:<40}│")
        print(f"└{'─'*70}┘")
        
        # =====================================================================
        # MODEL HYPERPARAMETERS
        # =====================================================================
        if self.model_type == "qlearning":
            print(f"\n┌─ Q-LEARNING HYPERPARAMETERS {'─'*40}┐")
            print(f"│  Alpha (α):        {self.hyperparams['alpha']:<40.3f}│")
            print(f"│  Gamma (γ):        {self.hyperparams['gamma']:<40.3f}│")
            print(f"│  Epsilon Start:    {self.hyperparams['epsilon_start']:<40.3f}│")
            print(f"│  Epsilon Min:      {self.hyperparams['epsilon_min']:<40.3f}│")
            print(f"│  Epsilon Decay:    {self.hyperparams['epsilon_decay']:<40.3f}│")
            print(f"│  Use NOOP Action:  {self.hyperparams['use_noop']:<40}│")
            print(f"└{'─'*70}┘")
        elif self.model_type == "maddpg":
            maddpg_cfg = self.config.get("maddpg", {})
            print(f"\n┌─ MADDPG HYPERPARAMETERS {'─'*43}┐")
            print(f"│  Actor LR:         {maddpg_cfg.get('actor_lr', 0.0005):<40}│")
            print(f"│  Critic LR:        {maddpg_cfg.get('critic_lr', 0.001):<40}│")
            print(f"│  Gamma (γ):        {maddpg_cfg.get('gamma', 0.95):<40}│")
            print(f"│  Tau:              {maddpg_cfg.get('tau', 0.01):<40}│")
            print(f"│  Batch Size:       {maddpg_cfg.get('batch_size', 128):<40}│")
            print(f"│  Buffer Size:      {maddpg_cfg.get('buffer_size', 100000):<40}│")
            print(f"│  Update Every:     {maddpg_cfg.get('update_every', 1):<40}│")
            print(f"│  Epsilon Start:    {maddpg_cfg.get('epsilon_start', 0.2):<40}│")
            print(f"│  Epsilon Min:      {maddpg_cfg.get('epsilon_min', 0.02):<40}│")
            print(f"│  Epsilon Decay:    {maddpg_cfg.get('epsilon_decay', 0.995):<40}│")
            print(f"│  Hidden Sizes:     {str(maddpg_cfg.get('hidden_sizes', [128, 128])):<40}│")
            print(f"└{'─'*70}┘")
        elif self.model_type == "ppo":
            print(f"\n┌─ PPO HYPERPARAMETERS {'─'*46}┐")
            print(f"│  LR:               {self.hyperparams['lr']:<40}│")
            print(f"│  Gamma (γ):        {self.hyperparams['gamma']:<40}│")
            print(f"│  GAE Lambda:       {self.hyperparams['gae_lambda']:<40}│")
            print(f"│  Clip Epsilon:     {self.hyperparams['clip_epsilon']:<40}│")
            print(f"│  Value Coef:       {self.hyperparams['value_coef']:<40}│")
            print(f"│  Entropy Coef:     {self.hyperparams['entropy_coef']:<40}│")
            print(f"│  Update Epochs:    {self.hyperparams['update_epochs']:<40}│")
            print(f"│  Minibatch Size:   {self.hyperparams['minibatch_size']:<40}│")
            print(f"│  Max Grad Norm:    {self.hyperparams['max_grad_norm']:<40}│")
            print(f"│  Hidden Sizes:     {str(self.hyperparams['hidden_sizes']):<40}│")
            print(f"└{'─'*70}┘")
        
        # =====================================================================
        # REWARD STRUCTURE
        # =====================================================================
        print(f"\n┌─ REWARD STRUCTURE {'─'*50}┐")
        print(f"│  Collision (any ghost catches PM):  {REWARD_COLLISION:>8} │")
        print(f"│  Pellets Done (PM wins):            {REWARD_PELLETS_EATEN:>8} │")
        print(f"│  Moved Closer to Pac-Man:           {REWARD_CLOSER:>8} │")
        print(f"│  Moved Farther from Pac-Man:        {REWARD_FARTHER:>8} │")
        print(f"│  Invalid Move (wall):               {REWARD_INVALID:>8} │")
        print(f"│  Repeat Pattern Penalty:            {REWARD_REPEAT:>8} │")
        print(f"└{'─'*70}┘")
        
        # =====================================================================
        # STATE SPACE
        # =====================================================================
        state_count_label = (
            f"{total_states} (avg {avg_states:.0f}/ghost)"
            if self.model_type == "qlearning"
            else "n/a (neural function approximator)"
        )
        print(f"\n┌─ STATE SPACE {'─'*55}┐")
        print(f"│  State Encoding:   (dist_pm, dir_pm, dist_ghost, dir_ghost, │")
        print(f"│                    can_up, can_down, can_left, can_right,  │")
        print(f"│                    pellet_near)                            │")
        print(f"│  Grid Size:        {f'{self.grid_size}x{self.grid_size}':<40}│")
        print(f"│  Distance Bins:    {str(self.config.get('state', {}).get('dist_bins', [0, 2, 4, 8, 16, 30])):<40}│")
        print(f"│  Directions:       UP, DOWN, LEFT, RIGHT (+ NOOP if enabled)│")
        print(f"│  Repeat Window:    {REPEAT_PATTERN_WINDOW:<40}│")
        print(f"│  Repeat Threshold: {REPEAT_PATTERN_THRESHOLD:<40}│")
        print(f"│  Total Q-States:   {state_count_label:<40}│")
        print(f"└{'─'*70}┘")

        # =====================================================================
        # COORDINATION
        # =====================================================================
        coord_cfg = self.config.get("coordination", {})
        print(f"\n┌─ COORDINATION {'─'*54}┐")
        print(f"│  Enabled:          {coord_cfg.get('enabled', True):<40}│")
        print(f"│  Vision Range:     {coord_cfg.get('vision_range', 8):<40}│")
        print(f"│  Surround Targets: {coord_cfg.get('surround', True):<40}│")
        print(f"│  Search Refresh:   {coord_cfg.get('search_target_refresh', 12):<40}│")
        print(f"│  Search History:   {coord_cfg.get('search_history_limit', 6):<40}│")
        print(f"└{'─'*70}┘")
        
        # =====================================================================
        # RESULTS
        # =====================================================================
        print(f"\n┌─ RESULTS {'─'*59}┐")
        print(f"│  Ghost Wins:       {self.wins_ghosts:>6} ({ghost_win_rate:>5.1f}%)                         │")
        print(f"│  Pac-Man Wins:     {self.wins_pacman:>6} ({pacman_win_rate:>5.1f}%)                         │")
        print(f"│  Total Games:      {total_games:>6}                                         │")
        print(f"└{'─'*70}┘")
        
        # =====================================================================
        # EPISODE STATISTICS
        # =====================================================================
        if self.episode_lengths:
            avg_length = np.mean(self.episode_lengths)
            min_length = np.min(self.episode_lengths)
            max_length = np.max(self.episode_lengths)
            std_length = np.std(self.episode_lengths)
            print(f"\n┌─ EPISODE LENGTH STATISTICS {'─'*41}┐")
            print(f"│  Average:          {avg_length:>10.1f} steps                              │")
            print(f"│  Std Dev:          {std_length:>10.1f} steps                              │")
            print(f"│  Min:              {min_length:>10} steps                              │")
            print(f"│  Max:              {max_length:>10} steps                              │")
            print(f"│  Total Steps:      {sum(self.episode_lengths):>10,}                                │")
            print(f"└{'─'*70}┘")
        
        # =====================================================================
        # REWARD STATISTICS
        # =====================================================================
        if self.episode_rewards_history:
            avg_reward = np.mean(self.episode_rewards_history)
            final_100_avg = np.mean(self.episode_rewards_history[-100:]) if len(self.episode_rewards_history) >= 100 else avg_reward
            first_100_avg = np.mean(self.episode_rewards_history[:100]) if len(self.episode_rewards_history) >= 100 else avg_reward
            max_reward = np.max(self.episode_rewards_history)
            min_reward = np.min(self.episode_rewards_history)
            print(f"\n┌─ REWARD STATISTICS {'─'*49}┐")
            print(f"│  Average Total:    {avg_reward:>12.1f}                                     │")
            print(f"│  First 100 Avg:    {first_100_avg:>12.1f}                                     │")
            print(f"│  Final 100 Avg:    {final_100_avg:>12.1f}                                     │")
            print(f"│  Max Episode:      {max_reward:>12.1f}                                     │")
            print(f"│  Min Episode:      {min_reward:>12.1f}                                     │")
            print(f"└{'─'*70}┘")
        
        # =====================================================================
        # PER-GHOST DETAILED STATS
        # =====================================================================
        if self.model_type == "qlearning":
            print(f"\n┌─ PER-GHOST STATISTICS {'─'*46}┐")
            print(f"│  {'Ghost':<8} {'Avg Reward':>12} {'States':>10} {'ε Final':>10} {'Repeats':>10} │")
            print(f"│  {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*10} │")
            for i, agent in enumerate(self.agents.agents):
                if agent.episode_rewards:
                    avg_r = np.mean(agent.episode_rewards)
                    states = len(agent.q_table)
                    repeat_count = getattr(agent, "repeat_penalty_count", 0)
                    print(f"│  {i:<8} {avg_r:>12.1f} {states:>10} {agent.epsilon:>10.3f} {repeat_count:>10} │")
            print(f"└{'─'*70}┘")
        
        # =====================================================================
        # MODEL FILES
        # =====================================================================
        print(f"\n┌─ SAVED MODEL FILES {'─'*49}┐")
        if self.model_type == "qlearning":
            for i in range(self.num_ghosts):
                filename = f"ghost_{i}_qtable.json"
                filepath = os.path.join(self.save_dir, filename)
                if os.path.exists(filepath):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"│  {filename:<30} ({size_kb:>6.1f} KB)                 │")
                else:
                    print(f"│  {filename:<30} (NOT FOUND)                        │")
        elif self.model_type == "maddpg":
            for i in range(self.num_ghosts):
                actor = f"ghost_{i}_actor.pt"
                critic = f"ghost_{i}_critic.pt"
                for filename in (actor, critic):
                    filepath = os.path.join(self.save_dir, filename)
                    if os.path.exists(filepath):
                        size_kb = os.path.getsize(filepath) / 1024
                        print(f"│  {filename:<30} ({size_kb:>6.1f} KB)                 │")
                    else:
                        print(f"│  {filename:<30} (NOT FOUND)                        │")
        elif self.model_type == "dqn":
            filename = "shared_dqn.pt"
            filepath = os.path.join(self.save_dir, filename)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"│  {filename:<30} ({size_kb:>6.1f} KB)                 │")
            else:
                print(f"│  {filename:<30} (NOT FOUND)                        │")
        elif self.model_type == "ppo":
            filename = "shared_ppo.pt"
            filepath = os.path.join(self.save_dir, filename)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"│  {filename:<30} ({size_kb:>6.1f} KB)                 │")
            else:
                print(f"│  {filename:<30} (NOT FOUND)                        │")
        elif self.model_type == "qmix":
            filenames = ["qmix_mixing.pt"] + [f"ghost_{i}_qmix_q.pt" for i in range(self.num_ghosts)]
            for filename in filenames:
                filepath = os.path.join(self.save_dir, filename)
                if os.path.exists(filepath):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"│  {filename:<30} ({size_kb:>6.1f} KB)                 │")
                else:
                    print(f"│  {filename:<30} (NOT FOUND)                        │")
        print(f"└{'─'*70}┘")
        
        # =====================================================================
        # CROSS-MAP TESTING INFO
        # =====================================================================
        if self.train_map != self.test_map:
            print(f"\n┌─ CROSS-MAP EVALUATION {'─'*46}┐")
            print(f"│  Model trained on: {self.train_map_name:<40}│")
            print(f"│  Ready to test on: {self.test_map_name:<40}│")
            print(f"│  To evaluate: load from {self.save_dir} and run eval  │")
            print(f"└{'─'*70}┘")
        
        # =====================================================================
        # MODEL IDENTIFIER (for comparison)
        # =====================================================================
        import hashlib
        import json as json_module
        config_str = json_module.dumps({
            "train_map": "random" if self.train_random_map else self.train_map,
            "train_maps": self.train_maps,
            "train_grid_sizes": self.train_grid_sizes,
            "num_ghosts": self.num_ghosts,
            "hyperparams": self.hyperparams,
            "episodes": total_episodes,
        }, sort_keys=True)
        model_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        print(f"\n┌─ MODEL IDENTIFIER {'─'*50}┐")
        print(f"│  Model ID:         {model_id:<40}│")
        print(f"│  (Use for comparison with other experiments)               │")
        print(f"└{'─'*70}┘")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}\n")
        
        # Export results to CSV
        self._save_results_csv(total_episodes)

    def _save_results_csv(self, total_episodes: int):
        """Save training results to CSV file with model name, ghost count, and date."""
        from datetime import datetime
        import csv
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = getattr(self, 'model_type', 'unknown')
        ghosts = self.num_ghosts
        
        filename = f"results_{model}_{ghosts}ghosts_{date_str}.csv"
        
        # Calculate stats
        avg_reward = np.mean(self.episode_rewards_history) if self.episode_rewards_history else 0
        final_100_avg = np.mean(self.episode_rewards_history[-100:]) if len(self.episode_rewards_history) >= 100 else avg_reward
        ghost_win_rate = self.wins_ghosts / max(total_episodes, 1)
        pm_win_rate = self.wins_pacman / max(total_episodes, 1)
        avg_steps = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        obs_cfg = self.config.get("observation", {})
        obs_range = obs_cfg.get("range", 30)
        shared_pm = obs_cfg.get("shared_pacman", False)
        
        row = {
            "date": datetime.now().isoformat(),
            "model": model,
            "ghosts": ghosts,
            "grid_size": self.grid_size,
            "train_grid_sizes": str(getattr(self, 'train_grid_sizes', [self.grid_size])),
            "episodes": total_episodes,
            "ghost_win_rate": round(ghost_win_rate, 4),
            "pacman_win_rate": round(pm_win_rate, 4),
            "avg_reward": round(avg_reward, 2),
            "final_100_avg_reward": round(final_100_avg, 2),
            "avg_episode_steps": round(avg_steps, 2),
            "obs_range": obs_range,
            "shared_pacman": shared_pm,
            "train_map": getattr(self, 'train_map', 0),
            "train_maps": str(getattr(self, 'train_maps', [getattr(self, 'train_map', 0)])),
            "test_map": getattr(self, 'test_map', 0),
        }
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"[CSV] Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ghost Q-learning agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on default map (map 0)
  python train_ghosts.py --episodes 1000 --save-dir models/
  
  # Train on specific map (0-5)
  python train_ghosts.py --episodes 1000 --train-map 2 --save-dir models_m2/
  
  # Train on random maps
  python train_ghosts.py --episodes 1000 --train-random-map --save-dir models_random/
  
  # Train on selected maps and random grid sizes each episode
  python train_ghosts.py --episodes 1000 --train-maps 0,2,4 --train-grid-sizes 20,30,40 --save-dir models_mixed/
  
  # Train on map 0, test on map 3
  python train_ghosts.py --episodes 1000 --train-map 0 --test-map 3 --save-dir models_m0_t3/
  
  # Watch training live
  python train_ghosts.py --render --episodes 100 --fps 10
  
  # Use custom config
  python train_ghosts.py --episodes 1000 --config my_hyperparams.json --save-dir models/
        """
    )
    
    # ========================================================================
    # GAME-RELATED OPTIONS (CLI)
    # ========================================================================
    parser.add_argument("--episodes", type=int, default=1000,
                        help="[GAME] Number of training episodes (default: 1000)")
    parser.add_argument("--ghosts", type=int, default=4,
                        help="[GAME] Number of ghosts: 2 or 4 (default: 4)")
    parser.add_argument("--pellets", type=int, default=100,
                        help="[GAME] Max pellets to spawn (default: 100)")
    parser.add_argument("--time-limit", type=float, default=20.0,
                        help="[GAME] Seconds for Pac-Man time-out win (default: 20)")
    
    # ========================================================================
    # MAP SELECTION (CLI)
    # ========================================================================
    parser.add_argument("--train-map", type=int, default=0,
                        help="[MAP] Map index for training, 0-5 (default: 0)")
    parser.add_argument("--train-maps", type=str, default=None,
                        help="[MAP] Comma-separated map indices to sample from each episode, e.g. 0,2,4")
    parser.add_argument("--train-random-map", action="store_true",
                        help="[MAP] Train on random map each episode (uses all maps unless --train-maps is provided)")
    parser.add_argument("--train-grid-sizes", type=str, default=None,
                        help="[GAME] Comma-separated grid sizes to sample from each episode, e.g. 20,30,40")
    parser.add_argument("--test-map", type=int, default=None,
                        help="[MAP] Map index for testing (default: same as train-map)")
    
    # ========================================================================
    # TRAINING OPTIONS (CLI)
    # ========================================================================
    parser.add_argument("--save-dir", type=str, default="checkpoints/",
                        help="[TRAIN] Directory to save Q-tables (default: checkpoints/)")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="[TRAIN] Directory to load Q-tables from (continue training)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="[TRAIN] Save checkpoint every N episodes (default: 100)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="[TRAIN] Path to config.json for hyperparameters (default: config.json)")
    parser.add_argument("--model", type=str, default="qlearning",
                        choices=["qlearning", "maddpg", "dqn", "ppo", "qmix"],
                        help="[MODEL] Algorithm: qlearning, maddpg, dqn, ppo, or qmix (default: qlearning)")
    parser.add_argument("--eval-only", action="store_true",
                        help="[TRAIN] Run evaluation only (requires --load-dir)")
    
    # ========================================================================
    # RENDERING OPTIONS (CLI)
    # ========================================================================
    parser.add_argument("--render", action="store_true",
                        help="[RENDER] Show game window during training")
    parser.add_argument("--render-every", type=int, default=0,
                        help="[RENDER] Render every N episodes (e.g., --render-every 10)")
    parser.add_argument("--fps", type=int, default=10,
                        help="[RENDER] FPS when rendering (default: 10)")
    parser.add_argument("--log-moves", action="store_true",
                        help="[DEBUG] Log ghost and pacman moves to evaluation_moves.txt (eval-only)")
    
    args = parser.parse_args()
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if args.ghosts not in [2, 4]:
        logger.warning(f"Ghost count {args.ghosts} not standard. Using {args.ghosts}.")
    
    if args.train_map < 0 or args.train_map > 5:
        logger.warning(f"train-map {args.train_map} out of range 0-5. Using 0.")
        args.train_map = 0

    parsed_train_maps = None
    if args.train_maps:
        try:
            parsed_train_maps = [int(x.strip()) for x in args.train_maps.split(',') if x.strip() != '']
        except ValueError:
            logger.error("[MAP] --train-maps must contain comma-separated integers, e.g. 0,2,4")
            sys.exit(1)
        invalid_maps = [m for m in parsed_train_maps if m < 0 or m > 5]
        if invalid_maps:
            logger.error(f"[MAP] Invalid train maps: {invalid_maps}. Allowed range is 0-5.")
            sys.exit(1)
        if not parsed_train_maps:
            logger.error("[MAP] --train-maps was provided but no valid map indices were found.")
            sys.exit(1)

    parsed_train_grid_sizes = None
    if args.train_grid_sizes:
        try:
            parsed_train_grid_sizes = [int(x.strip()) for x in args.train_grid_sizes.split(',') if x.strip() != '']
        except ValueError:
            logger.error("[GAME] --train-grid-sizes must contain comma-separated integers, e.g. 20,30,40")
            sys.exit(1)
        invalid_grid_sizes = [g for g in parsed_train_grid_sizes if g < 10]
        if invalid_grid_sizes:
            logger.error(f"[GAME] Invalid grid sizes: {invalid_grid_sizes}. Grid size must be at least 10.")
            sys.exit(1)
        if not parsed_train_grid_sizes:
            logger.error("[GAME] --train-grid-sizes was provided but no valid grid sizes were found.")
            sys.exit(1)
    
    if args.test_map is not None and (args.test_map < 0 or args.test_map > 5):
        logger.warning(f"test-map {args.test_map} out of range 0-5. Using train-map.")
        args.test_map = args.train_map
    
    if args.eval_only and not args.load_dir:
        logger.error("[EVAL] --eval-only requires --load-dir with trained models.")
        sys.exit(1)
    
    # ========================================================================
    # INITIALIZE ENVIRONMENT
    # ========================================================================
    render = args.render or args.render_every > 0
    
    logger.info(f"[CLI] Loading config from: {args.config}")
    config = load_config(args.config)
    
    env = RLTrainingEnvironment(
        render=render, 
        fps=args.fps, 
        num_ghosts=args.ghosts,
        train_map=args.train_map,
        test_map=args.test_map,
        config=config,
        model_type=args.model,
        log_moves=args.log_moves,
        train_random_map=args.train_random_map,
        train_maps=parsed_train_maps,
        train_grid_sizes=parsed_train_grid_sizes,
    )
    
    # Log full config after init
    log_config(
        config=config,
        train_map=args.train_map,
        test_map=args.test_map if args.test_map is not None else args.train_map,
        train_map_name=env.train_map_name,
        test_map_name=env.test_map_name,
        num_ghosts=args.ghosts,
        episodes=args.episodes,
        model_type=args.model,
        train_maps=env.train_maps,
        train_grid_sizes=env.train_grid_sizes,
    )
    
    # Pass win conditions to game
    env.game.max_pellets = args.pellets
    env.game.time_limit = args.time_limit
    
    # ========================================================================
    # RUN EVALUATION ONLY (NO TRAINING)
    # ========================================================================
    if args.eval_only:
        logger.info(f"[EVAL] Loading models from: {args.load_dir}")
        env.agents.load_all(args.load_dir)
        env.run_evaluation(
            episodes=args.episodes,
            map_idx=args.test_map,
            render_every=args.render_every if args.render_every > 0 else (1 if args.render else 0)
        )
        return
    
    # ========================================================================
    # RUN TRAINING
    # ========================================================================
    logger.info(f"[TRAIN] Starting training: {args.episodes} episodes on {env.train_map_name}")
    
    env.run(
        episodes=args.episodes,
        save_every=args.save_every,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        render_every=args.render_every if args.render_every > 0 else (1 if args.render else 0)
    )
    
    logger.info(f"[TRAIN] Training complete! Models saved to {args.save_dir}")


if __name__ == "__main__":
    main()
