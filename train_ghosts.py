"""
Q-Learning Training Script for Ghost Agents
Trains ghosts to catch Pac-Man using Q-learning.

Usage:
    # Basic training (4 ghosts)
    python train_ghosts.py --episodes 1000 --save-dir models/
    
    # Train with 2 ghosts
    python train_ghosts.py --ghosts 2 --episodes 1000 --save-dir models_2g/
    
    # Continue training from saved models
    python train_ghosts.py --load-dir models/ --episodes 500
    
    # Watch training live
    python train_ghosts.py --render --episodes 100
    
    # Render every 10th episode
    python train_ghosts.py --render-every 10 --episodes 500
"""

import argparse
import os
import numpy as np
import sys
from typing import Optional

# Import game components
from pacman_game import (
    PacManGame, GRID_SIZE, NUM_GHOSTS, 
    PacMan, Ghost, Pellet, DIRECTIONS, is_valid_position
)
from ghost_rl import (
    MultiGhostQLearning, encode_state, execute_action,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
)

import pygame


class RLTrainingEnvironment:
    """Wraps PacManGame for RL training. Set render=True to watch training."""
    
    def __init__(self, render: bool = False, fps: int = 10, num_ghosts: int = 4):
        self.render_mode = render
        self.fps = fps
        self.num_ghosts = num_ghosts
        
        # Update global NUM_GHOSTS for this session
        import pacman_game
        pacman_game.NUM_GHOSTS = num_ghosts
        
        if not render:
            # Headless mode - no window
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        pygame.init()
        
        self.game = PacManGame()
        self.game.in_menu = False  # Skip menu
        self.game._load_map(0)
        self.game.reset()
        
        # Clock for rendering
        self.clock = pygame.time.Clock()
        
        # RL agents
        self.agents = MultiGhostQLearning(num_ghosts=num_ghosts)
        
        # Episode tracking
        self.episode = 0
        self.total_episodes = 0
        self.wins_pacman = 0
        self.wins_ghosts = 0
        self.episode_lengths = []
        self.episode_rewards_history = []
    
    def reset(self, new_map: bool = False):
        """Reset for new episode."""
        if new_map:
            self.game.current_map_idx = (self.game.current_map_idx + 1) % len(self.game.maps)
        self.game._load_map(self.game.current_map_idx)
        self.game.reset()
        self.agents.reset_all()
        return self._get_states()
    
    def _get_states(self):
        """Get current states for all ghosts."""
        states = []
        for i, ghost in enumerate(self.game.ghosts):
            other_ghosts = [g.pos for j, g in enumerate(self.game.ghosts) if j != i]
            state = encode_state(
                ghost.pos, 
                self.game.pacman.pos,
                other_ghosts,
                self.game.walls,
                self.game.pellets,
                GRID_SIZE
            )
            states.append(state)
        return states
    
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
        
        # Move ghosts with RL actions
        valid_moves = []
        for i, (ghost, action) in enumerate(zip(self.game.ghosts, actions)):
            new_pos, valid = execute_action(ghost.pos, action, self.game.walls, GRID_SIZE)
            ghost.pos = new_pos
            valid_moves.append(valid)
        
        # Move Pac-Man (heuristic AI)
        next_pm_pos = self.game.pacman.get_next_move(self.game.pellets, self.game.ghosts)
        if next_pm_pos:
            self.game.pacman.move(next_pm_pos)
        
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
    
    def render(self):
        """Manually render current frame."""
        if self.render_mode:
            self.game.draw()
            self.clock.tick(self.fps)
    
    def train_episode(self, max_steps: int = 500):
        """Run one training episode."""
        states = self.reset()
        total_rewards = [0] * self.num_ghosts
        
        for step in range(max_steps):
            # Get actions for all ghosts
            actions = self.agents.get_actions(states, training=True)
            
            # Step environment
            next_states, rewards, done, info = self.step(actions)
            
            # Update Q-tables
            self.agents.update_all(states, actions, rewards, next_states, done)
            
            # Track rewards
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            
            states = next_states
            
            if done:
                break
        
        # Decay exploration
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
    
    def run(self, episodes: int = 1000, save_every: int = 100, 
            save_dir: str = "models/", load_dir: Optional[str] = None,
            render_every: int = 0):
        """
        Run full training.
        render_every: 0=never, 1=always, N=every N episodes
        """
        print(f"\n{'='*60}")
        print("GHOST Q-LEARNING TRAINING")
        print(f"{'='*60}")
        print(f"Ghosts:     {self.num_ghosts}")
        print(f"Episodes:   {episodes}")
        print(f"Save every: {save_every}")
        print(f"Save dir:   {save_dir}")
        if render_every > 0:
            print(f"Rendering:  every {render_every} episode(s)")
        else:
            print("Rendering:  disabled (headless)")
        print(f"{'='*60}\n")
        
        # Load existing models if provided
        if load_dir and os.path.exists(load_dir):
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
            
            # Progress
            if (ep + 1) % 50 == 0:
                avg_reward = np.mean([sum(a.episode_rewards[-50:]) for a in self.agents.agents 
                                      if len(a.episode_rewards) >= 50])
                win_rate = self.wins_ghosts / (self.wins_ghosts + self.wins_pacman + 1) * 100
                print(f"Ep {ep+1:5d} | AvgR: {avg_reward:8.1f} | "
                      f"GhostWin: {win_rate:5.1f}% | ε: {self.agents.agents[0].epsilon:.3f}")
            
            # Save checkpoint
            if (ep + 1) % save_every == 0:
                self.agents.save_all(save_dir)
                print(f"  → Saved checkpoint at episode {ep+1}")
        
        # Final save
        self.agents.save_all(save_dir)
        
        # Print summary report
        self._print_summary(episodes)
    
    def _print_summary(self, total_episodes: int):
        """Print detailed training summary."""
        total_games = self.wins_ghosts + self.wins_pacman
        ghost_win_rate = (self.wins_ghosts / total_games * 100) if total_games > 0 else 0
        pacman_win_rate = (self.wins_pacman / total_games * 100) if total_games > 0 else 0
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY REPORT")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Ghosts:           {self.num_ghosts}")
        print(f"  Total Episodes:   {total_episodes}")
        print(f"  Models Saved to:  {self.save_dir}")
        print(f"\nResults:")
        print(f"  Ghost Wins:       {self.wins_ghosts} ({ghost_win_rate:.1f}%)")
        print(f"  Pac-Man Wins:     {self.wins_pacman} ({pacman_win_rate:.1f}%)")
        print(f"  Total Games:      {total_games}")
        
        if self.episode_lengths:
            avg_length = np.mean(self.episode_lengths)
            min_length = np.min(self.episode_lengths)
            max_length = np.max(self.episode_lengths)
            print(f"\nEpisode Lengths:")
            print(f"  Average:          {avg_length:.1f} steps")
            print(f"  Min:              {min_length} steps")
            print(f"  Max:              {max_length} steps")
        
        if self.episode_rewards_history:
            avg_reward = np.mean(self.episode_rewards_history)
            final_100_avg = np.mean(self.episode_rewards_history[-100:]) if len(self.episode_rewards_history) >= 100 else avg_reward
            print(f"\nRewards:")
            print(f"  Average Total:    {avg_reward:.1f}")
            print(f"  Final 100 Avg:    {final_100_avg:.1f}")
        
        print(f"\nPer-Ghost Stats:")
        for i, agent in enumerate(self.agents.agents):
            if agent.episode_rewards:
                avg_r = np.mean(agent.episode_rewards)
                states = len(agent.q_table)
                print(f"  Ghost {i}: AvgR={avg_r:7.1f}, States={states}, ε={agent.epsilon:.3f}")
        
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train ghost Q-learning agents")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--ghosts", type=int, default=4,
                        help="Number of ghosts (2 or 4)")
    parser.add_argument("--pellets", type=int, default=100,
                        help="Max pellets to spawn (default 100)")
    parser.add_argument("--time-limit", type=float, default=20.0,
                        help="Seconds for Pac-Man time-out win (default 20)")
    parser.add_argument("--save-dir", type=str, default="models/",
                        help="Directory to save Q-tables")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="Directory to load Q-tables from (continue training)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--render", action="store_true",
                        help="Show game window during training (slower)")
    parser.add_argument("--render-every", type=int, default=0,
                        help="Render every N episodes (e.g., --render-every 10)")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS when rendering (default 10)")
    args = parser.parse_args()
    
    # Validate ghost count
    if args.ghosts not in [2, 4]:
        print(f"Warning: Ghost count {args.ghosts} not standard. Using {args.ghosts} ghosts.")
    
    # Determine render mode
    render = args.render or args.render_every > 0
    
    env = RLTrainingEnvironment(render=render, fps=args.fps, num_ghosts=args.ghosts)
    env.save_dir = args.save_dir  # Store for summary
    
    # Pass win conditions to game
    env.game.max_pellets = args.pellets
    env.game.time_limit = args.time_limit
    
    env.run(
        episodes=args.episodes,
        save_every=args.save_every,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        render_every=args.render_every if args.render_every > 0 else (1 if args.render else 0)
    )


if __name__ == "__main__":
    main()
