#!/usr/bin/env python3
"""
Compare Multi-Agent RL Models for Ghost Control in Pac-Man.

Trains all available models (qlearning, maddpg, dqn, ppo, qmix) separately
and evaluates them on the test map, then prints a comparison summary.

Usage:
    # Basic usage with defaults
    python compare_models.py

    # Custom configuration
    python compare_models.py --grid-sizes 20,30 --train-maps 0,1,2 --train-episodes 500 --test-episodes 100 --test-maps 3

    # Train on specific maps, test on different maps
    python compare_models.py --train-maps 0,2,4 --test-maps 1,3,5 --train-episodes 1000 --test-episodes 200

    # Single grid size, single train/test map
    python compare_models.py --grid-sizes 30 --train-maps 0 --test-maps 0 --train-episodes 1000 --test-episodes 100
"""

import argparse
import os
import sys
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Optional
from io import StringIO
from contextlib import contextmanager

# Suppress ALL logging and print statements from internal modules
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('train_ghosts').setLevel(logging.CRITICAL)
logging.getLogger('pacman_game').setLevel(logging.CRITICAL)

# Suppress pygame init messages
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# All available model types
MODEL_TYPES = ["qlearning", "maddpg", "dqn", "ppo", "qmix", "vdn"]

# Model display names
MODEL_NAMES = {
    "qlearning": "Q-Learning",
    "maddpg": "MADDPG",
    "dqn": "DQN",
    "ppo": "PPO",
    "qmix": "QMIX",
    "vdn": "VDN"
}

# Column width for uniform output
COL_WIDTH = 12


@contextmanager
def suppress_output():
    """Context manager to suppress all stdout/stderr output."""
    _stdout = sys.stdout
    _stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr


# Import game and RL components (after suppressing logging)
_old_stdout = sys.stdout
sys.stdout = StringIO()
try:
    from pacman_game import PacManGame
    from rl_utils import encode_state, ACTIONS
    from train_ghosts import RLTrainingEnvironment, load_config
finally:
    sys.stdout = _old_stdout


def parse_list_arg(arg_str: str, arg_name: str) -> List[int]:
    """Parse comma-separated list argument into list of integers."""
    if not arg_str:
        return []
    try:
        values = [int(x.strip()) for x in arg_str.split(',') if x.strip() != '']
        return values
    except ValueError:
        raise ValueError(f"--{arg_name} must contain comma-separated integers, e.g. 0,1,2")


def generate_training_sequence(train_episodes: int, train_maps: List[int], 
                                grid_sizes: List[int], seed: int = 42) -> List[tuple]:
    """
    Generate a deterministic sequence of (map_idx, grid_size) for fair comparison.
    All models will use the same sequence.
    """
    np.random.seed(seed)
    sequence = []
    for _ in range(train_episodes):
        map_idx = int(np.random.choice(train_maps))
        grid_size = int(np.random.choice(grid_sizes))
        sequence.append((map_idx, grid_size))
    return sequence


def train_and_evaluate_model(
    model_type: str,
    num_ghosts: int,
    grid_sizes: List[int],
    train_maps: List[int],
    train_episodes: int,
    test_maps: List[int],
    test_episodes: int,
    config: dict,
    save_dir: str = "compare_checkpoints",
    map_grid_sequence: Optional[List[tuple]] = None,
    use_roles: bool = False
) -> dict:
    """
    Train a single model and evaluate it on test maps.
    
    Args:
        model_type: Type of model to train
        num_ghosts: Number of ghosts
        grid_sizes: List of grid sizes to use
        train_maps: List of map indices for training
        train_episodes: Number of training episodes
        test_maps: List of map indices for testing
        test_episodes: Number of test episodes per map
        config: Configuration dictionary
        save_dir: Directory to save checkpoints
        map_grid_sequence: Pre-determined sequence of (map_idx, grid_size) for each episode
                          for fair comparison across models
    
    Returns a dictionary with evaluation results.
    """
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create environment with suppressed output
    with suppress_output():
        env = RLTrainingEnvironment(
            render=False,
            fps=0,
            num_ghosts=num_ghosts,
            train_map=train_maps[0] if train_maps else 0,
            test_map=test_maps[0] if test_maps else 0,
            config=config,
            model_type=model_type,
            log_moves=False,
            train_random_map=len(train_maps) > 1,
            train_maps=train_maps if len(train_maps) > 1 else None,
            train_grid_sizes=grid_sizes if len(grid_sizes) > 1 else None,
            use_roles=use_roles,
        )
        
        # Set single grid size if provided
        if len(grid_sizes) == 1:
            env.grid_size = grid_sizes[0]
            env.config.setdefault("state", {})["grid_size"] = grid_sizes[0]
            env.game = PacManGame(grid_size=grid_sizes[0])
            env.game.in_menu = False
            env.game.current_map_idx = train_maps[0] if train_maps else 0
            env.game._load_map(env.game.current_map_idx)
            env.game.reset()
        
        # Set game parameters
        env.game.max_pellets = 100
        env.game.time_limit = 20.0
    
    # Reset stats
    env.wins_ghosts = 0
    env.wins_pacman = 0
    env.episode_lengths = []
    env.episode_rewards_history = []
    env.eval_wins_ghosts = 0
    env.eval_wins_pacman = 0
    env.eval_episode_lengths = []
    env.eval_rewards_history = []
    
    # Training phase with uniform progress output
    model_name = MODEL_NAMES[model_type]
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    for ep in range(train_episodes):
        with suppress_output():
            env.episode = ep
            
            # Use predetermined map/grid sequence if provided for fair comparison
            if map_grid_sequence and ep < len(map_grid_sequence):
                map_idx, grid_size = map_grid_sequence[ep]
                env.current_train_map = map_idx
                env.game.current_map_idx = map_idx
                env.grid_size = grid_size
                env.config.setdefault("state", {})["grid_size"] = grid_size
            
            stats = env.train_episode()
        
        # Uniform progress output every 50 episodes
        if (ep + 1) % 50 == 0:
            recent_rewards = env.episode_rewards_history[-50:] if env.episode_rewards_history else [0]
            avg_reward = np.mean(recent_rewards)
            total_games = env.wins_ghosts + env.wins_pacman
            win_rate = (env.wins_ghosts / max(total_games, 1)) * 100
            print(f"  [{model_name:>10s}] Ep {ep+1:5d}/{train_episodes} | "
                  f"Reward: {avg_reward:7.1f} | WinRate: {win_rate:5.1f}%")
    
    # Final training summary
    final_rewards = env.episode_rewards_history[-50:] if len(env.episode_rewards_history) >= 50 else env.episode_rewards_history
    final_avg_reward = np.mean(final_rewards) if final_rewards else 0
    total_train_games = env.wins_ghosts + env.wins_pacman
    final_win_rate = (env.wins_ghosts / max(total_train_games, 1)) * 100
    print(f"  [{model_name:>10s}] Training Complete | "
          f"Final Reward: {final_avg_reward:7.1f} | Final WinRate: {final_win_rate:5.1f}%")
    
    # Evaluation phase - evaluate on all test maps
    all_ghost_wins = 0
    all_pacman_wins = 0
    all_episode_lengths = []
    all_rewards = []
    per_map_results = {}
    
    print(f"\n  [{model_name:>10s}] Evaluating on {len(test_maps)} test map(s)...")
    
    for test_map_idx in test_maps:
        with suppress_output():
            # Reset evaluation stats for this map
            env.eval_wins_ghosts = 0
            env.eval_wins_pacman = 0
            env.eval_episode_lengths = []
            env.eval_rewards_history = []
            
            for ep in range(test_episodes):
                env.episode = ep
                stats = env.evaluate_episode(map_idx=test_map_idx)
        
        total_games = env.eval_wins_ghosts + env.eval_wins_pacman
        ghost_win_rate = (env.eval_wins_ghosts / max(total_games, 1)) * 100
        
        per_map_results[test_map_idx] = {
            "ghost_wins": env.eval_wins_ghosts,
            "pacman_wins": env.eval_wins_pacman,
            "total_games": total_games,
            "ghost_win_rate": ghost_win_rate,
            "avg_reward": np.mean(env.eval_rewards_history) if env.eval_rewards_history else 0,
            "avg_length": np.mean(env.eval_episode_lengths) if env.eval_episode_lengths else 0,
        }
        
        all_ghost_wins += env.eval_wins_ghosts
        all_pacman_wins += env.eval_wins_pacman
        all_episode_lengths.extend(env.eval_episode_lengths)
        all_rewards.extend(env.eval_rewards_history)
        
        print(f"  [{model_name:>10s}] Map {test_map_idx}: WinRate {ghost_win_rate:5.1f}% | "
              f"AvgR {np.mean(env.eval_rewards_history) if env.eval_rewards_history else 0:7.1f}")
    
    # Aggregate results
    total_test_games = all_ghost_wins + all_pacman_wins
    overall_ghost_win_rate = (all_ghost_wins / max(total_test_games, 1)) * 100
    overall_pacman_win_rate = (all_pacman_wins / max(total_test_games, 1)) * 100
    
    # Capture role info if role mode active
    role_info = None
    if use_roles and hasattr(env, 'role_manager') and env.role_manager:
        role_names = {0: "Chaser", 1: "Blocker", 2: "Ambusher"}
        role_info = {
            "roles": [role_names.get(r.value, "Unknown") for r in env.role_manager.roles],
            "role_epsilon": env.role_manager.role_epsilon,
        }
    
    results = {
        "model_type": model_type,
        "model_name": model_name,
        "ghost_wins": all_ghost_wins,
        "pacman_wins": all_pacman_wins,
        "total_games": total_test_games,
        "ghost_win_rate": overall_ghost_win_rate,
        "pacman_win_rate": overall_pacman_win_rate,
        "avg_reward": np.mean(all_rewards) if all_rewards else 0,
        "avg_length": np.mean(all_episode_lengths) if all_episode_lengths else 0,
        "per_map_results": per_map_results,
        "use_roles": use_roles,
        "role_info": role_info,
    }
    
    return results


def print_comparison_table(results_list: List[dict], test_maps: List[int]):
    """Print a formatted comparison table of all model results."""
    print("\n" + "=" * 100)
    print("MULTI-AGENT RL MODEL COMPARISON RESULTS")
    print("=" * 100)
    print(f"Test Maps: {test_maps}")
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # Check if any model used roles
    any_use_roles = any(r.get("use_roles", False) for r in results_list)
    
    # Header
    print(f"\n{'Model':<12} {'Ghost Wins':>12} {'Pac-Man Wins':>14} {'Ghost Win %':>12} {'Avg Reward':>12} {'Avg Length':>12}")
    print("-" * 100)
    
    # Sort by ghost win rate (descending)
    sorted_results = sorted(results_list, key=lambda x: x["ghost_win_rate"], reverse=True)
    
    for i, results in enumerate(sorted_results):
        rank = f"{i+1}."
        print(f"{rank} {results['model_name']:<10} {results['ghost_wins']:>10} {results['pacman_wins']:>14} {results['ghost_win_rate']:>11.1f}% {results['avg_reward']:>12.1f} {results['avg_length']:>12.1f}")
    
    print("-" * 100)
    
    # Best model highlight
    best = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best['model_name']} ({best['ghost_win_rate']:.1f}% ghost win rate)")
    print(f"   - Ghost Wins: {best['ghost_wins']}/{best['total_games']}")
    print(f"   - Average Reward: {best['avg_reward']:.2f}")
    print(f"   - Average Episode Length: {best['avg_length']:.1f} steps")
    
    # Role information section (if role mode active)
    if any_use_roles:
        print("\n" + "=" * 100)
        print("GHOST ROLES (Role Mode Active)")
        print("=" * 100)
        for results in sorted_results:
            if results.get("use_roles") and results.get("role_info"):
                role_info = results["role_info"]
                roles_str = ", ".join([f"G{i}:{r}" for i, r in enumerate(role_info["roles"])])
                print(f"  {results['model_name']:<12}: {roles_str} (ε={role_info['role_epsilon']:.3f})")
    
    # Per-map breakdown
    if len(test_maps) > 1:
        print("\n" + "=" * 100)
        print("PER-MAP BREAKDOWN")
        print("=" * 100)
        
        for test_map_idx in test_maps:
            print(f"\nMap {test_map_idx}:")
            print(f"{'Model':<12} {'Ghost Win %':>12}")
            print("-" * 30)
            map_sorted = sorted(results_list, key=lambda x: x["per_map_results"].get(test_map_idx, {}).get("ghost_win_rate", 0), reverse=True)
            for results in map_sorted:
                map_res = results["per_map_results"].get(test_map_idx, {})
                print(f"{results['model_name']:<12} {map_res.get('ghost_win_rate', 0):>11.1f}%")
    
    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100 + "\n")


def save_results_csv(results_list: List[dict], test_maps: List[int], train_maps: List[int], 
                      train_episodes: int, test_episodes: int, grid_sizes: List[int]):
    """Save comparison results to CSV file."""
    import csv
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{date_str}.csv"
    
    fieldnames = [
        "model", "ghost_wins", "pacman_wins", "total_games",
        "ghost_win_rate", "pacman_win_rate", "avg_reward", "avg_length",
        "train_maps", "test_maps", "train_episodes", "test_episodes",
        "grid_sizes"
    ]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for results in results_list:
            row = {
                "model": results["model_type"],
                "ghost_wins": results["ghost_wins"],
                "pacman_wins": results["pacman_wins"],
                "total_games": results["total_games"],
                "ghost_win_rate": round(results["ghost_win_rate"], 4),
                "pacman_win_rate": round(results["pacman_win_rate"], 4),
                "avg_reward": round(results["avg_reward"], 2),
                "avg_length": round(results["avg_length"], 2),
                "train_maps": str(train_maps),
                "test_maps": str(test_maps),
                "train_episodes": train_episodes,
                "test_episodes": test_episodes,
                "grid_sizes": str(grid_sizes),
            }
            writer.writerow(row)
    
    print(f"📊 Results saved to: {filename}")


def generate_pdf_report(results_list: List[dict], test_maps: List[int], 
                        train_maps: List[int], train_episodes: int, 
                        test_episodes: int, grid_sizes: List[int]):
    """Generate a PDF report with comparison charts and table."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("⚠️  matplotlib not available. Install with: pip install matplotlib")
        print("   PDF generation skipped.")
        return
    
    from datetime import datetime
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{date_str}.pdf"
    
    # Sort results by ghost win rate
    sorted_results = sorted(results_list, key=lambda x: x["ghost_win_rate"], reverse=True)
    
    with PdfPages(filename) as pdf:
        # Page 1: Summary table and bar chart
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
        
        # Title
        fig.suptitle("Multi-Agent RL Model Comparison Results", fontsize=16, fontweight='bold', y=0.98)
        
        # Subtitle
        fig.text(0.5, 0.93, f"Train Maps: {train_maps} | Test Maps: {test_maps} | "
                f"Train Eps: {train_episodes} | Test Eps: {test_episodes} | "
                f"Grid: {grid_sizes}", ha='center', fontsize=10)
        
        # Table
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')
        
        table_data = []
        for i, r in enumerate(sorted_results):
            table_data.append([
                f"{i+1}. {r['model_name']}",
                f"{r['ghost_win_rate']:.1f}%",
                f"{r['avg_reward']:.1f}",
                f"{r['avg_length']:.1f}"
            ])
        
        table = ax1.table(
            cellText=table_data,
            colLabels=["Model", "Ghost Win %", "Avg Reward", "Avg Length"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight best model
        for i in range(len(table_data[0])):
            table[(1, i)].set_facecolor('#90EE90')
        
        # Bar chart - Ghost Win Rate
        ax2 = fig.add_subplot(gs[1])
        models = [r['model_name'] for r in sorted_results]
        win_rates = [r['ghost_win_rate'] for r in sorted_results]
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(models)))
        
        bars = ax2.barh(models, win_rates, color=colors, edgecolor='black')
        ax2.set_xlabel('Ghost Win Rate (%)', fontsize=12)
        ax2.set_title('Ghost Win Rate by Model', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, win_rates):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontsize=10)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Average Reward chart
        fig2, ax3 = plt.subplots(figsize=(12, 6))
        rewards = [r['avg_reward'] for r in sorted_results]
        colors2 = plt.cm.Greens(np.linspace(0.3, 0.8, len(models)))
        
        bars2 = ax3.bar(models, rewards, color=colors2, edgecolor='black')
        ax3.set_ylabel('Average Reward', fontsize=12)
        ax3.set_title('Average Reward by Model', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars2, rewards):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Page 3: Episode Length chart
        fig3, ax4 = plt.subplots(figsize=(12, 6))
        lengths = [r['avg_length'] for r in sorted_results]
        colors3 = plt.cm.Reds(np.linspace(0.3, 0.8, len(models)))
        
        bars3 = ax4.bar(models, lengths, color=colors3, edgecolor='black')
        ax4.set_ylabel('Average Episode Length (steps)', fontsize=12)
        ax4.set_title('Average Episode Length by Model', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars3, lengths):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
        
        # Metadata
        d = pdf.infodict()
        d['Title'] = 'Multi-Agent RL Model Comparison'
        d['Author'] = 'RL Pacman'
        d['Subject'] = 'Model Performance Comparison'
        d['Keywords'] = 'RL, Pacman, Multi-Agent, Comparison'
        d['CreationDate'] = datetime.now()
    
    print(f"📄 PDF report saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multi-agent RL models for ghost control in Pac-Man",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python compare_models.py

  # Custom configuration
  python compare_models.py --grid-sizes 20,30 --train-maps 0,1,2 --train-episodes 500 --test-episodes 100 --test-maps 3

  # Train on specific maps, test on different maps
  python compare_models.py --train-maps 0,2,4 --test-maps 1,3,5 --train-episodes 1000 --test-episodes 200

  # Single grid size, single train/test map
  python compare_models.py --grid-sizes 30 --train-maps 0 --test-maps 0 --train-episodes 1000 --test-episodes 100
        """
    )
    
    # Grid sizes
    parser.add_argument("--grid-sizes", type=str, default="30",
                        help="[GAME] Comma-separated grid sizes for training, e.g. 20,30,40 (default: 30)")
    
    # Map selection
    parser.add_argument("--train-maps", type=str, default="0",
                        help="[MAP] Comma-separated map indices for training, e.g. 0,1,2 (default: 0)")
    parser.add_argument("--test-maps", type=str, default=None,
                        help="[MAP] Comma-separated map indices for testing, e.g. 3,4,5 (default: same as train-maps)")
    
    # Episode counts
    parser.add_argument("--train-episodes", type=int, default=1000,
                        help="[TRAIN] Number of training episodes per model (default: 1000)")
    parser.add_argument("--test-episodes", type=int, default=100,
                        help="[EVAL] Number of evaluation episodes per model per test map (default: 100)")
    
    # Game settings
    parser.add_argument("--ghosts", type=int, default=4,
                        help="[GAME] Number of ghosts: 2 or 4 (default: 4)")
    
    # Config
    parser.add_argument("--config", type=str, default="config.json",
                        help="[TRAIN] Path to config.json for hyperparameters (default: config.json)")
    
    # Save directory
    parser.add_argument("--save-dir", type=str, default="compare_checkpoints",
                        help="[TRAIN] Directory to save model checkpoints (default: compare_checkpoints)")
    
    # Models to compare (optional filter)
    parser.add_argument("--models", type=str, default=None,
                        help="[MODEL] Comma-separated model types to compare: qlearning,maddpg,dqn,ppo,qmix (default: all)")
    
    # Skip saving results
    parser.add_argument("--no-save", action="store_true",
                        help="[OUTPUT] Do not save results to CSV")
    
    # Generate PDF report
    parser.add_argument("--pdf", action="store_true",
                        help="[OUTPUT] Generate PDF report with charts")
    
    # Use role-based multi-head policy
    parser.add_argument("--use-roles", action="store_true",
                        help="[ROLES] Enable role-based multi-head policy (Chaser/Blocker/Ambusher)")
    
    args = parser.parse_args()
    
    # Parse arguments
    try:
        grid_sizes = parse_list_arg(args.grid_sizes, "grid-sizes")
        if not grid_sizes:
            grid_sizes = [30]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    try:
        train_maps = parse_list_arg(args.train_maps, "train-maps")
        if not train_maps:
            train_maps = [0]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.test_maps:
        try:
            test_maps = parse_list_arg(args.test_maps, "test-maps")
            if not test_maps:
                test_maps = train_maps[:]
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        test_maps = train_maps[:]
    
    # Parse model filter
    if args.models:
        try:
            selected_models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
            invalid = [m for m in selected_models if m not in MODEL_TYPES]
            if invalid:
                print(f"Error: Invalid model types: {invalid}. Valid types: {MODEL_TYPES}")
                sys.exit(1)
            model_types = selected_models
        except:
            model_types = MODEL_TYPES[:]
    else:
        model_types = MODEL_TYPES[:]
    
    # Load config
    config = load_config(args.config)
    
    # Print configuration
    print("\n" + "=" * 100)
    print("MULTI-AGENT RL MODEL COMPARISON")
    print("=" * 100)
    print(f"\n📋 CONFIGURATION:")
    print(f"   Grid Sizes:      {grid_sizes}")
    print(f"   Train Maps:      {train_maps}")
    print(f"   Test Maps:       {test_maps}")
    print(f"   Train Episodes:  {args.train_episodes}")
    print(f"   Test Episodes:   {args.test_episodes}")
    print(f"   Number of Ghosts: {args.ghosts}")
    print(f"   Models to Compare: {model_types}")
    print(f"   Config File:     {args.config}")
    print(f"   Save Directory:  {args.save_dir}")
    print("\n" + "-" * 100)
    print("TRAINING & EVALUATION")
    print("-" * 100)
    
    # Generate deterministic training sequence for fair comparison across all models
    map_grid_sequence = generate_training_sequence(args.train_episodes, train_maps, grid_sizes)
    
    # Train and evaluate each model
    results_list = []
    
    for model_type in model_types:
        try:
            results = train_and_evaluate_model(
                model_type=model_type,
                map_grid_sequence=map_grid_sequence,
                num_ghosts=args.ghosts,
                grid_sizes=grid_sizes,
                train_maps=train_maps,
                train_episodes=args.train_episodes,
                test_maps=test_maps,
                test_episodes=args.test_episodes,
                config=config,
                save_dir=args.save_dir,
                use_roles=args.use_roles
            )
            results_list.append(results)
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Training: {MODEL_NAMES[model_type]}")
            print(f"{'='*60}")
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    if not results_list:
        print("\n❌ No models completed successfully. Exiting.")
        sys.exit(1)
    
    # Print comparison table
    print_comparison_table(results_list, test_maps)
    
    # Save to CSV
    if not args.no_save:
        save_results_csv(results_list, test_maps, train_maps, args.train_episodes, 
                        args.test_episodes, grid_sizes)
    
    # Generate PDF report
    if args.pdf:
        generate_pdf_report(results_list, test_maps, train_maps, args.train_episodes,
                           args.test_episodes, grid_sizes)


if __name__ == "__main__":
    main()
