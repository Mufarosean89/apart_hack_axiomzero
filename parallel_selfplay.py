#!/usr/bin/env python3
"""
Parallel Self-Play Training
Uses multiprocessing to generate training data faster.
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import random

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single self-play game."""
    theorem_id: str
    success: bool
    trajectory: List[Dict[str, Any]]  # (state, action, reward) sequence
    total_reward: float
    steps: int
    search_time: float


class ParallelSelfPlay:
    """
    Parallel self-play generation using process pool.

    Generates training data much faster than sequential self-play
    by running multiple games simultaneously.
    """

    def __init__(
        self,
        num_workers: int = None,
        mcts_simulations: int = 200,
        max_steps: int = 50
    ):
        """
        Initialize parallel self-play.

        Args:
            num_workers: Number of parallel workers (default: CPU count)
            mcts_simulations: MCTS simulations per move
            max_steps: Maximum steps per game
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.mcts_simulations = mcts_simulations
        self.max_steps = max_steps

        logger.info(
            f"Initialized parallel self-play with {self.num_workers} workers")

    def generate_games(
        self,
        theorems: List[Dict[str, Any]],
        games_per_theorem: int = 10
    ) -> List[GameResult]:
        """
        Generate self-play games in parallel.

        Args:
            theorems: List of theorem dictionaries
            games_per_theorem: Number of games per theorem

        Returns:
            List of game results
        """
        # Create tasks
        tasks = []
        for theorem in theorems:
            for game_idx in range(games_per_theorem):
                tasks.append({
                    'theorem': theorem,
                    'game_idx': game_idx,
                    'mcts_simulations': self.mcts_simulations,
                    'max_steps': self.max_steps,
                })

        logger.info(
            f"Generating {len(tasks)} games with {self.num_workers} workers")

        # Run in parallel
        start_time = time.time()

        with Pool(self.num_workers) as pool:
            results = pool.map(self._play_single_game, tasks)

        elapsed = time.time() - start_time

        # Filter valid results
        valid_results = [r for r in results if r is not None]

        logger.info(
            f"Generated {len(valid_results)} games in {elapsed:.1f}s "
            f"({len(valid_results)/elapsed:.1f} games/s)"
        )

        return valid_results

    @staticmethod
    def _play_single_game(task: Dict[str, Any]) -> Optional[GameResult]:
        """
        Play a single self-play game.

        This runs in a separate process.

        Args:
            task: Game task dictionary

        Returns:
            GameResult or None if failed
        """
        try:
            theorem = task['theorem']
            game_idx = task['game_idx']
            mcts_simulations = task['mcts_simulations']
            max_steps = task['max_steps']

            # Simulate self-play (replace with actual MCTS + agent)
            trajectory = []
            total_reward = 0.0
            success = False

            # Simulate game steps
            num_steps = random.randint(3, 15)
            num_steps = min(num_steps, max_steps)

            for step in range(num_steps):
                # Simulate state, action, reward
                state = {
                    'step': step,
                    'goals_remaining': max(0, num_steps - step),
                }

                # Random action (would be MCTS-selected in real system)
                action = random.choice(['simp', 'ring', 'induction', 'apply'])

                # Reward shaping
                if step == num_steps - 1:
                    reward = 1.0  # Proof complete!
                    success = True
                else:
                    reward = -0.05  # Small penalty per step

                total_reward += reward

                trajectory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                })

            search_time = random.uniform(0.5, 3.0)

            return GameResult(
                theorem_id=theorem['id'],
                success=success,
                trajectory=trajectory,
                total_reward=total_reward,
                steps=num_steps,
                search_time=search_time,
            )

        except Exception as e:
            logger.error(f"Error in self-play game: {e}")
            return None

    def generate_batch(
        self,
        theorems: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[GameResult]:
        """
        Generate games in batches to avoid memory issues.

        Args:
            theorems: List of theorems
            batch_size: Games per batch

        Returns:
            List of all game results
        """
        all_results = []
        total_games = len(theorems) * 10  # 10 games per theorem
        batches = (total_games + batch_size - 1) // batch_size

        logger.info(f"Generating {total_games} games in {batches} batches")

        games_generated = 0

        for batch_idx in range(batches):
            # Get subset of theorems for this batch
            batch_theorems = theorems[:batch_size // 10]

            # Generate batch
            batch_results = self.generate_games(
                batch_theorems, games_per_theorem=10)
            all_results.extend(batch_results)

            games_generated += len(batch_results)

            logger.info(
                f"Batch {batch_idx + 1}/{batches}: "
                f"{games_generated}/{total_games} games"
            )

        return all_results


class DistributedReplayBuffer:
    """
    Replay buffer that works with parallel self-play.
    Thread-safe storage for training data.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of games to store
        """
        self.capacity = capacity
        self.buffer = Manager().list()
        self.lock = Manager().Lock()

    def add_game(self, game: GameResult):
        """
        Add game result to buffer.

        Args:
            game: GameResult to add
        """
        with self.lock:
            self.buffer.append({
                'theorem_id': game.theorem_id,
                'success': game.success,
                'trajectory': game.trajectory,
                'total_reward': game.total_reward,
                'steps': game.steps,
            })

            # Remove oldest if over capacity
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)

    def add_batch(self, games: List[GameResult]):
        """Add multiple games to buffer."""
        for game in games:
            self.add_game(game)

    def sample(self, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Number of samples

        Returns:
            List of training samples
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)

            return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            if not self.buffer:
                return {'size': 0}

            successes = sum(1 for g in self.buffer if g['success'])
            avg_reward = sum(g['total_reward']
                             for g in self.buffer) / len(self.buffer)
            avg_steps = sum(g['steps'] for g in self.buffer) / len(self.buffer)

            return {
                'size': len(self.buffer),
                'capacity': self.capacity,
                'success_rate': successes / len(self.buffer),
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
            }


class ParallelTrainer:
    """
    Complete parallel training system.

    Coordinates:
    - Parallel self-play generation
    - Distributed replay buffer
    - Model training
    - Checkpointing
    """

    def __init__(
        self,
        num_workers: int = None,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        checkpoint_interval: int = 100
    ):
        """
        Initialize parallel trainer.

        Args:
            num_workers: Number of self-play workers
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            checkpoint_interval: Episodes between checkpoints
        """
        self.self_play = ParallelSelfPlay(num_workers=num_workers)
        self.buffer = DistributedReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        self.training_log = []

    def train_iteration(
        self,
        theorems: List[Dict[str, Any]],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Run one training iteration.

        Args:
            theorems: Theorems to train on
            iteration: Iteration number

        Returns:
            Training statistics
        """
        logger.info(f"=== Training Iteration {iteration} ===")

        # Phase 1: Generate self-play games
        logger.info("Phase 1: Generating self-play games...")
        games = self.self_play.generate_games(theorems, games_per_theorem=10)

        # Add to replay buffer
        self.buffer.add_batch(games)

        logger.info(f"Generated {len(games)} games")
        logger.info(f"Buffer size: {len(self.buffer)}")

        # Phase 2: Train on replay buffer
        logger.info("Phase 2: Training on replay buffer...")

        if len(self.buffer) >= self.batch_size:
            batch = self.buffer.sample(self.batch_size)

            # Simulate training (replace with actual neural network training)
            loss = self._simulate_training(batch)

            logger.info(f"Training loss: {loss:.4f}")
        else:
            loss = None
            logger.info("Buffer too small for training")

        # Phase 3: Evaluate
        logger.info("Phase 3: Evaluating...")
        success_rate = sum(1 for g in games if g.success) / \
            len(games) if games else 0

        stats = {
            'iteration': iteration,
            'games_generated': len(games),
            'buffer_size': len(self.buffer),
            'success_rate': success_rate,
            'loss': loss,
            'buffer_stats': self.buffer.get_stats(),
        }

        self.training_log.append(stats)

        return stats

    def _simulate_training(self, batch: List[Dict[str, Any]]) -> float:
        """Simulate training loss computation."""
        # In real system, this would be actual neural network training
        return random.uniform(0.1, 1.0)

    def run_training(
        self,
        theorems: List[Dict[str, Any]],
        num_iterations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run complete parallel training.

        Args:
            theorems: Theorems to train on
            num_iterations: Number of training iterations

        Returns:
            Training log
        """
        logger.info(f"Starting parallel training: {num_iterations} iterations")

        all_stats = []

        for iteration in range(1, num_iterations + 1):
            stats = self.train_iteration(theorems, iteration)
            all_stats.append(stats)

            # Print progress
            print(f"\nIteration {iteration}/{num_iterations}")
            print(f"  Games: {stats['games_generated']}")
            print(f"  Buffer: {stats['buffer_size']}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            if stats['loss']:
                print(f"  Loss: {stats['loss']:.4f}")

            # Checkpoint
            if iteration % self.checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at iteration {iteration}")
                self._save_checkpoint(iteration, stats)

        logger.info("Training complete!")

        return all_stats

    def _save_checkpoint(self, iteration: int, stats: Dict[str, Any]):
        """Save training checkpoint."""
        import json
        from pathlib import Path

        checkpoint = {
            'iteration': iteration,
            'stats': stats,
            'buffer_stats': self.buffer.get_stats(),
        }

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_iter_{iteration}.json"

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {checkpoint_file}")


if __name__ == "__main__":
    # Example usage
    from benchmarks import BENCHMARKS

    print("="*70)
    print("PARALLEL SELF-PLAY TRAINING")
    print("="*70)
    print()

    # Initialize trainer
    trainer = ParallelTrainer(
        num_workers=4,
        buffer_capacity=10000,
        batch_size=64,
        checkpoint_interval=5
    )

    # Run training
    stats = trainer.run_training(BENCHMARKS, num_iterations=10)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTotal iterations: {len(stats)}")
    print(f"Final buffer size: {stats[-1]['buffer_size']}")
    print(f"Final success rate: {stats[-1]['success_rate']:.1%}")
