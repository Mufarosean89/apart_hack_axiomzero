#!/usr/bin/env python3
"""
Curriculum Learning for Axiom Zero
Progressively trains on benchmarks from easy to hard.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    """A single level in the curriculum."""
    level: int
    name: str
    benchmark_ids: List[str]
    min_success_rate: float = 0.9  # 90% success to advance
    max_episodes: int = 100
    description: str = ""


class CurriculumLearning:
    """
    Implements curriculum learning for proof search.

    Starts with easy problems and progressively increases difficulty
    based on agent performance.
    """

    def __init__(self):
        """Initialize curriculum with 5 levels."""
        self.curriculum = [
            CurriculumLevel(
                level=1,
                name="Basic Arithmetic",
                benchmark_ids=["add_comm", "mul_one"],
                min_success_rate=0.9,
                max_episodes=50,
                description="Simple arithmetic properties (commutativity, identity)"
            ),
            CurriculumLevel(
                level=2,
                name="List Operations",
                benchmark_ids=["list_append_nil", "list_length_append"],
                min_success_rate=0.85,
                max_episodes=100,
                description="List properties (append, length)"
            ),
            CurriculumLevel(
                level=3,
                name="Loops & Recursion",
                benchmark_ids=["sum_formula", "factorial"],
                min_success_rate=0.8,
                max_episodes=150,
                description="Loop invariants and recursive functions"
            ),
            CurriculumLevel(
                level=4,
                name="Conditionals",
                benchmark_ids=["max_correct", "abs_value"],
                min_success_rate=0.75,
                max_episodes=200,
                description="Conditional reasoning and case analysis"
            ),
            CurriculumLevel(
                level=5,
                name="PyTorch/Tensors",
                benchmark_ids=["tensor_add", "matrix_vec_mul"],
                min_success_rate=0.7,
                max_episodes=300,
                description="Tensor shape properties and matrix operations"
            ),
        ]

        self.current_level_idx = 0
        self.level_stats = {}
        self.training_log = []

        # Initialize stats for each level
        for level in self.curriculum:
            self.level_stats[level.level] = {
                'episodes': 0,
                'successes': 0,
                'failures': 0,
                'success_rate': 0.0,
                'avg_steps': 0.0,
                'total_steps': 0,
                'start_time': None,
                'end_time': None,
            }

    @property
    def current_level(self) -> CurriculumLevel:
        """Get current curriculum level."""
        return self.curriculum[self.current_level_idx]

    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_level_idx >= len(self.curriculum)

    def get_current_benchmarks(self) -> List[str]:
        """Get benchmark IDs for current level."""
        if self.is_complete:
            return []
        return self.current_level.benchmark_ids

    def record_result(
        self,
        benchmark_id: str,
        success: bool,
        steps: int,
        time_seconds: float
    ):
        """
        Record training result for a benchmark.

        Args:
            benchmark_id: Benchmark identifier
            success: Whether proof was found
            steps: Number of tactics used
            time_seconds: Time taken
        """
        if self.is_complete:
            return

        stats = self.level_stats[self.current_level.level]
        stats['episodes'] += 1

        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

        stats['total_steps'] += steps
        stats['avg_steps'] = stats['total_steps'] / stats['episodes']

        if stats['episodes'] > 0:
            stats['success_rate'] = stats['successes'] / stats['episodes']

        # Log result
        self.training_log.append({
            'level': self.current_level.level,
            'benchmark': benchmark_id,
            'success': success,
            'steps': steps,
            'time': time_seconds,
            'episode': stats['episodes'],
        })

    def should_advance(self) -> bool:
        """
        Check if agent should advance to next level.

        Returns:
            True if success rate threshold met or max episodes reached
        """
        if self.is_complete:
            return False

        stats = self.level_stats[self.current_level.level]
        level = self.current_level

        # Check if max episodes reached
        if stats['episodes'] >= level.max_episodes:
            logger.info(
                f"Max episodes ({level.max_episodes}) reached for level {level.level}")
            return True

        # Check if minimum episodes completed (at least 20)
        if stats['episodes'] < 20:
            return False

        # Check success rate over last 20 episodes
        recent = [log for log in self.training_log[-20:]
                  if log['level'] == self.current_level.level]

        if len(recent) >= 10:
            recent_success_rate = sum(
                1 for r in recent if r['success']) / len(recent)

            if recent_success_rate >= level.min_success_rate:
                logger.info(
                    f"Success rate {recent_success_rate:.2f} >= {level.min_success_rate:.2f} "
                    f"for level {level.level}"
                )
                return True

        return False

    def advance(self):
        """Advance to next curriculum level."""
        if self.is_complete:
            logger.warning("Curriculum already complete")
            return

        # Mark current level as complete
        stats = self.level_stats[self.current_level.level]
        stats['end_time'] = time.time()

        # Move to next level
        self.current_level_idx += 1

        if not self.is_complete:
            stats = self.level_stats[self.current_level.level]
            stats['start_time'] = time.time()

            logger.info(
                f"Advanced to Level {self.current_level.level}: {self.current_level.name}"
            )

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {
            'current_level': self.current_level.level if not self.is_complete else 'Complete',
            'total_levels': len(self.curriculum),
            'is_complete': self.is_complete,
            'levels': {},
        }

        for level in self.curriculum:
            stats = self.level_stats[level.level]
            summary['levels'][level.level] = {
                'name': level.name,
                'episodes': stats['episodes'],
                'success_rate': stats['success_rate'],
                'avg_steps': stats['avg_steps'],
                'status': 'completed' if stats['end_time'] else
                'current' if level.level == self.current_level.level else 'pending'
            }

        return summary

    def print_progress(self):
        """Print current progress to console."""
        print("\n" + "="*70)
        print("CURRICULUM LEARNING PROGRESS")
        print("="*70)

        for level in self.curriculum:
            stats = self.level_stats[level.level]

            # Status indicator
            if stats['end_time']:
                status = "✓"
            elif level.level == self.current_level.level:
                status = "▶"
            else:
                status = "○"

            print(f"\n{status} Level {level.level}: {level.name}")
            print(f"   Episodes: {stats['episodes']}/{level.max_episodes}")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print(f"   Avg Steps: {stats['avg_steps']:.1f}")
            print(f"   Min Required: {level.min_success_rate:.0%}")

            if stats['episodes'] > 0:
                print(
                    f"   Progress: {stats['successes']}✓ / {stats['failures']}✗")

        print("\n" + "="*70)


class CurriculumTrainer:
    """
    High-level trainer that coordinates curriculum learning with RL agent.
    """

    def __init__(
        self,
        agent=None,
        environment=None,
        use_wandb: bool = False
    ):
        """
        Initialize curriculum trainer.

        Args:
            agent: RL agent (ProofAgent)
            environment: Proof environment (LeanEnvironment)
            use_wandb: Whether to use W&B tracking
        """
        self.curriculum = CurriculumLearning()
        self.agent = agent
        self.environment = environment
        self.use_wandb = use_wandb

        if use_wandb:
            from wandb_tracker import WandBTracker
            self.wandb = WandBTracker(
                project="axiom-zero",
                config={"curriculum": True}
            )
        else:
            self.wandb = None

    def train_on_benchmark(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train agent on a single benchmark.

        Args:
            benchmark: Benchmark dictionary

        Returns:
            Training result dictionary
        """
        benchmark_id = benchmark['id']
        difficulty = benchmark.get('difficulty', 'medium')

        print(f"\n  Training on: {benchmark_id} ({difficulty})")

        # Simulate training (replace with actual RL training)
        result = self._simulate_training(benchmark)

        # Record result
        self.curriculum.record_result(
            benchmark_id=benchmark_id,
            success=result['success'],
            steps=result['steps'],
            time_seconds=result['time']
        )

        # Log to W&B
        if self.wandb:
            self.wandb.log_benchmark_results(
                benchmark_id=benchmark_id,
                success=result['success'],
                proof_length=result['steps'],
                search_time=result['time'],
                mcts_simulations=result.get('mcts_simulations', 0)
            )

        return result

    def _simulate_training(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate training on a benchmark.

        Replace this with actual RL training loop.
        """
        import random

        difficulty = benchmark.get('difficulty', 'medium')
        level = self.curriculum.current_level.level

        # Simulate improving success rate with training
        base_success_rate = {
            'easy': 0.8,
            'medium': 0.6,
            'hard': 0.4
        }.get(difficulty, 0.5)

        # Improve with level and episodes
        episodes = self.curriculum.level_stats[level]['episodes']
        improvement = min(0.3, episodes * 0.01)
        success_rate = min(0.95, base_success_rate + improvement)

        # Random outcome based on success rate
        success = random.random() < success_rate

        # Simulate steps and time
        steps = random.randint(3, 10) if success else 0
        time_seconds = random.uniform(0.5, 5.0) if success else 0

        return {
            'success': success,
            'steps': steps,
            'time': time_seconds,
            'mcts_simulations': random.randint(100, 500) if success else 0
        }

    def run_curriculum(self):
        """Run complete curriculum training loop."""
        from benchmarks import get_benchmark_by_id

        print("\n" + "="*70)
        print("STARTING CURRICULUM LEARNING")
        print("="*70)

        start_time = time.time()

        while not self.curriculum.is_complete:
            level = self.curriculum.current_level
            print(f"\n{'='*70}")
            print(f"LEVEL {level.level}: {level.name}")
            print(f"{'='*70}")
            print(f"Description: {level.description}")
            print(f"Benchmarks: {', '.join(level.benchmark_ids)}")
            print(f"Target success rate: {level.min_success_rate:.0%}")

            # Train on each benchmark in current level
            for benchmark_id in level.benchmark_ids:
                benchmark = get_benchmark_by_id(benchmark_id)

                if not benchmark:
                    print(f"  ⊗ Benchmark {benchmark_id} not found, skipping")
                    continue

                # Run episode
                result = self.train_on_benchmark(benchmark)

                status = "✓" if result['success'] else "✗"
                print(f"    {status} Episode {self.curriculum.level_stats[level.level]['episodes']}: "
                      f"{'Success' if result['success'] else 'Failed'} "
                      f"({result['steps']} steps, {result['time']:.2f}s)")

            # Check if should advance
            self.curriculum.print_progress()

            if self.curriculum.should_advance():
                print(f"\n✓ Advancing to next level!")
                self.curriculum.advance()
            else:
                print(f"\n⊗ Need more training at current level")
                # In real training, would continue training here

        # Curriculum complete
        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("✓ CURRICULUM COMPLETE!")
        print("="*70)
        print(f"\nTotal time: {total_time:.1f}s")
        print(
            f"Total episodes: {sum(s['episodes'] for s in self.curriculum.level_stats.values())}")

        # Print final summary
        self.curriculum.print_progress()

        # Log to W&B
        if self.wandb:
            self.wandb.finish()

        return self.curriculum.get_progress_summary()


if __name__ == "__main__":
    # Test curriculum learning
    trainer = CurriculumTrainer(use_wandb=False)
    summary = trainer.run_curriculum()

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nCurriculum complete: {summary['is_complete']}")
    print(
        f"Levels completed: {summary['current_level']}/{summary['total_levels']}")
