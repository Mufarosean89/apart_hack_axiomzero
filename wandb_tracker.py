#!/usr/bin/env python3
"""
Weights & Biases integration for Axiom Zero training.
Tracks experiments, metrics, and model performance.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WandBTracker:
    """
    Weights & Biases experiment tracker.
    
    Logs training metrics, model performance, and proof statistics
    to wandb.ai for experiment tracking and visualization.
    """
    
    def __init__(
        self,
        project: str = "axiom-zero",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            config: Configuration dictionary to log
            use_wandb: Whether to actually use W&B (False for testing)
        """
        self.use_wandb = use_wandb
        self.wandb = None
        
        if use_wandb:
            try:
                import wandb
                
                # Check if logged in
                if not os.environ.get("WANDB_API_KEY"):
                    logger.warning("WANDB_API_KEY not set. W&B tracking disabled.")
                    self.use_wandb = False
                else:
                    self.wandb = wandb
                    
                    # Initialize run
                    self.wandb.init(
                        project=project,
                        entity=entity,
                        config=config or {},
                        name=config.get("run_name", None) if config else None
                    )
                    
                    logger.info(f"W&B initialized: {self.wandb.run.name}")
                    
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        if self.use_wandb and self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_training_step(
        self,
        episode: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float,
        success_rate: float = None
    ):
        """
        Log a training step.
        
        Args:
            episode: Episode number
            loss: Total loss
            policy_loss: Policy loss component
            value_loss: Value loss component
            learning_rate: Current learning rate
            success_rate: Optional proof success rate
        """
        metrics = {
            "training/episode": episode,
            "training/loss": loss,
            "training/policy_loss": policy_loss,
            "training/value_loss": value_loss,
            "training/learning_rate": learning_rate,
        }
        
        if success_rate is not None:
            metrics["training/success_rate"] = success_rate
        
        self.log_metrics(metrics, step=episode)
    
    def log_mcts_stats(
        self,
        episode: int,
        nodes_explored: int,
        search_depth: int,
        best_value: float,
        policy_entropy: float
    ):
        """
        Log MCTS search statistics.
        
        Args:
            episode: Episode number
            nodes_explored: Number of MCTS nodes explored
            search_depth: Maximum search depth
            best_value: Best value found
            policy_entropy: Policy entropy
        """
        metrics = {
            "mcts/nodes_explored": nodes_explored,
            "mcts/search_depth": search_depth,
            "mcts/best_value": best_value,
            "mcts/policy_entropy": policy_entropy,
        }
        
        self.log_metrics(metrics, step=episode)
    
    def log_proof_result(
        self,
        theorem_id: str,
        success: bool,
        steps: int,
        time_seconds: float,
        tactics_used: list,
        difficulty: str = "medium"
    ):
        """
        Log proof attempt result.
        
        Args:
            theorem_id: Theorem identifier
            success: Whether proof was found
            steps: Number of tactics used
            time_seconds: Time taken
            tactics_used: List of tactics used
            difficulty: Problem difficulty
        """
        metrics = {
            "proofs/success": 1.0 if success else 0.0,
            "proofs/steps": steps,
            "proofs/time_seconds": time_seconds,
            "proofs/num_tactics": len(tactics_used),
            f"proofs_by_difficulty/{difficulty}_success": 1.0 if success else 0.0,
        }
        
        self.log_metrics(metrics)
    
    def log_benchmark_results(
        self,
        benchmark_id: str,
        success: bool,
        proof_length: int,
        search_time: float,
        mcts_simulations: int
    ):
        """
        Log benchmark result.
        
        Args:
            benchmark_id: Benchmark identifier
            success: Whether proof was found
            proof_length: Length of proof
            search_time: Search time in seconds
            mcts_simulations: Number of MCTS simulations
        """
        metrics = {
            f"benchmarks/{benchmark_id}/success": 1.0 if success else 0.0,
            f"benchmarks/{benchmark_id}/proof_length": proof_length,
            f"benchmarks/{benchmark_id}/search_time": search_time,
            f"benchmarks/{benchmark_id}/mcts_simulations": mcts_simulations,
        }
        
        self.log_metrics(metrics)
    
    def log_model_info(
        self,
        model,
        sample_input,
        description: str = "ProofStateEncoder"
    ):
        """
        Log model architecture and watch gradients.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            description: Model description
        """
        if self.use_wandb and self.wandb:
            try:
                # Watch model gradients
                self.wandb.watch(model, log="all", log_freq=100)
                
                # Log model architecture
                self.wandb.config.update({
                    f"{description}/parameters": sum(p.numel() for p in model.parameters()),
                    f"{description}/trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                })
                
            except Exception as e:
                logger.warning(f"Failed to log model info: {e}")
    
    def log_checkpoint(
        self,
        model_state: dict,
        optimizer_state: dict,
        episode: int,
        best_reward: float
    ):
        """
        Save model checkpoint to W&B.
        
        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            episode: Current episode
            best_reward: Best reward achieved
        """
        if self.use_wandb and self.wandb:
            try:
                checkpoint = {
                    'episode': episode,
                    'best_reward': best_reward,
                    'model_state': model_state,
                    'optimizer_state': optimizer_state,
                }
                
                # Save as artifact
                artifact = self.wandb.Artifact(
                    name=f"model-episode-{episode}",
                    type="model",
                    description=f"Model checkpoint at episode {episode}"
                )
                
                # Save to file and add to artifact
                import torch
                torch.save(checkpoint, "checkpoint.pt")
                artifact.add_file("checkpoint.pt")
                
                self.wandb.log_artifact(artifact)
                
                logger.info(f"Checkpoint saved: episode {episode}")
                
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.use_wandb and self.wandb:
            self.wandb.finish()
            logger.info("W&B run finished")


class TrainingMonitor:
    """
    High-level training monitor that coordinates all logging.
    """
    
    def __init__(self, config: dict, use_wandb: bool = True):
        """
        Initialize training monitor.
        
        Args:
            config: Training configuration
            use_wandb: Whether to use W&B
        """
        self.tracker = WandBTracker(
            project="axiom-zero",
            config=config,
            use_wandb=use_wandb
        )
        
        self.episode_rewards = []
        self.success_count = 0
        self.total_proofs = 0
        
    def on_episode_end(
        self,
        episode: int,
        reward: float,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float,
        mcts_stats: dict = None
    ):
        """
        Called at end of each training episode.
        
        Args:
            episode: Episode number
            reward: Episode reward
            loss: Total loss
            policy_loss: Policy loss
            value_loss: Value loss
            learning_rate: Current learning rate
            mcts_stats: Optional MCTS statistics
        """
        self.episode_rewards.append(reward)
        
        # Calculate success rate (last 100 episodes)
        recent_rewards = self.episode_rewards[-100:]
        success_rate = sum(1 for r in recent_rewards if r > 0.5) / len(recent_rewards)
        
        # Log training metrics
        self.tracker.log_training_step(
            episode=episode,
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            learning_rate=learning_rate,
            success_rate=success_rate
        )
        
        # Log MCTS stats if available
        if mcts_stats:
            self.tracker.log_mcts_stats(episode, **mcts_stats)
    
    def on_proof_complete(
        self,
        theorem_id: str,
        steps: int,
        time_seconds: float,
        tactics_used: list,
        difficulty: str = "medium"
    ):
        """
        Called when a proof is successfully found.
        
        Args:
            theorem_id: Theorem identifier
            steps: Number of steps
            time_seconds: Time taken
            tactics_used: Tactics used
            difficulty: Problem difficulty
        """
        self.success_count += 1
        self.total_proofs += 1
        
        self.tracker.log_proof_result(
            theorem_id=theorem_id,
            success=True,
            steps=steps,
            time_seconds=time_seconds,
            tactics_used=tactics_used,
            difficulty=difficulty
        )
    
    def on_proof_failed(self, theorem_id: str, difficulty: str = "medium"):
        """
        Called when proof search fails.
        
        Args:
            theorem_id: Theorem identifier
            difficulty: Problem difficulty
        """
        self.total_proofs += 1
        
        self.tracker.log_proof_result(
            theorem_id=theorem_id,
            success=False,
            steps=0,
            time_seconds=0,
            tactics_used=[],
            difficulty=difficulty
        )
    
    def on_checkpoint(
        self,
        model_state: dict,
        optimizer_state: dict,
        episode: int,
        best_reward: float
    ):
        """
        Called when saving checkpoint.
        
        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            episode: Episode number
            best_reward: Best reward
        """
        self.tracker.log_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            episode=episode,
            best_reward=best_reward
        )
    
    def finish(self):
        """Finish training run."""
        self.tracker.finish()


if __name__ == "__main__":
    # Example usage
    print("Weights & Biases Integration for Axiom Zero")
    print("="*60)
    print()
    print("Setup:")
    print("  1. pip install wandb")
    print("  2. wandb login")
    print("  3. Set WANDB_API_KEY environment variable")
    print()
    print("Usage:")
    print("  tracker = WandBTracker(project='axiom-zero', config={...})")
    print("  tracker.log_training_step(episode=1, loss=0.5, ...)")
    print("  tracker.log_proof_result('add_comm', success=True, ...)")
    print("  tracker.finish()")
    print()
    print("Dashboard: https://wandb.ai/your-username/axiom-zero")
