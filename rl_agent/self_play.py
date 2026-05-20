"""
Self-play training loop and replay buffer for RL proof agent.
Generates training data through autonomous proof attempts.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from .networks import ProofAgent, ProofStateGraphBuilder
from .mcts import MCTS
from proof_engine import ProofState, LeanEnvironment, TacticSpace

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    Single training example from self-play.
    (state, policy_target, value_target)
    """
    state_features: torch.Tensor  # Encoded state
    policy_target: torch.Tensor   # MCTS-improved policy
    value_target: float           # Game outcome
    tactic_sequence: List[str]    # Tactics applied
    theorem_name: str             # Theorem being proved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_target': self.policy_target.tolist(),
            'value_target': self.value_target,
            'tactic_sequence': self.tactic_sequence,
            'theorem_name': self.theorem_name
        }


class ReplayBuffer(Dataset):
    """
    Replay buffer for storing self-play games.
    Implements experience replay for stable training.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store
        """
        self.capacity = capacity
        self.examples: List[TrainingExample] = []
        self.position = 0

    def add(self, example: TrainingExample):
        """
        Add training example to buffer.

        Args:
            example: Training example
        """
        if len(self.examples) < self.capacity:
            self.examples.append(example)
        else:
            self.examples[self.position] = example

        self.position = (self.position + 1) % self.capacity

    def add_batch(self, examples: List[TrainingExample]):
        """Add multiple examples."""
        for example in examples:
            self.add(example)

    def __len__(self) -> int:
        """Get number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Get training example."""
        example = self.examples[idx]
        return example.state_features, example.policy_target, example.value_target

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Tuple of (states, policy_targets, value_targets)
        """
        indices = random.sample(range(len(self.examples)), min(
            batch_size, len(self.examples)))

        states = torch.stack(
            [self.examples[i].state_features for i in indices])
        policy_targets = torch.stack(
            [self.examples[i].policy_target for i in indices])
        value_targets = torch.tensor(
            [self.examples[i].value_target for i in indices])

        return states, policy_targets, value_targets

    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            'examples': [ex.to_dict() for ex in self.examples],
            'capacity': self.capacity,
            'position': self.position
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.capacity = data['capacity']
        self.position = data['position']
        self.examples = []

        # Note: Would need to reconstruct tensors from saved data
        logger.info("Loaded %d examples from %s", len(data['examples']), path)


class SelfPlayTrainer:
    """
    Self-play training loop for proof agent.
    Generates training data by playing proof games against itself.
    """

    def __init__(self, agent: ProofAgent, tactic_space: TacticSpace,
                 buffer_capacity: int = 100000, batch_size: int = 256,
                 learning_rate: float = 1e-4, c_puct: float = 1.0,
                 num_simulations: int = 100,
                 mcts_temperature: float = 1.0):
        """
        Initialize trainer.

        Args:
            agent: ProofAgent to train
            tactic_space: Available tactics
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            learning_rate: Learning rate
            c_puct: MCTS exploration constant
            num_simulations: MCTS simulations per move
        """
        self.agent = agent
        self.tactic_space = tactic_space
        self.mcts_temperature = mcts_temperature

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

        # Loss weights
        self.policy_loss_weight = 1.0
        self.value_loss_weight = 1.0

        # MCTS
        self.mcts = MCTS(
            agent=agent,
            tactic_space=tactic_space,
            c_puct=c_puct,
            num_simulations=num_simulations
        )

        # Lean environment (for verification)
        self.lean_env = LeanEnvironment()

        # Statistics
        self.training_steps = 0
        self.games_played = 0
        self.proofs_found = 0

    def generate_self_play_game(self, theorem_statement: str,
                                imports: List[str] = None) -> List[TrainingExample]:
        """
        Generate a self-play game for a theorem.

        Args:
            theorem_statement: Theorem to prove
            imports: Required imports

        Returns:
            List of training examples from the game
        """
        # Initialize environment
        state = self.lean_env.initialize(theorem_statement, imports)

        examples = []
        tactic_sequence = []
        max_steps = 50

        logger.info("Starting self-play game for: %s", state.theorem_name)

        for step in range(max_steps):
            # Get current state features
            graph_data = ProofStateGraphBuilder.build_graph(state)

            # Run MCTS to get improved policy
            best_tactic, mcts_root = self.mcts.search(state)

            if best_tactic is None:
                logger.debug("Step %d: No valid tactics", step)
                break

            # Get MCTS policy (visit counts as probabilities)
            mcts_policy = self._extract_mcts_policy(mcts_root)

            # Execute tactic in environment
            new_state, reward, done, info = self.lean_env.step(best_tactic)

            tactic_sequence.append(best_tactic)

            logger.debug("Step %d: %s (reward: %.2f)", step, best_tactic, reward)

            # Create training example
            example = TrainingExample(
                state_features=graph_data.x,
                policy_target=mcts_policy,
                value_target=reward if done else 0.0,
                tactic_sequence=tactic_sequence.copy(),
                theorem_name=state.theorem_name
            )

            examples.append(example)

            # Update state
            state = new_state

            if done:
                if state.is_proved():
                    logger.info("Proof found in %d steps", len(tactic_sequence))
                    self.proofs_found += 1

                    # Backfill value targets with final outcome
                    for ex in examples:
                        ex.value_target = 1.0
                else:
                    logger.info("Proof failed: %s", info.get('error', 'unknown'))

                    # Backfill with failure
                    for ex in examples:
                        ex.value_target = -1.0

                break

        self.games_played += 1
        return examples

    def _extract_mcts_policy(self, mcts_root) -> torch.Tensor:
        """
        Extract policy from MCTS visit counts.

        Args:
            mcts_root: Root MCTS node

        Returns:
            Policy distribution over tactics
        """
        tactic_names = list(self.tactic_space.tactics.keys())
        policy = torch.zeros(len(tactic_names))

        total_visits = sum(
            child.visits for child in mcts_root.children.values())

        if total_visits == 0:
            return policy

        for i, tactic_name in enumerate(tactic_names):
            if tactic_name in mcts_root.children:
                policy[i] = mcts_root.children[tactic_name].visits / total_visits

        if self.mcts_temperature != 1.0:
            policy = policy ** (1.0 / self.mcts_temperature)
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum

        return policy

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        # Sample batch
        states, policy_targets, value_targets = self.buffer.sample(
            self.batch_size)

        # Forward pass
        self.agent.train()

        # Note: In real implementation, would need to reconstruct graph data
        # This is simplified for demonstration
        latent = self.agent.encoder.node_encoder(states)
        tactic_probs, param_logits = self.agent.policy(latent)
        value_pred = self.agent.value(latent)

        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(
            tactic_probs,
            policy_targets.argmax(dim=-1)
        )

        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(), value_targets)

        # Total loss
        loss = (self.policy_loss_weight * policy_loss +
                self.value_loss_weight * value_loss)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train(self, theorems: List[Dict[str, Any]],
              num_games: int = 100, num_training_steps: int = 1000,
              save_path: str = "models/proof_agent.pth"):
        """
        Complete training loop.

        Args:
            theorems: List of theorem definitions
            num_games: Number of self-play games
            num_training_steps: Number of training steps
            save_path: Path to save model
        """
        logger.info("Starting self-play training: %d games", num_games)

        for game_idx in range(num_games):
            logger.info("Game %d/%d", game_idx + 1, num_games)

            # Select theorem
            theorem = random.choice(theorems)

            # Generate self-play game
            examples = self.generate_self_play_game(
                theorem['statement'],
                theorem.get('imports', [])
            )

            # Add to replay buffer
            self.buffer.add_batch(examples)

            # Train
            if len(self.buffer) >= self.batch_size:
                metrics = self.train_step()
                logger.info("Training: loss=%.4f", metrics['loss'])

            # Save periodically
            if (game_idx + 1) % 10 == 0:
                self.save(save_path)
                logger.info("Model saved to %s", save_path)

        success_rate = self.proofs_found / max(1, self.games_played)
        logger.info(
            "Training complete: %d games, %d proofs (%.2f%%), %d steps, %d buffered",
            self.games_played, self.proofs_found, success_rate * 100,
            self.training_steps, len(self.buffer)
        )

    def save(self, path: str):
        """Save agent and buffer."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'games_played': self.games_played,
            'proofs_found': self.proofs_found
        }, path)

        # Save buffer separately
        buffer_path = path.replace('.pth', '_buffer.json')
        self.buffer.save(buffer_path)

    def load(self, path: str):
        """Load agent and buffer."""
        checkpoint = torch.load(path, weights_only=False)

        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.games_played = checkpoint['games_played']
        self.proofs_found = checkpoint['proofs_found']

        logger.info(
            "Loaded agent from %s (steps=%d, games=%d, proofs=%d)",
            path, self.training_steps, self.games_played, self.proofs_found
        )
