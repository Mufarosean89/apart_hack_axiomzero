"""
Monte Carlo Tree Search for proof finding.
Implements AlphaZero-style MCTS over the proof tree.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch

from .networks import ProofAgent, ProofStateGraphBuilder
from proof_engine import ProofState, TacticSpace


@dataclass
class MCTSNode:
    """
    Node in the MCTS search tree.
    Represents a proof state.
    """
    proof_state: ProofState
    parent: Optional['MCTSNode'] = None
    tactic_applied: Optional[str] = None  # Tactic that led to this state
    reward: Optional[float] = None  # Reward from environment

    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0
    prior_probability: float = 0.0

    # Children
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)

    # Cached values
    _value: Optional[float] = None

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (proof complete or failed)."""
        return self.proof_state.is_proved() or self.proof_state.has_error

    def add_child(self, tactic: str, child_node: 'MCTSNode'):
        """Add a child node."""
        self.children[tactic] = child_node

    def get_child(self, tactic: str) -> Optional['MCTSNode']:
        """Get child node by tactic."""
        return self.children.get(tactic)


class MCTS:
    """
    Monte Carlo Tree Search for proof search.
    Implements the four phases: Selection, Expansion, Simulation, Backpropagation.
    """

    def __init__(self, agent: ProofAgent, tactic_space: TacticSpace,
                 c_puct: float = 1.0, num_simulations: int = 100,
                 max_depth: int = 50):
        """
        Initialize MCTS.

        Args:
            agent: ProofAgent (policy + value networks)
            tactic_space: Available tactics
            c_puct: Exploration constant for PUCT formula
            num_simulations: Number of MCTS simulations per search
            max_depth: Maximum search depth
        """
        self.agent = agent
        self.tactic_space = tactic_space
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth

        # Statistics
        self.total_nodes_explored = 0
        self.total_simulations = 0

    def search(self, root_state: ProofState) -> Tuple[str, MCTSNode]:
        """
        Run MCTS search from root state.

        Args:
            root_state: Current proof state

        Returns:
            Tuple of (best_tactic, root_node)
        """
        # Create root node
        root = MCTSNode(proof_state=root_state)

        # Get initial policy prior from network
        self._initialize_root(root)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, depth=0)
            self.total_simulations += 1

        # Select best action
        best_tactic = self._select_best_action(root)

        return best_tactic, root

    def _initialize_root(self, root: MCTSNode):
        """Initialize root node with neural network predictions."""
        # Convert proof state to graph
        graph_data = ProofStateGraphBuilder.build_graph(root.proof_state)

        # Get policy prediction
        prediction = self.agent.predict(graph_data)

        # Set prior probabilities for all tactics
        tactic_probs = prediction['tactic_probs'].squeeze()
        for i, tactic_name in enumerate(self.tactic_space.tactics.keys()):
            if i < len(tactic_probs):
                root.children[tactic_name] = MCTSNode(
                    proof_state=None,  # Will be set on expansion
                    parent=root,
                    tactic_applied=tactic_name,
                    prior_probability=tactic_probs[i].item()
                )

    def _simulate(self, node: MCTSNode, depth: int) -> float:
        """
        Run one MCTS simulation.

        Args:
            node: Current node
            depth: Current depth in search tree

        Returns:
            Value of the simulation
        """
        self.total_nodes_explored += 1

        # Check if terminal
        if node.is_terminal:
            value = 1.0 if node.proof_state.is_proved() else -1.0
            return value

        # Check max depth
        if depth >= self.max_depth:
            # Use value network estimate
            return self._evaluate_state(node.proof_state)

        # Selection: traverse to leaf using PUCT
        if node.is_expanded:
            best_child = self._select_child(node)
            value = self._simulate(best_child, depth + 1)
        else:
            # Expansion: expand node and evaluate
            value = self._expand_and_evaluate(node)

        # Backpropagation
        self._backpropagate(node, value)

        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child using PUCT (Predictor + UCT) formula.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Args:
            node: Parent node

        Returns:
            Selected child node
        """
        best_score = -float('inf')
        best_child = None

        for tactic, child in node.children.items():
            if child.visits == 0:
                # Unvisited node - high priority
                score = self.c_puct * child.prior_probability * \
                    math.sqrt(node.visits)
            else:
                # PUCT formula
                q_value = child.value
                exploration = (self.c_puct * child.prior_probability *
                               math.sqrt(node.visits) / (1 + child.visits))
                score = q_value + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node by applying tactics and evaluate children.

        Args:
            node: Node to expand

        Returns:
            Value estimate
        """
        # Get valid actions for this state
        valid_tactics = self.tactic_space.suggest_tactics(
            node.proof_state.current_goal.target if node.proof_state.current_goal else "True"
        )

        # Evaluate each child
        for tactic_template in valid_tactics:
            tactic_name = tactic_template.name

            # Skip if already exists
            if tactic_name in node.children:
                continue

            # Create child node with tactic applied
            # Note: In real implementation, would execute tactic via Lean environment
            child_state = self._apply_tactic_simulation(
                node.proof_state, tactic_name)

            child = MCTSNode(
                proof_state=child_state,
                parent=node,
                tactic_applied=tactic_name,
                prior_probability=1.0 / max(1, len(valid_tactics))
            )

            node.add_child(tactic_name, child)

        # Evaluate using value network
        value = self._evaluate_state(node.proof_state)

        return value

    def _apply_tactic_simulation(self, state: ProofState, tactic: str) -> ProofState:
        """
        Simulate applying a tactic (without Lean execution).
        For real usage, this would call the Lean environment.

        Args:
            state: Current proof state
            tactic: Tactic to apply

        Returns:
            New proof state
        """
        new_state = copy.deepcopy(state)

        # Simulate tactic application
        # This is a simplified simulation - real version uses Lean environment
        if tactic == 'intro':
            # Intro removes a ∀ or → from goal
            if '∀' in new_state.current_goal.target or '→' in new_state.current_goal.target:
                new_state.apply_tactic(tactic, success=True)
            else:
                new_state.apply_tactic(
                    tactic, success=False, error="Cannot intro")

        elif tactic in ['simp', 'ring', 'norm_num']:
            # Simplification tactics
            new_state.apply_tactic(tactic, success=True)

        elif tactic in ['linarith', 'omega']:
            # Arithmetic tactics
            if any(op in new_state.current_goal.target for op in ['<', '>', '+', '-']):
                new_state.apply_tactic(tactic, success=True)
            else:
                new_state.apply_tactic(
                    tactic, success=False, error="Not arithmetic")

        else:
            # Default: partial success
            new_state.apply_tactic(tactic, success=True)

        return new_state

    def _evaluate_state(self, state: ProofState) -> float:
        """
        Evaluate state using value network.

        Args:
            state: Proof state

        Returns:
            Value estimate
        """
        # Convert to graph
        graph_data = ProofStateGraphBuilder.build_graph(state)

        # Get value prediction
        with torch.no_grad():
            latent = self.agent.encoder(graph_data)
            value = self.agent.value(latent)

        return value.item()

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.

        Args:
            node: Leaf node
            value: Value to backpropagate
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def _select_best_action(self, root: MCTSNode) -> str:
        """
        Select best action based on visit counts.

        Args:
            root: Root node

        Returns:
            Best tactic
        """
        if not root.children:
            return None

        # Select most visited child
        best_tactic = None
        best_visits = -1

        for tactic, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_tactic = tactic

        return best_tactic

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get MCTS search statistics."""
        return {
            'total_simulations': self.total_simulations,
            'total_nodes_explored': self.total_nodes_explored,
            'nodes_per_simulation': self.total_nodes_explored / max(1, self.total_simulations)
        }
