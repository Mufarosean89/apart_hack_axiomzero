"""
Axiom Zero - RL Proof Agent Neural Networks
Policy and Value networks for AlphaZero-style proof search.
Uses GNN/Transformer to encode proof states into latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional


class ProofStateEncoder(nn.Module):
    """
    Encodes proof state into latent representation.
    Uses Graph Neural Network to capture logical structure.

    Graph representation:
    - Nodes: hypotheses, goal, variables
    - Edges: logical dependencies, type relationships
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3,
                 use_gat: bool = True):
        """
        Initialize state encoder.

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            use_gat: Use GAT (Graph Attention) instead of GCN
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            # Input: encoded hypothesis/goal features
            nn.Linear(50, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN layers
        if use_gat:
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                for _ in range(num_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Global state aggregator
        self.global_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, graph_data: Data) -> torch.Tensor:
        """
        Encode proof state graph to latent vector.

        Args:
            graph_data: PyG Data object with:
                - x: node features [num_nodes, 50]
                - edge_index: graph connectivity [2, num_edges]
                - edge_attr: edge features [num_edges, 20]
                - batch: batch indices [num_nodes]

        Returns:
            Latent representation [batch_size, hidden_dim]
        """
        # Encode node features
        x = self.node_encoder(graph_data.x)

        # Encode edge features
        edge_attr = self.edge_encoder(graph_data.edge_attr)

        # Apply GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x = gnn(x, graph_data.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        # Global pooling
        global_state = global_mean_pool(x, graph_data.batch)

        # Aggregate
        latent = self.global_aggregator(global_state)

        return latent


class PolicyNetwork(nn.Module):
    """
    Policy head: outputs probability distribution over tactics.
    Predicts which tactic to apply next.
    """

    def __init__(self, hidden_dim: int = 256, num_tactics: int = 19,
                 max_parameters: int = 10):
        """
        Initialize policy network.

        Args:
            hidden_dim: Hidden dimension (must match encoder)
            num_tactics: Number of available tactics
            max_parameters: Maximum number of parameters per tactic
        """
        super().__init__()
        self.num_tactics = num_tactics

        # Tactic selection head
        self.tactic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tactics)
        )

        # Parameter prediction head (for parameterized tactics)
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tactics * max_parameters)
        )

        self.max_parameters = max_parameters

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict tactic probabilities and parameters.

        Args:
            latent: Latent state representation [batch_size, hidden_dim]

        Returns:
            Tuple of (tactic_probs, parameter_logits)
            - tactic_probs: [batch_size, num_tactics]
            - parameter_logits: [batch_size, num_tactics * max_parameters]
        """
        # Tactic probabilities
        tactic_logits = self.tactic_head(latent)
        tactic_probs = F.softmax(tactic_logits, dim=-1)

        # Parameter predictions
        parameter_logits = self.parameter_head(latent)
        parameter_logits = parameter_logits.view(
            -1, self.num_tactics, self.max_parameters)

        return tactic_probs, parameter_logits


class ValueNetwork(nn.Module):
    """
    Value head: estimates distance to proof completion.
    Predicts expected reward from current state.
    """

    def __init__(self, hidden_dim: int = 256):
        """
        Initialize value network.

        Args:
            hidden_dim: Hidden dimension (must match encoder)
        """
        super().__init__()

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict value (expected reward) of state.

        Args:
            latent: Latent state representation [batch_size, hidden_dim]

        Returns:
            Value predictions [batch_size, 1]
        """
        value = self.value_head(latent)
        return value


class ProofAgent(nn.Module):
    """
    Complete AlphaZero-style proof agent.
    Combines encoder, policy, and value networks.
    """

    def __init__(self, hidden_dim: int = 256, num_tactics: int = 19,
                 num_gnn_layers: int = 3, use_gat: bool = True):
        """
        Initialize proof agent.

        Args:
            hidden_dim: Hidden dimension
            num_tactics: Number of available tactics
            num_gnn_layers: Number of GNN layers
            use_gat: Use GAT instead of GCN
        """
        super().__init__()

        self.encoder = ProofStateEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            use_gat=use_gat
        )

        self.policy = PolicyNetwork(
            hidden_dim=hidden_dim,
            num_tactics=num_tactics
        )

        self.value = ValueNetwork(hidden_dim=hidden_dim)

        # Store configuration
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics

    def forward(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode state and predict policy + value.

        Args:
            graph_data: Proof state graph

        Returns:
            Tuple of (tactic_probs, parameter_logits, value)
        """
        # Encode state
        latent = self.encoder(graph_data)

        # Predict policy
        tactic_probs, parameter_logits = self.policy(latent)

        # Predict value
        value = self.value(latent)

        return tactic_probs, parameter_logits, value

    def predict(self, graph_data: Data, temperature: float = 1.0) -> Dict:
        """
        Make prediction for proof search.

        Args:
            graph_data: Proof state graph
            temperature: Temperature for policy sampling (lower = more greedy)

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            latent = self.encoder(graph_data)
            tactic_probs, parameter_logits = self.policy(latent)
            value = self.value(latent)

            # Apply temperature
            tactic_probs = tactic_probs ** (1.0 / temperature)
            tactic_probs = tactic_probs / \
                tactic_probs.sum(dim=-1, keepdim=True)

            # Sample tactic
            tactic_idx = torch.multinomial(tactic_probs, num_samples=1)

            # Get best parameters for selected tactic
            batch_idx = torch.arange(tactic_probs.size(0))
            param_logits = parameter_logits[batch_idx, tactic_idx.squeeze()]
            params = torch.argmax(param_logits, dim=-1)

        return {
            'tactic_idx': tactic_idx.item(),
            'tactic_probs': tactic_probs,
            'params': params,
            'value': value.item(),
            'latent': latent
        }


class ProofStateGraphBuilder:
    """
    Converts ProofState objects to PyG Data objects for GNN.
    """

    @staticmethod
    def build_graph(proof_state, max_nodes: int = 50) -> Data:
        """
        Build graph representation of proof state.

        Node types:
        - 0: Goal node
        - 1: Hypothesis node
        - 2: Variable node

        Features per node (50-dim):
        - One-hot node type (3 dims)
        - Encoded target/context type (32 dims)
        - Structural features (15 dims): depth, index, etc.

        Args:
            proof_state: ProofState object
            max_nodes: Maximum nodes in graph

        Returns:
            PyG Data object
        """
        # Collect nodes
        nodes = []
        node_types = []

        # Add goal node
        goal = proof_state.current_goal
        if goal:
            goal_features = ProofStateGraphBuilder._encode_goal(goal)
            nodes.append(goal_features)
            node_types.append(0)  # Goal type

        # Add hypothesis nodes
        if goal:
            for hyp in goal.context:
                hyp_features = ProofStateGraphBuilder._encode_hypothesis(hyp)
                nodes.append(hyp_features)
                node_types.append(1)  # Hypothesis type

        # Convert to tensor
        if not nodes:
            # Empty graph fallback
            x = torch.zeros((1, 50))
            node_type = torch.zeros(1, dtype=torch.long)
        else:
            x = torch.stack(nodes)
            node_type = torch.tensor(node_types, dtype=torch.long)

        # Build edges (fully connected for simplicity)
        num_nodes = x.size(0)
        edge_index = torch.zeros((2, num_nodes * num_nodes), dtype=torch.long)
        edge_attr = torch.zeros((num_nodes * num_nodes, 20))

        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index[0, idx] = i
                edge_index[1, idx] = j
                # Edge features: relationship type
                # Is connected to goal
                edge_attr[idx, 0] = 1.0 if i == 0 else 0.0
                edge_attr[idx, 1] = float(i == j)  # Self-loop
                idx += 1

        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Pad to consistent graph size for batching
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            x = torch.cat([x, torch.zeros((pad_size, x.size(1)))], dim=0)
            batch = torch.cat(
                [batch, torch.zeros(pad_size, dtype=torch.long)])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )

    @staticmethod
    def _encode_goal(goal) -> torch.Tensor:
        """Encode goal into feature vector."""
        features = torch.zeros(50)

        # Node type (one-hot)
        features[0] = 1.0  # Goal type

        # Encode target complexity
        target = goal.target
        features[3] = len(target) / 100.0  # Normalized length
        features[4] = float('∀' in target or 'forall' in target)
        features[5] = float('∃' in target or 'exists' in target)
        features[6] = float('→' in target or '->' in target)
        features[7] = float('=' in target)
        features[8] = float('∧' in target or 'and' in target.lower())
        features[9] = float('∨' in target or 'or' in target.lower())

        # Structural features
        features[20] = goal.depth / 10.0  # Normalized depth
        features[21] = len(goal.tactic_history) / 20.0  # Tactics applied

        return features

    @staticmethod
    def _encode_hypothesis(hyp) -> torch.Tensor:
        """Encode hypothesis into feature vector."""
        features = torch.zeros(50)

        # Node type (one-hot)
        features[1] = 1.0  # Hypothesis type

        # Encode hypothesis type
        hyp_type = hyp.var_type
        features[10] = float('Matrix' in hyp_type or 'Tensor' in hyp_type)
        features[11] = float('Nat' in hyp_type or 'ℕ' in hyp_type)
        features[12] = float('Real' in hyp_type or 'ℝ' in hyp_type)
        features[13] = float('Prop' in hyp_type)

        # Is it an assumption or definition
        features[18] = float(hyp.is_hypothesis)

        return features
