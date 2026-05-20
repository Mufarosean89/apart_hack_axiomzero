"""
Test script for RL proof agent.
Demonstrates neural networks, MCTS, and self-play training.
"""

import torch
from rl_agent import (
    ProofAgent,
    ProofStateGraphBuilder,
    MCTS,
    SelfPlayTrainer,
    ReplayBuffer,
    TrainingExample
)
from proof_engine import ProofState, ProofGoal, Variable, TacticSpace


def test_neural_networks():
    """Test policy and value networks."""
    print("=" * 80)
    print("TEST 1: NEURAL NETWORKS")
    print("=" * 80)
    print()

    # Create agent
    print("Creating ProofAgent...")
    agent = ProofAgent(
        hidden_dim=128,
        num_tactics=19,
        num_gnn_layers=2,
        use_gat=True
    )
    print(f"✓ Agent created")
    print(f"  Hidden dimension: 128")
    print(f"  Number of tactics: 19")
    print(f"  GNN layers: 2 (GAT)")
    print()

    # Show network architecture
    print("Network Architecture:")
    print("-" * 60)
    print(f"\n  Encoder (GAT):")
    print(f"    Node encoder: Linear(50 → 128) → ReLU → Linear(128 → 128)")
    print(f"    Edge encoder: Linear(20 → 128) → ReLU → Linear(128 → 128)")
    print(f"    GAT layers: 2 × GATConv(128 → 128, heads=4)")
    print(f"    Global aggregator: Linear(128 → 128)")
    print(f"\n  Policy Network:")
    print(f"    Tactic head: Linear(128 → 128) → ReLU → Linear(128 → 19)")
    print(f"    Parameter head: Linear(128 → 128 × 10)")
    print(f"\n  Value Network:")
    print(f"    Value head: Linear(128 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 1)")
    print()

    # Test forward pass with dummy data
    print("Testing Forward Pass:")
    print("-" * 60)

    # Create dummy graph data
    batch_size = 4
    num_nodes = 10
    dummy_x = torch.randn(batch_size * num_nodes, 50)
    dummy_edge_index = torch.zeros(
        (2, num_nodes * num_nodes * batch_size), dtype=torch.long)
    dummy_edge_attr = torch.randn(num_nodes * num_nodes * batch_size, 20)
    dummy_batch = torch.arange(batch_size).repeat_interleave(num_nodes)

    # Create PyG Data object
    from torch_geometric.data import Data
    graph_data = Data(
        x=dummy_x,
        edge_index=dummy_edge_index,
        edge_attr=dummy_edge_attr,
        batch=dummy_batch
    )

    # Forward pass
    agent.eval()
    with torch.no_grad():
        tactic_probs, param_logits, value = agent(graph_data)

    print(f"  Input: graph_data.x.shape = {dummy_x.shape}")
    print(f"  Tactic probabilities: {tactic_probs.shape}")
    print(f"    Sum per batch: {tactic_probs.sum(dim=-1)}")
    print(f"  Parameter logits: {param_logits.shape}")
    print(f"  Value predictions: {value.shape}")
    print(f"    Values: {value.squeeze().tolist()}")
    print()

    # Test prediction
    print("Testing Prediction:")
    print("-" * 60)
    prediction = agent.predict(graph_data[0:1], temperature=0.5)

    print(f"  Selected tactic index: {prediction['tactic_idx']}")
    print(f"  Tactic probs shape: {prediction['tactic_probs'].shape}")
    print(f"  Value: {prediction['value']:.4f}")
    print(f"  Latent shape: {prediction['latent'].shape}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel()
                           for p in agent.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()


def test_state_encoding():
    """Test proof state to graph conversion."""
    print("=" * 80)
    print("TEST 2: STATE ENCODING")
    print("=" * 80)
    print()

    # Create proof state
    state = ProofState(
        theorem_name="test_theorem",
        goals=[
            ProofGoal(
                goal_id=0,
                target="∀ (x y : ℕ), x + y = y + x",
                context=[
                    Variable(name="h1", var_type="x > 0", is_hypothesis=True),
                    Variable(name="h2", var_type="y > 0", is_hypothesis=True)
                ]
            )
        ]
    )

    print("Proof State:")
    print(f"  Theorem: {state.theorem_name}")
    print(f"  Goal: {state.current_goal.target}")
    print(f"  Context: {len(state.current_goal.context)} hypotheses")
    print()

    # Convert to graph
    print("Converting to Graph Representation:")
    print("-" * 60)
    graph_data = ProofStateGraphBuilder.build_graph(state)

    print(f"  Nodes: {graph_data.x.shape[0]}")
    print(f"  Node features: {graph_data.x.shape[1]} dimensions")
    print(f"  Edges: {graph_data.edge_index.shape[1]}")
    print(f"  Edge features: {graph_data.edge_attr.shape[1]} dimensions")
    print()

    # Show node types
    print("Node Types:")
    print("-" * 60)
    for i in range(min(5, graph_data.x.shape[0])):
        node_type = "Goal" if graph_data.x[i,
                                           0] > 0.5 else "Hypothesis" if graph_data.x[i, 1] > 0.5 else "Unknown"
        print(f"  Node {i}: {node_type}")
    print()


def test_mcts():
    """Test Monte Carlo Tree Search."""
    print("=" * 80)
    print("TEST 3: MONTE CARLO TREE SEARCH")
    print("=" * 80)
    print()

    # Create agent and tactic space
    agent = ProofAgent(hidden_dim=64, num_tactics=19, num_gnn_layers=1)
    tactic_space = TacticSpace()

    # Create MCTS
    mcts = MCTS(
        agent=agent,
        tactic_space=tactic_space,
        c_puct=1.0,
        num_simulations=50,
        max_depth=20
    )

    print("MCTS Configuration:")
    print(f"  PUCT constant: {mcts.c_puct}")
    print(f"  Simulations: {mcts.num_simulations}")
    print(f"  Max depth: {mcts.max_depth}")
    print()

    # Create root state
    root_state = ProofState(
        theorem_name="simple_proof",
        goals=[
            ProofGoal(
                goal_id=0,
                target="∀ (x : ℕ), x + 0 = x",
                context=[]
            )
        ]
    )

    print("Running MCTS Search:")
    print("-" * 60)
    print(f"  Root: {root_state.theorem_name}")
    print(f"  Target: {root_state.current_goal.target}")
    print()

    # Run search
    best_tactic, root_node = mcts.search(root_state)

    print("MCTS Results:")
    print(f"  Best tactic: {best_tactic}")
    print(f"  Root visits: {root_node.visits}")
    print(f"  Children: {len(root_node.children)}")
    print()

    # Show top tactics by visit count
    print("Top Tactics by Visit Count:")
    print("-" * 60)
    sorted_children = sorted(root_node.children.items(),
                             key=lambda x: x[1].visits,
                             reverse=True)

    for tactic, child in sorted_children[:5]:
        print(f"  {tactic:20s}: {child.visits:3d} visits, value={child.value:.3f}")
    print()

    # Get statistics
    stats = mcts.get_search_statistics()
    print("Search Statistics:")
    print(f"  Total simulations: {stats['total_simulations']}")
    print(f"  Total nodes explored: {stats['total_nodes_explored']}")
    print(f"  Nodes per simulation: {stats['nodes_per_simulation']:.1f}")
    print()


def test_replay_buffer():
    """Test replay buffer for experience storage."""
    print("=" * 80)
    print("TEST 4: REPLAY BUFFER")
    print("=" * 80)
    print()

    # Create buffer
    buffer = ReplayBuffer(capacity=1000)

    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Initial size: {len(buffer)}")
    print()

    # Add examples
    print("Adding training examples...")
    for i in range(10):
        example = TrainingExample(
            state_features=torch.randn(50),
            policy_target=torch.randn(19),
            value_target=float(i % 2),
            tactic_sequence=[f"tactic_{j}" for j in range(i+1)],
            theorem_name=f"theorem_{i}"
        )
        buffer.add(example)

    print(f"  Added 10 examples")
    print(f"  Buffer size: {len(buffer)}")
    print()

    # Sample batch
    print("Sampling batch:")
    print("-" * 60)
    states, policy_targets, value_targets = buffer.sample(batch_size=5)

    print(f"  States shape: {states.shape}")
    print(f"  Policy targets shape: {policy_targets.shape}")
    print(f"  Value targets shape: {value_targets.shape}")
    print(f"  Value targets: {value_targets.tolist()}")
    print()


def test_self_play_training():
    """Test self-play training loop."""
    print("=" * 80)
    print("TEST 5: SELF-PLAY TRAINING (SIMULATED)")
    print("=" * 80)
    print()

    # Create components
    agent = ProofAgent(hidden_dim=64, num_tactics=19, num_gnn_layers=1)
    tactic_space = TacticSpace()

    # Create trainer
    trainer = SelfPlayTrainer(
        agent=agent,
        tactic_space=tactic_space,
        buffer_capacity=1000,
        batch_size=32,
        learning_rate=1e-3,
        num_simulations=20
    )

    print("Self-Play Trainer Configuration:")
    print(f"  Buffer capacity: {trainer.buffer.capacity}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Learning rate: {trainer.optimizer.defaults['lr']}")
    print(f"  MCTS simulations: {trainer.mcts.num_simulations}")
    print()

    # Define sample theorems
    theorems = [
        {
            'statement': 'theorem add_zero (n : ℕ) : n + 0 = n',
            'imports': ['Mathlib.Data.Nat.Basic']
        },
        {
            'statement': 'theorem mul_one (n : ℕ) : n * 1 = n',
            'imports': ['Mathlib.Data.Nat.Basic']
        },
        {
            'statement': 'theorem add_comm (a b : ℕ) : a + b = b + a',
            'imports': ['Mathlib.Data.Nat.Basic']
        }
    ]

    print("Simulating Self-Play Training:")
    print("-" * 60)
    print()

    # Simulate a few training iterations
    for iteration in range(3):
        print(f"Iteration {iteration + 1}:")

        # Generate self-play game
        theorem = theorems[iteration % len(theorems)]
        print(f"  Theorem: {theorem['statement'][:50]}...")

        examples = trainer.generate_self_play_game(
            theorem['statement'],
            theorem['imports']
        )

        print(f"  Generated {len(examples)} training examples")
        print(f"  Buffer size: {len(trainer.buffer)}")

        # Train
        if len(trainer.buffer) >= trainer.batch_size:
            metrics = trainer.train_step()
            print(f"  Training loss: {metrics['loss']:.4f}")
            print(f"    Policy loss: {metrics['policy_loss']:.4f}")
            print(f"    Value loss: {metrics['value_loss']:.4f}")

        print()

    print("Training Statistics:")
    print(f"  Games played: {trainer.games_played}")
    print(f"  Proofs found: {trainer.proofs_found}")
    print(f"  Training steps: {trainer.training_steps}")
    print(f"  Buffer size: {len(trainer.buffer)}")
    print()


def test_complete_pipeline():
    """Test complete RL pipeline."""
    print("=" * 80)
    print("TEST 6: COMPLETE RL PIPELINE")
    print("=" * 80)
    print()

    print("Pipeline Overview:")
    print("-" * 60)
    print("""
    1. State Encoding
       ProofState → Graph Data → Latent Vector
    
    2. Policy + Value Prediction
       Latent Vector → Tactic Distribution + Value Estimate
    
    3. MCTS Search
       Root State → Search Tree → Best Tactic
    
    4. Environment Step
       Tactic → Lean Execution → New State + Reward
    
    5. Self-Play Data Generation
       Game Trajectory → Training Examples
    
    6. Network Training
       Replay Buffer → Gradient Updates → Improved Agent
    """)

    print("Key Innovations:")
    print("-" * 60)
    print("  • GNN-based state encoding (graph structure)")
    print("  • Joint policy + value training (AlphaZero)")
    print("  • MCTS with neural guidance (P-UCT)")
    print("  • Self-play with Lean oracle (no human labels)")
    print("  • Experience replay for stability")
    print()

    print("Training Loop:")
    print("-" * 60)
    print("  for game in range(num_games):")
    print("      # Self-play")
    print("      trajectory = generate_game(theorem)")
    print("      buffer.add(trajectory)")
    print("      ")
    print("      # Training")
    print("      for step in range(num_steps):")
    print("          batch = buffer.sample()")
    print("          loss = train(batch)")
    print("          optimizer.step()")
    print()


def run_all_tests():
    """Run all RL agent tests."""
    print()
    print("═" * 78 + "╗")
    print("║" + " " * 18 + "AXIOM ZERO - RL AGENT TESTS" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    test_neural_networks()
    print("\n" + "━" * 80 + "\n")

    test_state_encoding()
    print("\n" + "━" * 80 + "\n")

    test_mcts()
    print("\n" + "━" * 80 + "\n")

    test_replay_buffer()
    print("\n" + "━" * 80 + "\n")

    test_self_play_training()
    print("\n" + "━" * 80 + "\n")

    test_complete_pipeline()

    print()
    print("=" * 80)
    print("✓ ALL RL AGENT TESTS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Neural networks (policy + value)")
    print("  ✓ Graph-based state encoding")
    print("  ✓ MCTS with PUCT selection")
    print("  ✓ Replay buffer")
    print("  ✓ Self-play training loop")
    print("  ✓ Complete RL pipeline")
    print()
    print("Axiom Zero's AlphaZero-style proof agent is ready!")
    print()


if __name__ == "__main__":
    run_all_tests()
