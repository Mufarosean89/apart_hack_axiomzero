#!/usr/bin/env python3
"""
Integration test for PyTorch scaling, parallel self-play, and proof caching.
"""

import time
import sys


def test_pytorch_parser():
    """Test PyTorch model parser."""
    print("\n" + "="*70)
    print("TEST 1: PyTorch Model Parser")
    print("="*70)
    
    try:
        from pytorch_parser import PyTorchModelParser
        
        example_code = '''
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
'''
        
        parser = PyTorchModelParser()
        model_ir = parser.parse_model(example_code)
        
        # Compute shapes
        parser.compute_shapes([1, 784])
        parser.count_parameters()
        
        print(f"✓ Model parsed: {model_ir.model_name}")
        print(f"  Layers: {len(model_ir.layers)}")
        print(f"  Input shape: {model_ir.input_shape}")
        print(f"  Output shape: {model_ir.output_shape}")
        print(f"  Parameters: {model_ir.parameters_count:,}")
        
        for layer in model_ir.layers:
            print(f"    {layer.name}: {layer.layer_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_selfplay():
    """Test parallel self-play."""
    print("\n" + "="*70)
    print("TEST 2: Parallel Self-Play")
    print("="*70)
    
    try:
        from parallel_selfplay import ParallelSelfPlay, DistributedReplayBuffer
        from benchmarks import BENCHMARKS
        
        # Test parallel generation
        self_play = ParallelSelfPlay(num_workers=2)
        
        print(f"✓ Parallel self-play initialized")
        print(f"  Workers: {self_play.num_workers}")
        print(f"  MCTS sims: {self_play.mcts_simulations}")
        
        # Generate small batch
        test_theorems = BENCHMARKS[:2]  # Just 2 theorems
        
        start_time = time.time()
        games = self_play.generate_games(test_theorems, games_per_theorem=3)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Generated {len(games)} games in {elapsed:.2f}s")
        print(f"  Rate: {len(games)/elapsed:.1f} games/s")
        
        successes = sum(1 for g in games if g.success)
        print(f"  Success rate: {successes}/{len(games)}")
        
        # Test replay buffer
        buffer = DistributedReplayBuffer(capacity=1000)
        buffer.add_batch(games)
        
        stats = buffer.get_stats()
        print(f"\n✓ Replay buffer:")
        print(f"  Size: {stats['size']}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"  Avg steps: {stats.get('avg_steps', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_proof_caching():
    """Test proof caching system."""
    print("\n" + "="*70)
    print("TEST 3: Proof Caching & Transfer Learning")
    print("="*70)
    
    try:
        from proof_caching import ProofCachingSystem
        
        # Initialize system
        system = ProofCachingSystem(cache_file="test_cache.json")
        
        print(f"✓ Proof caching system initialized")
        
        # Simulate solving theorems
        theorems = [
            {'id': 'add_comm', 'statement': '∀ (a b : ℕ), a + b = b + a', 'difficulty': 'easy'},
            {'id': 'mul_assoc', 'statement': '∀ (a b c : ℕ), (a * b) * c = a * (b * c)', 'difficulty': 'medium'},
        ]
        
        def mock_solver(statement, init_params):
            return {
                'success': True,
                'proof_tactics': ['intro', 'induction', 'simp'],
                'difficulty': 'medium',
            }
        
        # Solve and cache
        for theorem in theorems:
            result = system.solve_with_caching(
                theorem['id'],
                theorem['statement'],
                mock_solver
            )
            print(f"  ✓ {theorem['id']}: {result['source']}")
        
        # Test cache retrieval
        print(f"\n✓ Cache statistics:")
        stats = system.get_cache_stats()
        print(f"  Size: {stats['size']}")
        print(f"  Avg proof length: {stats.get('avg_proof_length', 0):.1f}")
        
        # Test transfer learning
        print(f"\n✓ Transfer learning:")
        new_theorem = "∀ (x y : ℕ), x + y = y + x"
        suggestions = system.transfer_agent.initialize_search(new_theorem)
        
        print(f"  New theorem: {new_theorem}")
        print(f"  Similar proofs: {suggestions.get('num_similar_proofs', 0)}")
        print(f"  Suggested tactics: {suggestions.get('suggested_tactics', [])[:3]}")
        
        # Test similarity search
        similar = system.cache.find_similar_proofs(new_theorem, top_k=2)
        print(f"\n✓ Similar proofs found: {len(similar)}")
        for score, proof in similar:
            print(f"  - {proof.theorem_id} (similarity: {score:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*70)
    print("AXIOM ZERO - ADVANCED FEATURES INTEGRATION TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: PyTorch Parser
    results['pytorch_parser'] = test_pytorch_parser()
    
    # Test 2: Parallel Self-Play
    results['parallel_selfplay'] = test_parallel_selfplay()
    
    # Test 3: Proof Caching
    results['proof_caching'] = test_proof_caching()
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total - passed}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
