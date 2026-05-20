#!/usr/bin/env python3
"""
Complete Training & Evaluation Pipeline
Curriculum Learning → Training → Evaluation
"""

import time
import json
from pathlib import Path


def main():
    """Run complete training and evaluation pipeline."""
    
    print("="*70)
    print("AXIOM ZERO - COMPLETE TRAINING & EVALUATION PIPELINE")
    print("="*70)
    print()
    
    # Phase 1: Curriculum Learning
    print("="*70)
    print("PHASE 1: CURRICULUM LEARNING")
    print("="*70)
    print()
    print("Training strategy:")
    print("  1. Start with easy problems (arithmetic)")
    print("  2. Progress to medium (lists, loops)")
    print("  3. Advance to hard (tensors, matrices)")
    print("  4. Adaptive difficulty based on performance")
    print()
    
    from curriculum_learning import CurriculumTrainer
    
    trainer = CurriculumTrainer(use_wandb=False)
    
    print("Starting curriculum training...")
    curriculum_summary = trainer.run_curriculum()
    
    print("\n✓ Curriculum learning complete!")
    print(f"  Levels completed: {curriculum_summary['current_level']}/{curriculum_summary['total_levels']}")
    
    # Phase 2: Evaluation
    print("\n" + "="*70)
    print("PHASE 2: BENCHMARK EVALUATION")
    print("="*70)
    print()
    
    from evaluate import BenchmarkEvaluator
    
    evaluator = BenchmarkEvaluator()
    eval_summary = evaluator.run_evaluation()
    
    print("\n✓ Evaluation complete!")
    print(f"  Success rate: {eval_summary['success_rate']:.1%}")
    print(f"  Benchmarks solved: {eval_summary['successful']}/{eval_summary['total_benchmarks']}")
    
    # Phase 3: Results Analysis
    print("\n" + "="*70)
    print("PHASE 3: RESULTS ANALYSIS")
    print("="*70)
    print()
    
    # Print comprehensive summary
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("-"*70)
    print()
    
    # Training results
    print("1. TRAINING RESULTS")
    print("-"*70)
    
    for level_num in range(1, 6):
        if level_num in curriculum_summary['levels']:
            level_data = curriculum_summary['levels'][level_num]
            status_icon = "✓" if level_data['status'] == 'completed' else "▶" if level_data['status'] == 'current' else "○"
            
            print(f"  {status_icon} Level {level_num}: {level_data['name']}")
            print(f"     Episodes: {level_data['episodes']}")
            print(f"     Success Rate: {level_data['success_rate']:.1%}")
            print(f"     Avg Steps: {level_data['avg_steps']:.1f}")
            print()
    
    # Evaluation results
    print("2. EVALUATION RESULTS")
    print("-"*70)
    print(f"  Overall Success Rate: {eval_summary['success_rate']:.1%}")
    print(f"  Total Benchmarks: {eval_summary['total_benchmarks']}")
    print(f"  Successful: {eval_summary['successful']}")
    print(f"  Failed: {eval_summary['failed']}")
    print()
    
    print(f"  By Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in eval_summary['by_difficulty']:
            stats = eval_summary['by_difficulty'][diff]
            print(f"    {diff:10s}: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})")
    
    print()
    print(f"  Proof Quality:")
    print(f"    Avg Proof Length: {eval_summary['avg_proof_length']:.1f} tactics")
    print(f"    Avg Search Time: {eval_summary['avg_search_time']:.2f}s")
    print(f"    Avg MCTS Sims: {eval_summary['avg_mcts_simulations']:.0f}")
    
    # Key insights
    print("\n3. KEY INSIGHTS")
    print("-"*70)
    
    # Analyze performance patterns
    easy_rate = eval_summary['by_difficulty'].get('easy', {}).get('success_rate', 0)
    medium_rate = eval_summary['by_difficulty'].get('medium', {}).get('success_rate', 0)
    hard_rate = eval_summary['by_difficulty'].get('hard', {}).get('success_rate', 0)
    
    if easy_rate > 0.9:
        print("  ✓ Excellent performance on easy problems (>90%)")
    if medium_rate > 0.7:
        print("  ✓ Good performance on medium problems (>70%)")
    if hard_rate > 0.4:
        print("  ⊗ Room for improvement on hard problems")
    
    if eval_summary['avg_proof_length'] < 6:
        print("  ✓ Short, efficient proofs (avg < 6 tactics)")
    if eval_summary['avg_search_time'] < 3:
        print("  ✓ Fast proof search (avg < 3s)")
    
    print()
    print("  Curriculum Learning Benefits:")
    print("    • Progressive difficulty prevents early failures")
    print("    • Skills transfer from easy to hard problems")
    print("    • Efficient use of training compute")
    print("    • Measurable progress at each level")
    
    # Save complete results
    print("\n4. SAVING RESULTS")
    print("-"*70)
    
    results = {
        'timestamp': time.time(),
        'curriculum': curriculum_summary,
        'evaluation': {k: v for k, v in eval_summary.items() if k != 'results'},
        'evaluation_details': eval_summary.get('results', []),
    }
    
    results_file = Path("training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  ✓ Results saved to: {results_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print()
    print("Summary:")
    print(f"  ✓ Curriculum learning: {curriculum_summary['current_level']}/{curriculum_summary['total_levels']} levels")
    print(f"  ✓ Benchmark evaluation: {eval_summary['success_rate']:.1%} success rate")
    print(f"  ✓ Results saved: {results_file}")
    print()
    print("Next steps:")
    print("  1. Analyze failed benchmarks for improvement opportunities")
    print("  2. Increase MCTS simulations for hard problems")
    print("  3. Add more training episodes at challenging levels")
    print("  4. Fine-tune neural network architecture")
    print("  5. Expand benchmark suite with more diverse problems")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
