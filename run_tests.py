#!/usr/bin/env python3
"""
Run full test suite for Axiom Zero
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file: str) -> bool:
    """Run a single test file."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("="*70)
    print("AXIOM ZERO - FULL TEST SUITE")
    print("="*70)
    
    tests = [
        "test_ast_extractor.py",
        "test_abstract_interpreter.py",
        "test_spec_ingestion.py",
        "test_proof_engine.py",
        "test_rl_agent.py",
        "test_rl_concepts.py",
    ]
    
    results = {}
    
    for test in tests:
        if Path(test).exists():
            results[test] = run_test(test)
        else:
            print(f"\n⊗ Skipped: {test} (not found)")
            results[test] = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for test, result in results.items():
        if result is True:
            print(f"✓ {test}")
        elif result is False:
            print(f"✗ {test}")
        else:
            print(f"⊗ {test} (skipped)")
    
    print()
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print()
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
