"""
Test script for specification ingestion module.
Demonstrates parsing formal specifications from decorated Python and Lean files.
"""

from spec_ingestion import (
    parse_specifications,
    extract_from_decorators,
    extract_from_lean_file
)


# Sample 1: Python with decorator-based specifications
PYTHON_DECORATOR_SAMPLE = '''
import torch
from typing import List

@requires("batch_size > 0")
@requires("hidden_dim > 0")
@ensures("output.shape[0] == batch_size")
@ensures("output.shape[1] == hidden_dim")
def create_tensor(batch_size: int, hidden_dim: int) -> torch.Tensor:
    """Create a tensor with specified dimensions."""
    return torch.zeros(batch_size, hidden_dim)

@requires("A.shape[1] == B.shape[0]")
@requires("B.shape[1] == C.shape[0]")
@ensures("result.shape[0] == A.shape[0]")
@ensures("result.shape[1] == C.shape[1]")
def matrix_chain_multiply(A: torch.Tensor, B: torch.Tensor, 
                         C: torch.Tensor) -> torch.Tensor:
    """Multiply three matrices with shape validation."""
    AB = torch.matmul(A, B)
    ABC = torch.matmul(AB, C)
    return ABC

@requires("len(data) > 0")
@ensures("result >= min(data)")
@ensures("result <= max(data)")
@invariant("total >= 0")
def compute_statistics(data: List[float]) -> dict:
    """Compute statistics with bounds checking."""
    total = 0.0
    for value in data:
        total = total + value
    
    mean = total / len(data)
    return {'mean': mean, 'sum': total}

@requires("x.shape[0] > 0")
@requires("x.shape[1] > 0")
@ensures("output.shape == x.shape")
def apply_relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU activation."""
    return torch.relu(x)
'''

# Sample 2: Python with inline specifications
PYTHON_INLINE_SAMPLE = '''
import torch

def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # @requires b != 0
    # @ensures result * b == a
    assert b is not None  # Safety check
    
    result = torch.div(a, b)
    return result

def transformer_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # @requires Q.shape[-1] == K.shape[-1]
    # @requires K.shape[-2] == V.shape[-2]
    # @ensures output.shape[0] == Q.shape[0]
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    attention = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention, V)
    return output

def loop_with_invariant(n: int) -> int:
    # @invariant sum >= 0
    # @ensures result == n * (n + 1) / 2
    sum = 0
    for i in range(n):
        # @invariant sum == i * (i + 1) / 2
        sum = sum + i
    return sum
'''

# Sample 3: Lean 4 specification
LEAN_SPEC_SAMPLE = '''
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

-- Matrix multiplication associativity theorem
theorem matrix_mul_assoc {n m k l : ℕ} (A : Matrix (Fin n) (Fin m) ℝ) 
  (B : Matrix (Fin m) (Fin k) ℝ) (C : Matrix (Fin k) (Fin l) ℝ) :
  (A * B) * C = A * (B * C) := by
  sorry

-- Transformer attention specification
def safe_attention (Q K V : Matrix (Fin n) (Fin d) ℝ) 
  (h_dim : d > 0) (h_batch : n > 0) : Matrix (Fin n) (Fin d) ℝ
  requires d > 0
  requires n > 0
  ensures ∀ i j, result i j ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

-- Shape preservation lemma
lemma reshape_preserves_elements {n m p : ℕ} (h : n * m = p) 
  (A : Matrix (Fin n) (Fin m) ℝ) :
  (A.reshape h).size = A.size := by
  sorry

-- Norm bounds for neural network layers
theorem layer_norm_bound (W : Matrix (Fin n) (Fin m) ℝ) (x : Fin m → ℝ) 
  (h_W : ‖W‖ ≤ 1) (h_x : ‖x‖ ≤ 1) :
  ‖W * x‖ ≤ 1 := by
  sorry

variable (batch_size seq_len hidden_dim : ℕ)
hypothesis h_pos : batch_size > 0
hypothesis h_dim : hidden_dim > 0
'''


def test_python_decorators():
    """Test parsing Python decorator-based specifications."""
    print("=" * 80)
    print("TEST 1: PYTHON DECORATOR-BASED SPECIFICATIONS")
    print("=" * 80)
    print()

    obligation_set = extract_from_decorators(
        PYTHON_DECORATOR_SAMPLE, "pytorch_model.py")

    print(obligation_set.summary_str())
    print()

    print("=" * 80)
    print("DETAILED OBLIGATIONS")
    print("=" * 80)
    print()

    # Show by function
    for func_name, obligations in obligation_set.by_function.items():
        print(f"Function: {func_name}")
        print("-" * 60)
        for i, ob in enumerate(obligations, 1):
            print(f"  {i}. [{ob.kind.value.upper()}] {ob.statement}")
            print(f"     Variables: {ob.variables}")
            print(f"     Priority: {ob.priority}")
        print()


def test_python_inline():
    """Test parsing inline specifications."""
    print("=" * 80)
    print("TEST 2: PYTHON INLINE SPECIFICATIONS")
    print("=" * 80)
    print()

    obligation_set = parse_specifications(
        PYTHON_INLINE_SAMPLE, 'inline', "inline_specs.py")

    print(obligation_set.summary_str())
    print()

    print("=" * 80)
    print("EXTRACTED OBLIGATIONS")
    print("=" * 80)
    print()

    for ob in obligation_set.obligations:
        print(f"  [{ob.kind.value.upper():20s}] {ob.statement}")
        if ob.function_name:
            print(f"                       Function: {ob.function_name}")
        print()


def test_lean_specs():
    """Test parsing Lean 4 specifications."""
    print("=" * 80)
    print("TEST 3: LEAN 4 SPECIFICATIONS")
    print("=" * 80)
    print()

    obligation_set = extract_from_lean_file(LEAN_SPEC_SAMPLE, "specs.lean")

    print(obligation_set.summary_str())
    print()

    print("=" * 80)
    print("THEOREMS AND LEMMAS")
    print("=" * 80)
    print()

    for ob in obligation_set.theorems:
        print(f"Theorem: {ob.tags[1] if len(ob.tags) > 1 else 'unknown'}")
        print(f"  Statement: {ob.statement[:100]}...")
        print(f"  Variables: {ob.variables}")
        print()

    print("=" * 80)
    print("FUNCTION CONTRACTS")
    print("=" * 80)
    print()

    for ob in obligation_set.preconditions + obligation_set.postconditions:
        print(f"  [{ob.kind.value.upper()}] {ob.statement}")
        if ob.function_name:
            print(f"               For: {ob.function_name}")
        print()


def test_auto_detection():
    """Test automatic spec format detection."""
    print("=" * 80)
    print("TEST 4: AUTO-DETECTION OF SPEC FORMAT")
    print("=" * 80)
    print()

    # Test auto-detection for Python decorators
    print("Auto-detecting Python decorator format...")
    ob_set1 = parse_specifications(PYTHON_DECORATOR_SAMPLE)
    print(f"✓ Detected: {ob_set1.total_obligations} obligations")
    print()

    # Test auto-detection for Lean
    print("Auto-detecting Lean format...")
    ob_set2 = parse_specifications(LEAN_SPEC_SAMPLE)
    print(f"✓ Detected: {ob_set2.total_obligations} obligations")
    print()


def test_serialization():
    """Test serialization of obligation sets."""
    print("=" * 80)
    print("TEST 5: SERIALIZATION")
    print("=" * 80)
    print()

    obligation_set = extract_from_decorators(
        PYTHON_DECORATOR_SAMPLE, "model.py")

    print("Converting to JSON-ready dictionary...")
    ob_dict = obligation_set.to_dict()

    print(f"✓ Serialized successfully!")
    print(f"  Top-level keys: {list(ob_dict.keys())}")
    print(f"  Total obligations: {ob_dict['total_obligations']}")
    print(f"  Summary: {ob_dict['summary']}")
    print()

    # Show first obligation structure
    if obligation_set.obligations:
        print("Sample obligation structure:")
        print(obligation_set.obligations[0].to_dict())


def run_all_tests():
    """Run all specification ingestion tests."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AXIOM ZERO - SPEC INGESTION TESTS" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    test_python_decorators()
    print("\n" + "━" * 80 + "\n")

    test_python_inline()
    print("\n" + "━" * 80 + "\n")

    test_lean_specs()
    print("\n" + "━" * 80 + "\n")

    test_auto_detection()
    print("\n" + "━" * 80 + "\n")

    test_serialization()

    print()
    print("=" * 80)
    print("✓ ALL SPEC INGESTION TESTS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Python decorator parsing")
    print("  ✓ Inline specification extraction")
    print("  ✓ Lean 4 theorem parsing")
    print("  ✓ Auto-detection of spec format")
    print("  ✓ Serialization to JSON-ready format")
    print()
    print("These proof obligations feed into the RL Proof Agent as goals to prove!")
    print()


if __name__ == "__main__":
    run_all_tests()
