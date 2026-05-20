"""
Test script for abstract interpreter module.
Demonstrates symbolic type inference and tensor shape analysis.
"""

from ast_extractor import parse_to_ir
from abstract_interpreter import run_abstract_interpretation


# Sample PyTorch code with tensor operations
PYTORCH_SAMPLE = '''
import torch
import torch.nn as nn
from typing import List

def matrix_chain_multiply(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Chain of matrix multiplications."""
    # A: [M, K], B: [K, N], C: [N, P]
    AB = torch.matmul(A, B)  # [M, N]
    ABC = torch.matmul(AB, C)  # [M, P]
    return ABC

def transformer_block(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, 
                     W_v: torch.Tensor, W_o: torch.Tensor) -> torch.Tensor:
    """Simplified transformer block."""
    # x: [B, T, D] where B=batch, T=sequence, D=dimension
    # Weight matrices: [D, D]
    
    Q = torch.matmul(x, W_q)  # [B, T, D]
    K = torch.matmul(x, W_k)  # [B, T, D]
    V = torch.matmul(x, W_v)  # [B, T, D]
    
    # Scaled dot-product attention (simplified)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, T, T]
    attention = torch.softmax(scores, dim=-1)  # [B, T, T]
    output = torch.matmul(attention, V)  # [B, T, D]
    
    # Output projection
    result = torch.matmul(output, W_o)  # [B, T, D]
    return result

def elementwise_operations(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Elementwise operations with broadcasting."""
    # a: [B, D], b: [D], c: [1, D]
    result1 = torch.add(a, b)  # [B, D] - broadcasting
    result2 = torch.mul(result1, c)  # [B, D] - broadcasting
    return result2

def compute_with_loops(data: List[float]) -> float:
    """Compute sum with explicit loop."""
    total = 0.0
    for i in range(len(data)):
        total = total + data[i]
    return total

def conditional_tensor_op(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Conditional tensor operation."""
    if x.shape[0] > 10:
        result = torch.matmul(x, x.transpose(0, 1))
    else:
        result = torch.add(x, x)
    return result
'''


def test_abstract_interpretation():
    """Test abstract interpretation on PyTorch code."""
    print("=" * 80)
    print("AXIOM ZERO - ABSTRACT INTERPRETATION TEST")
    print("=" * 80)
    print()

    # Step 1: Parse to IR
    print("Step 1: Parsing PyTorch code into normalized IR...")
    ir = parse_to_ir(PYTORCH_SAMPLE)
    print(f"✓ Parsed successfully")
    print(f"  Functions: {ir.total_functions}")
    print(f"  Tensor operations: {ir.total_tensor_ops}")
    print()

    # Step 2: Run abstract interpretation
    print("Step 2: Running abstract interpretation...")
    print("  - Type inference")
    print("  - Shape analysis")
    print("  - Data flow tracking")
    print()

    state = run_abstract_interpretation(ir)

    print("✓ Abstract interpretation complete!")
    print()

    # Display results
    print("=" * 80)
    print("FUNCTION SIGNATURES (WITH INFERRED TYPES)")
    print("=" * 80)
    for func_name, sig_info in state.function_signatures.items():
        print(f"\n  Function: {sig_info['name']}")
        print(f"    Has tensor ops: {sig_info['has_tensor_ops']}")
        print(f"    Parameters:")
        for param in sig_info['parameters']:
            type_str = param.get('type', param.get('inferred_type', 'unknown'))
            print(f"      - {param['name']}: {type_str}")
        if sig_info['return_type']:
            print(f"    Returns: {sig_info['return_type']}")

    print("\n" + "=" * 80)
    print("SHAPE FACTS (Background for Proof State)")
    print("=" * 80)
    if state.shape_facts:
        for i, fact in enumerate(state.shape_facts, 1):
            print(f"  {i:2d}. {fact}")
    else:
        print("  (No shape facts extracted)")

    print("\n" + "=" * 80)
    print("TYPE CONSTRAINTS")
    print("=" * 80)
    if state.type_constraints:
        for i, constraint in enumerate(state.type_constraints, 1):
            print(f"  {i:2d}. {constraint}")
    else:
        print("  (No type constraints)")

    print("\n" + "=" * 80)
    print("TENSOR OPERATIONS METADATA")
    print("=" * 80)
    if state.tensor_ops_metadata:
        for op_name, metadata in state.tensor_ops_metadata.items():
            print(f"\n  Operation: {op_name}")
            for key, value in metadata.items():
                print(f"    {key}: {value}")
    else:
        print("  (No tensor operation metadata)")

    print("\n" + "=" * 80)
    print("ANALYSIS WARNINGS")
    print("=" * 80)
    if state.warnings:
        for warning in state.warnings:
            print(f"  ⚠ {warning}")
    else:
        print("  ✓ No warnings")

    print("\n" + "=" * 80)
    print("LOCAL VARIABLE TYPES (Example: transformer_block)")
    print("=" * 80)
    if 'transformer_block' in state.function_envs:
        env = state.function_envs['transformer_block']
        print(f"\n  Variables in transformer_block:")
        for var_name, abs_value in env.items():
            print(f"    {var_name:15s} → {abs_value}")

    print("\n" + "=" * 80)
    print("SERIALIZATION READY")
    print("=" * 80)
    state_dict = state.to_dict()
    print(f"  AbstractState serialized to dict with {len(state_dict)} keys:")
    for key in state_dict.keys():
        print(f"    - {key}")

    print("\n" + "=" * 80)
    print("✓ ABSTRACT INTERPRETATION TEST COMPLETE")
    print("=" * 80)
    print()
    print("Next Step: Feed these facts into the RL Proof Agent")
    print("  - Shape facts become preconditions")
    print("  - Type constraints guide tactic selection")
    print("  - Data flow graph enables proof automation")
    print()


if __name__ == "__main__":
    test_abstract_interpretation()
