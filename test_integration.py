"""
Integration test showing the complete pipeline:
AST Extraction → Abstract Interpretation → Spec Ingestion
Demonstrates how all modules work together for Axiom Zero.
"""

from ast_extractor import parse_to_ir
from abstract_interpreter import run_abstract_interpretation
from spec_ingestion import extract_from_decorators


# Complete example: PyTorch code with specifications
COMPLETE_EXAMPLE = '''
import torch
from typing import List

@requires("batch_size > 0")
@requires("seq_len > 0")
@requires("hidden_dim > 0")
@ensures("output.shape[0] == batch_size")
@ensures("output.shape[1] == seq_len")
@ensures("output.shape[2] == hidden_dim")
def create_input_tensor(batch_size: int, seq_len: int, 
                       hidden_dim: int) -> torch.Tensor:
    """Create input tensor for transformer."""
    return torch.zeros(batch_size, seq_len, hidden_dim)

@requires("Q.shape == K.shape")
@requires("K.shape[0] == V.shape[0]")
@requires("Q.shape[2] == K.shape[2]")
@ensures("output.shape == Q.shape")
@invariant("attention_weights.sum(dim=-1) == 1.0")
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, 
                                 V: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention mechanism."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

@requires("x.shape[-1] == weight.shape[1]")
@ensures("output.shape[:-1] == x.shape[:-1]")
@ensures("output.shape[-1] == weight.shape[0]")
def linear_layer(x: torch.Tensor, weight: torch.Tensor, 
                bias: torch.Tensor) -> torch.Tensor:
    """Linear transformation layer."""
    output = torch.matmul(x, weight.transpose(0, 1))
    output = torch.add(output, bias)
    return output

@requires("len(sequence) > 0")
@ensures("result >= 0")
@ensures("result <= len(sequence)")
def find_max_position(sequence: List[float]) -> int:
    """Find position of maximum value."""
    max_val = sequence[0]
    max_pos = 0
    
    for i in range(1, len(sequence)):
        if sequence[i] > max_val:
            max_val = sequence[i]
            max_pos = i
    
    return max_pos
'''


def run_complete_pipeline():
    """Run the complete Axiom Zero pipeline."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "AXIOM ZERO - COMPLETE INTEGRATION TEST" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # =========================================================================
    # PHASE 1: AST EXTRACTION
    # =========================================================================
    print("=" * 80)
    print("PHASE 1: AST EXTRACTION")
    print("=" * 80)
    print()
    print("Parsing Python/PyTorch source code into normalized IR...")
    print()

    ir = parse_to_ir(COMPLETE_EXAMPLE)
    ir.source_file = "transformer.py"

    print("✓ AST Extraction Complete")
    print(f"  • Functions extracted: {ir.total_functions}")
    print(f"  • Tensor operations detected: {ir.total_tensor_ops}")
    print(f"  • Loops identified: {ir.total_loops}")
    print(f"  • Conditionals found: {ir.total_conditionals}")
    print()

    # Show function signatures
    print("Extracted Function Signatures:")
    print("-" * 60)
    for func in ir.functions:
        print(f"  {func.signature.name}(")
        for param in func.signature.parameters:
            type_str = param['type'].to_string(
            ) if param['type'] else 'untyped'
            print(f"    {param['name']}: {type_str}")
        ret_type = func.signature.return_type.to_string(
        ) if func.signature.return_type else 'untyped'
        print(f"  ) -> {ret_type}")
        if func.tensor_operations:
            print(
                f"    Tensor ops: {[op.value for op in func.tensor_operations]}")
        print()

    # =========================================================================
    # PHASE 2: ABSTRACT INTERPRETATION
    # =========================================================================
    print("=" * 80)
    print("PHASE 2: ABSTRACT INTERPRETATION")
    print("=" * 80)
    print()
    print("Running symbolic type inference and shape analysis...")
    print()

    state = run_abstract_interpretation(ir)

    print("✓ Abstract Interpretation Complete")
    print(f"  • Functions analyzed: {len(state.function_envs)}")
    print(f"  • Shape facts extracted: {len(state.shape_facts)}")
    print(f"  • Type constraints: {len(state.type_constraints)}")
    print(f"  • Analysis warnings: {len(state.warnings)}")
    print()

    # Show shape facts
    if state.shape_facts:
        print("Extracted Shape Facts (for proof state):")
        print("-" * 60)
        for i, fact in enumerate(state.shape_facts, 1):
            print(f"  {i:2d}. {fact}")
        print()

    # Show type constraints
    if state.type_constraints:
        print("Type Constraints:")
        print("-" * 60)
        for i, constraint in enumerate(state.type_constraints, 1):
            print(f"  {i:2d}. {constraint}")
        print()

    # Show variable types for one function
    if 'linear_layer' in state.function_envs:
        print("Inferred Variable Types (linear_layer):")
        print("-" * 60)
        env = state.function_envs['linear_layer']
        for var_name, abs_value in env.items():
            print(f"  {var_name:15s} → {abs_value}")
        print()

    # =========================================================================
    # PHASE 3: SPEC INGESTION
    # =========================================================================
    print("=" * 80)
    print("PHASE 3: SPEC INGESTION")
    print("=" * 80)
    print()
    print("Parsing formal specifications into proof obligations...")
    print()

    obligations = extract_from_decorators(COMPLETE_EXAMPLE, "transformer.py")

    print("✓ Spec Ingestion Complete")
    print(f"  • Total obligations: {obligations.total_obligations}")
    print(f"  • Critical (priority 1): {obligations.critical_obligations}")
    print(f"  • Preconditions: {len(obligations.preconditions)}")
    print(f"  • Postconditions: {len(obligations.postconditions)}")
    print(f"  • Invariants: {len(obligations.invariants)}")
    print()

    # Show obligations by function
    print("Proof Obligations by Function:")
    print("-" * 60)
    for func_name, obs in obligations.by_function.items():
        print(f"\n  Function: {func_name} ({len(obs)} obligations)")
        for ob in obs:
            kind_icon = {
                'precondition': '',
                'postcondition': '✓',
                'invariant': ''
            }.get(ob.kind.value, '•')
            print(
                f"    {kind_icon} [{ob.kind.value.upper():15s}] {ob.statement}")
    print()

    # =========================================================================
    # INTEGRATION SUMMARY
    # =========================================================================
    print("=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    print()
    print("Pipeline Results:")
    print("-" * 60)
    print(f"  ✓ AST Extraction:        {ir.total_functions} functions parsed")
    print(
        f"  ✓ Abstract Interpretation: {len(state.function_envs)} functions analyzed")
    print(
        f"  ✓ Spec Ingestion:        {obligations.total_obligations} obligations extracted")
    print()

    print("Data Flow:")
    print("-" * 60)
    print("  Python Source → Normalized IR → Abstract State → Proof Obligations")
    print("       ↓              ↓              ↓                ↓")
    print("    Parsing    Type/Shape      Background       Goals for RL")
    print("               Inference       Facts             Proof Agent")
    print()

    # Show how data flows between phases
    print("Cross-Phase Integration:")
    print("-" * 60)
    print("  1. IR function signatures → Type inference initialization")
    print("  2. Tensor op detection → Shape analysis rules")
    print("  3. Shape facts → Preconditions for proof")
    print("  4. Type constraints → Proof state background")
    print("  5. Spec obligations → RL agent goals")
    print()

    # Serialization demo
    print("Serialization Ready:")
    print("-" * 60)
    print("  • IR → dict:", list(ir.to_dict().keys()))
    print("  • AbstractState → dict:", list(state.to_dict().keys()))
    print("  • ObligationSet → dict:", list(obligations.to_dict().keys()))
    print()

    print("=" * 80)
    print("✓ COMPLETE PIPELINE SUCCESSFUL")
    print("=" * 80)
    print()
    print("Next Step: Feed this data into the RL Proof Agent!")
    print("  - Proof obligations become goals to prove")
    print("  - Abstract state provides background facts")
    print("  - IR structure guides tactic selection")
    print()


if __name__ == "__main__":
    run_complete_pipeline()
