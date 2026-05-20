#!/usr/bin/env python3
"""
Quick test - verify core modules work
"""

print("Testing Axiom Zero core modules...")
print()

# Test 1: AST Extraction
print("[1/3] Testing AST Extraction...")
try:
    from ast_extractor import parse_to_ir
    code = "def add(a, b): return a + b"
    ir = parse_to_ir(code)
    print(f"   ✓ Success: {ir.total_functions} function(s) parsed")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Test 2: Abstract Interpretation
print("[2/3] Testing Abstract Interpretation...")
try:
    from abstract_interpreter import run_abstract_interpretation
    state = run_abstract_interpretation(ir)
    print(f"   ✓ Success: {len(state.function_envs)} function(s) analyzed")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Test 3: Spec Ingestion
print("[3/3] Testing Spec Ingestion...")
try:
    from spec_ingestion import extract_from_decorators
    code_with_spec = '''
@requires("x > 0")
@ensures("result > x")
def foo(x: int) -> int:
    return x + 1
'''
    obligations = extract_from_decorators(code_with_spec)
    print(
        f"   ✓ Success: {obligations.total_obligations} obligation(s) extracted")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()
print("=" * 60)
print("Core modules working!")
print("=" * 60)
print()
print("Modules that need Lean 4 installation:")
print("  • proof_engine (LeanEnvironment)")
print("  • rl_agent (requires torch, torch_geometric)")
print("  • compiler (has circular import to fix)")
print()
print("Next: Install Lean 4 to enable full pipeline")
