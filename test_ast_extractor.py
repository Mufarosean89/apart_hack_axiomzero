"""
Test script for AST extraction module.
Demonstrates parsing Python/PyTorch code into normalized IR.
"""

from ast_extractor import parse_to_ir, extract_ast
from ast_extractor.parser import PythonASTParser


# Sample Python/PyTorch code for testing
SAMPLE_CODE = '''
import torch
import torch.nn as nn
from typing import List

class SimpleNetwork(nn.Module):
    """A simple neural network for demonstration."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with shape validation."""
    # Validate shapes
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible shapes")
    
    result = torch.matmul(a, b)
    return result

def compute_statistics(data: List[float]) -> dict:
    """Compute basic statistics."""
    if len(data) == 0:
        return {'mean': 0.0, 'sum': 0.0}
    
    total = 0.0
    for value in data:
        total = total + value
    
    mean = total / len(data)
    
    return {
        'mean': mean,
        'sum': total,
        'count': len(data)
    }

def find_maximum(numbers: List[int]) -> int:
    """Find maximum using loop."""
    if len(numbers) == 0:
        return 0
    
    max_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    
    return max_val
'''


def test_ast_extraction():
    """Test AST extraction and normalization."""
    print("=" * 80)
    print("AXIOM ZERO - AST EXTRACTION TEST")
    print("=" * 80)
    print()

    # Parse to normalized IR
    print("Parsing Python/PyTorch code into normalized IR...")
    ir = parse_to_ir(SAMPLE_CODE, use_tree_sitter=False)

    print(f"\n✓ Successfully parsed!")
    print(f"  Source file: {ir.source_file}")
    print(f"  Total functions: {ir.total_functions}")
    print(f"  Total loops: {ir.total_loops}")
    print(f"  Total conditionals: {ir.total_conditionals}")
    print(f"  Total tensor operations: {ir.total_tensor_ops}")

    print("\n" + "=" * 80)
    print("IMPORTS")
    print("=" * 80)
    for imp in ir.imports:
        print(f"  • {imp}")

    print("\n" + "=" * 80)
    print("CLASSES")
    print("=" * 80)
    for cls in ir.classes:
        print(f"\n  Class: {cls.name}")
        print(f"    Base classes: {cls.base_classes}")
        print(f"    Methods: {len(cls.methods)}")
        for method in cls.methods:
            print(f"      - {method.signature.name}()")

    print("\n" + "=" * 80)
    print("FUNCTIONS")
    print("=" * 80)
    for func in ir.functions:
        print(f"\n  Function: {func.signature.name}")
        print(f"    Line: {func.line_number}")
        print(f"    Parameters: {len(func.signature.parameters)}")
        for param in func.signature.parameters:
            type_str = param['type'].to_string(
            ) if param['type'] else 'untyped'
            print(f"      - {param['name']}: {type_str}")

        if func.signature.return_type:
            print(f"    Return type: {func.signature.return_type.to_string()}")

        if func.tensor_operations:
            print(
                f"    Tensor operations: {[op.value for op in func.tensor_operations]}")

        # Show body structure
        print(f"    Body statements: {len(func.body)}")
        for i, stmt in enumerate(func.body[:3]):  # Show first 3 statements
            if hasattr(stmt, 'stmt_type'):
                print(f"      [{i}] {stmt.stmt_type}")
            elif hasattr(stmt, 'loop_type'):
                print(f"      [{i}] {stmt.loop_type} loop")
            elif hasattr(stmt, 'condition'):
                print(f"      [{i}] conditional")

    print("\n" + "=" * 80)
    print("IR SERIALIZATION (JSON-ready)")
    print("=" * 80)
    ir_dict = ir.to_dict()
    print(f"  Serialized to dictionary with {len(ir_dict)} top-level keys")
    print(f"  Keys: {list(ir_dict.keys())}")

    print("\n" + "=" * 80)
    print("DETAILED AST DUMP (first function)")
    print("=" * 80)
    ast_tree = extract_ast(SAMPLE_CODE)
    first_func = None
    for node in ast_tree.body:
        if isinstance(node, __import__('ast').FunctionDef):
            first_func = node
            break

    if first_func:
        dump = PythonASTParser.dump_ast(first_func)
        print(dump[:500] + "..." if len(dump) > 500 else dump)

    print("\n" + "=" * 80)
    print("✓ AST EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_ast_extraction()
