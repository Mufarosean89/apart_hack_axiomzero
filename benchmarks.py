"""
Axiom Zero - Benchmark Suite
Simple, well-specified problems for training and evaluation.
"""

BENCHMARKS = [
    # Level 1: Basic Arithmetic
    {
        "id": "add_comm",
        "name": "Addition Commutativity",
        "level": 1,
        "code": """
def add(a: int, b: int) -> int:
    return a + b
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a + b"],
            "invariants": []
        },
        "expected_lean": "theorem add_comm (a b : ℕ) : a + b = b + a",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "mul_one",
        "name": "Multiplication by One",
        "level": 1,
        "code": """
def mul_one(n: int) -> int:
    return n * 1
""",
        "spec": {
            "requires": [],
            "ensures": ["result == n"],
            "invariants": []
        },
        "expected_lean": "theorem mul_one (n : ℕ) : n * 1 = n",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    # Level 2: List Operations
    {
        "id": "list_append_nil",
        "name": "List Append with Empty",
        "level": 2,
        "code": """
from typing import List
def append_nil(xs: List[int]) -> List[int]:
    return xs + []
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem list_append_nil (xs : List ℕ) : xs ++ [] = xs",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_length_append",
        "name": "Length of Appended Lists",
        "level": 2,
        "code": """
from typing import List
def append_length(xs: List[int], ys: List[int]) -> int:
    return len(xs + ys)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == len(xs) + len(ys)"],
            "invariants": []
        },
        "expected_lean": "theorem length_append (xs ys : List ℕ) : length (xs ++ ys) = length xs + length ys",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    # Level 3: Loops
    {
        "id": "sum_formula",
        "name": "Sum of First N Numbers",
        "level": 3,
        "code": """
def sum_first_n(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i
    return total
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result == n * (n - 1) / 2"],
            "invariants": ["total == i * (i - 1) / 2"]
        },
        "expected_lean": "theorem sum_formula (n : ℕ) : ∑ i in range n, i = n * (n - 1) / 2",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "factorial",
        "name": "Factorial Function",
        "level": 3,
        "code": """
def factorial(n: int) -> int:
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result > 0"],
            "invariants": ["result > 0"]
        },
        "expected_lean": "def factorial (n : ℕ) : ℕ",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    # Level 4: Conditionals
    {
        "id": "max_correct",
        "name": "Maximum Function",
        "level": 4,
        "code": """
def max_val(a: int, b: int) -> int:
    if a > b:
        return a
    else:
        return b
""",
        "spec": {
            "requires": [],
            "ensures": ["result >= a", "result >= b"],
            "invariants": []
        },
        "expected_lean": "theorem max_spec (a b : ℕ) : max a b ≥ a ∧ max a b ≥ b",
        "difficulty": "medium",
        "expected_tactic": "split_ifs"
    },
    
    {
        "id": "abs_value",
        "name": "Absolute Value",
        "level": 4,
        "code": """
def abs_val(x: int) -> int:
    if x >= 0:
        return x
    else:
        return -x
""",
        "spec": {
            "requires": [],
            "ensures": ["result >= 0"],
            "invariants": []
        },
        "expected_lean": "theorem abs_nonneg (x : ℤ) : |x| ≥ 0",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    # Level 5: PyTorch/Tensor Operations
    {
        "id": "tensor_add",
        "name": "Tensor Addition",
        "level": 5,
        "code": """
import torch
def tensor_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B
""",
        "spec": {
            "requires": ["A.shape == B.shape"],
            "ensures": ["result.shape == A.shape"],
            "invariants": []
        },
        "expected_lean": "theorem tensor_add_shape (A B : Matrix) : shape (A + B) = shape A",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "matrix_vec_mul",
        "name": "Matrix-Vector Multiplication",
        "level": 5,
        "code": """
import torch
def mat_vec_mul(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(M, v)
""",
        "spec": {
            "requires": ["M.shape[1] == v.shape[0]"],
            "ensures": ["result.shape[0] == M.shape[0]"],
            "invariants": []
        },
        "expected_lean": "theorem mat_vec_mul_shape (M : Matrix) (v : Vector) : shape (M * v) = shape M",
        "difficulty": "hard",
        "expected_tactic": "MCTS"
    }
]


def get_benchmarks_by_level(level: int = None):
    """Get benchmarks filtered by level."""
    if level is None:
        return BENCHMARKS
    return [b for b in BENCHMARKS if b["level"] == level]


def get_benchmark_by_id(benchmark_id: str):
    """Get specific benchmark by ID."""
    for b in BENCHMARKS:
        if b["id"] == benchmark_id:
            return b
    return None


def print_benchmark_summary():
    """Print summary of all benchmarks."""
    print("=" * 70)
    print("AXIOM ZERO - BENCHMARK SUITE")
    print("=" * 70)
    print()
    
    levels = {1: "Basic Arithmetic", 2: "List Operations", 3: "Loops", 
              4: "Conditionals", 5: "PyTorch/Tensors"}
    
    for level in range(1, 6):
        benchmarks = get_benchmarks_by_level(level)
        if benchmarks:
            print(f"Level {level}: {levels[level]}")
            print("-" * 70)
            for b in benchmarks:
                print(f"  {b['id']:25s} - {b['name']:30s} [{b['difficulty']}]")
            print()
    
    print(f"Total benchmarks: {len(BENCHMARKS)}")
    print()


if __name__ == "__main__":
    print_benchmark_summary()
