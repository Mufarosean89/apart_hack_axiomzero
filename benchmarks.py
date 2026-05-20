"""
Axiom Zero - Benchmark Suite
Simple, well-specified problems for training and evaluation.
Expanded to 50+ problems across 5 difficulty levels.
"""

BENCHMARKS = [
    # Level 1: Basic Arithmetic (15 problems)
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
    
    {
        "id": "add_zero",
        "name": "Addition with Zero",
        "level": 1,
        "code": """
def add_zero(n: int) -> int:
    return n + 0
""",
        "spec": {
            "requires": [],
            "ensures": ["result == n"],
            "invariants": []
        },
        "expected_lean": "theorem add_zero (n : ℕ) : n + 0 = n",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "mul_zero",
        "name": "Multiplication by Zero",
        "level": 1,
        "code": """
def mul_zero(n: int) -> int:
    return n * 0
""",
        "spec": {
            "requires": [],
            "ensures": ["result == 0"],
            "invariants": []
        },
        "expected_lean": "theorem mul_zero (n : ℕ) : n * 0 = 0",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "add_assoc",
        "name": "Addition Associativity",
        "level": 1,
        "code": """
def add_assoc(a: int, b: int, c: int) -> int:
    return (a + b) + c
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a + (b + c)"],
            "invariants": []
        },
        "expected_lean": "theorem add_assoc (a b c : ℕ) : (a + b) + c = a + (b + c)",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "mul_assoc",
        "name": "Multiplication Associativity",
        "level": 1,
        "code": """
def mul_assoc(a: int, b: int, c: int) -> int:
    return (a * b) * c
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a * (b * c)"],
            "invariants": []
        },
        "expected_lean": "theorem mul_assoc (a b c : ℕ) : (a * b) * c = a * (b * c)",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "mul_comm",
        "name": "Multiplication Commutativity",
        "level": 1,
        "code": """
def mul_comm(a: int, b: int) -> int:
    return a * b
""",
        "spec": {
            "requires": [],
            "ensures": ["result == b * a"],
            "invariants": []
        },
        "expected_lean": "theorem mul_comm (a b : ℕ) : a * b = b * a",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "add_mul_distrib",
        "name": "Distributivity (Left)",
        "level": 1,
        "code": """
def distrib_left(a: int, b: int, c: int) -> int:
    return a * (b + c)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a * b + a * c"],
            "invariants": []
        },
        "expected_lean": "theorem distrib_left (a b c : ℕ) : a * (b + c) = a * b + a * c",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "mul_add_distrib",
        "name": "Distributivity (Right)",
        "level": 1,
        "code": """
def distrib_right(a: int, b: int, c: int) -> int:
    return (a + b) * c
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a * c + b * c"],
            "invariants": []
        },
        "expected_lean": "theorem distrib_right (a b c : ℕ) : (a + b) * c = a * c + b * c",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "square_expand",
        "name": "Square Expansion",
        "level": 1,
        "code": """
def square_expand(a: int, b: int) -> int:
    return (a + b) * (a + b)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == a*a + 2*a*b + b*b"],
            "invariants": []
        },
        "expected_lean": "theorem square_expand (a b : ℕ) : (a + b)^2 = a^2 + 2*a*b + b^2",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "diff_of_squares",
        "name": "Difference of Squares",
        "level": 1,
        "code": """
def diff_squares(a: int, b: int) -> int:
    return (a + b) * (a - b)
""",
        "spec": {
            "requires": ["a >= b"],
            "ensures": ["result == a*a - b*b"],
            "invariants": []
        },
        "expected_lean": "theorem diff_squares (a b : ℕ) (h : a ≥ b) : (a + b) * (a - b) = a^2 - b^2",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "add_self",
        "name": "Add Self (Double)",
        "level": 1,
        "code": """
def double(n: int) -> int:
    return n + n
""",
        "spec": {
            "requires": [],
            "ensures": ["result == 2 * n"],
            "invariants": []
        },
        "expected_lean": "theorem double (n : ℕ) : n + n = 2 * n",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "mul_two",
        "name": "Multiply by Two",
        "level": 1,
        "code": """
def mul_two(n: int) -> int:
    return 2 * n
""",
        "spec": {
            "requires": [],
            "ensures": ["result == n + n"],
            "invariants": []
        },
        "expected_lean": "theorem mul_two (n : ℕ) : 2 * n = n + n",
        "difficulty": "easy",
        "expected_tactic": "ring"
    },
    
    {
        "id": "succ_add",
        "name": "Successor Addition",
        "level": 1,
        "code": """
def succ_add(n: int) -> int:
    return n + 1
""",
        "spec": {
            "requires": [],
            "ensures": ["result > n"],
            "invariants": []
        },
        "expected_lean": "theorem succ_add (n : ℕ) : n + 1 > n",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "zero_add",
        "name": "Zero Plus N",
        "level": 1,
        "code": """
def zero_add(n: int) -> int:
    return 0 + n
""",
        "spec": {
            "requires": [],
            "ensures": ["result == n"],
            "invariants": []
        },
        "expected_lean": "theorem zero_add (n : ℕ) : 0 + n = n",
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

# Import additional benchmarks
try:
    from benchmarks_additional import ADDITIONAL_BENCHMARKS
    BENCHMARKS.extend(ADDITIONAL_BENCHMARKS)
except ImportError:
    pass  # Additional benchmarks not available
