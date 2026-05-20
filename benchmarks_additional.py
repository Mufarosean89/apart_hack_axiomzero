"""
Additional benchmarks for Axiom Zero
Expands from 23 to 55 total benchmarks
"""

ADDITIONAL_BENCHMARKS = [
    # Level 2: List Operations (10 problems)
    {
        "id": "list_append_nil_right",
        "name": "List Append Empty (Right)",
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
        "expected_lean": "theorem append_nil (xs : List ℕ) : xs ++ [] = xs",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_nil_append",
        "name": "Empty List Append (Left)",
        "level": 2,
        "code": """
from typing import List
def nil_append(xs: List[int]) -> List[int]:
    return [] + xs
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem nil_append (xs : List ℕ) : [] ++ xs = xs",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_append_assoc",
        "name": "List Append Associativity",
        "level": 2,
        "code": """
from typing import List
def append_assoc(xs: List[int], ys: List[int], zs: List[int]) -> List[int]:
    return (xs + ys) + zs
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs + (ys + zs)"],
            "invariants": []
        },
        "expected_lean": "theorem append_assoc (xs ys zs : List ℕ) : (xs ++ ys) ++ zs = xs ++ (ys ++ zs)",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "list_length_nil",
        "name": "Length of Empty List",
        "level": 2,
        "code": """
from typing import List
def length_nil() -> int:
    xs: List[int] = []
    return len(xs)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == 0"],
            "invariants": []
        },
        "expected_lean": "theorem length_nil : length ([] : List ℕ) = 0",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_length_cons",
        "name": "Length of Cons List",
        "level": 2,
        "code": """
from typing import List
def length_cons(x: int, xs: List[int]) -> int:
    return len([x] + xs)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == 1 + len(xs)"],
            "invariants": []
        },
        "expected_lean": "theorem length_cons (x : ℕ) (xs : List ℕ) : length (x :: xs) = 1 + length xs",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_reverse_nil",
        "name": "Reverse of Empty List",
        "level": 2,
        "code": """
from typing import List
def reverse_nil() -> List[int]:
    xs: List[int] = []
    return xs[::-1]
""",
        "spec": {
            "requires": [],
            "ensures": ["result == []"],
            "invariants": []
        },
        "expected_lean": "theorem reverse_nil : reverse ([] : List ℕ) = []",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_reverse_singleton",
        "name": "Reverse of Singleton",
        "level": 2,
        "code": """
from typing import List
def reverse_singleton(x: int) -> List[int]:
    return [x][::-1]
""",
        "spec": {
            "requires": [],
            "ensures": ["result == [x]"],
            "invariants": []
        },
        "expected_lean": "theorem reverse_singleton (x : ℕ) : reverse [x] = [x]",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "list_map_id",
        "name": "Map Identity Function",
        "level": 2,
        "code": """
from typing import List
def map_id(xs: List[int]) -> List[int]:
    return [x for x in xs]
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem map_id (xs : List ℕ) : map id xs = xs",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "list_filter_true",
        "name": "Filter with True Predicate",
        "level": 2,
        "code": """
from typing import List
def filter_true(xs: List[int]) -> List[int]:
    return [x for x in xs if True]
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem filter_true (xs : List ℕ) : filter (λ x, true) xs = xs",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "list_concat_nil",
        "name": "Concat with Empty",
        "level": 2,
        "code": """
from typing import List
def concat_nil(xs: List[int]) -> List[int]:
    return xs + []
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem concat_nil (xs : List ℕ) : xs ++ [] = xs",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    # Level 3: Loops (10 problems)
    {
        "id": "sum_range_n",
        "name": "Sum of Range",
        "level": 3,
        "code": """
def sum_range(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result == n * (n - 1) // 2"],
            "invariants": ["total >= 0"]
        },
        "expected_lean": "theorem sum_range (n : ℕ) : ∑ i in range n, i = n * (n - 1) / 2",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "factorial_recursive",
        "name": "Factorial (Recursive)",
        "level": 3,
        "code": """
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result > 0"],
            "invariants": []
        },
        "expected_lean": "def factorial : ℕ → ℕ",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "power_of_two",
        "name": "Power of Two",
        "level": 3,
        "code": """
def power_of_two(n: int) -> int:
    result = 1
    for i in range(n):
        result *= 2
    return result
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result == 2^n"],
            "invariants": ["result > 0"]
        },
        "expected_lean": "theorem power_of_two (n : ℕ) : 2^n > 0",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "gcd_euclidean",
        "name": "GCD (Euclidean Algorithm)",
        "level": 3,
        "code": """
def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
""",
        "spec": {
            "requires": ["a > 0", "b > 0"],
            "ensures": ["result > 0"],
            "invariants": ["a > 0"]
        },
        "expected_lean": "theorem gcd_pos (a b : ℕ) (ha : a > 0) (hb : b > 0) : gcd a b > 0",
        "difficulty": "hard",
        "expected_tactic": "MCTS"
    },
    
    {
        "id": "fibonacci_loop",
        "name": "Fibonacci (Iterative)",
        "level": 3,
        "code": """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result >= 0"],
            "invariants": ["a >= 0", "b >= 0"]
        },
        "expected_lean": "theorem fib_nonneg (n : ℕ) : fib n ≥ 0",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "product_list",
        "name": "Product of List",
        "level": 3,
        "code": """
from typing import List
def product(xs: List[int]) -> int:
    result = 1
    for x in xs:
        result *= x
    return result
""",
        "spec": {
            "requires": [],
            "ensures": ["result >= 1"],
            "invariants": ["result >= 1"]
        },
        "expected_lean": "theorem product_ge_one (xs : List ℕ) : prod xs ≥ 1",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "max_element",
        "name": "Maximum Element in List",
        "level": 3,
        "code": """
from typing import List
def find_max(xs: List[int]) -> int:
    max_val = xs[0]
    for x in xs[1:]:
        if x > max_val:
            max_val = x
    return max_val
""",
        "spec": {
            "requires": ["len(xs) > 0"],
            "ensures": ["result >= x for x in xs"],
            "invariants": []
        },
        "expected_lean": "theorem max_ge_elements (xs : List ℕ) (h : xs ≠ []) : maximum xs ≥ x",
        "difficulty": "hard",
        "expected_tactic": "MCTS"
    },
    
    {
        "id": "count_elements",
        "name": "Count Elements",
        "level": 3,
        "code": """
from typing import List
def count(xs: List[int]) -> int:
    c = 0
    for x in xs:
        c += 1
    return c
""",
        "spec": {
            "requires": [],
            "ensures": ["result == len(xs)"],
            "invariants": ["c >= 0"]
        },
        "expected_lean": "theorem count_equals_length (xs : List ℕ) : count xs = length xs",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "reverse_twice",
        "name": "Reverse Twice is Identity",
        "level": 3,
        "code": """
from typing import List
def reverse_twice(xs: List[int]) -> List[int]:
    return xs[::-1][::-1]
""",
        "spec": {
            "requires": [],
            "ensures": ["result == xs"],
            "invariants": []
        },
        "expected_lean": "theorem reverse_reverse (xs : List ℕ) : reverse (reverse xs) = xs",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    {
        "id": "sum_const",
        "name": "Sum of Constant",
        "level": 3,
        "code": """
def sum_const(n: int, c: int) -> int:
    total = 0
    for i in range(n):
        total += c
    return total
""",
        "spec": {
            "requires": ["n >= 0"],
            "ensures": ["result == n * c"],
            "invariants": []
        },
        "expected_lean": "theorem sum_const (n c : ℕ) : ∑ i in range n, c = n * c",
        "difficulty": "medium",
        "expected_tactic": "induction"
    },
    
    # Level 4: Conditionals (10 problems)
    {
        "id": "max_symmetric",
        "name": "Max is Symmetric",
        "level": 4,
        "code": """
def max_sym(a: int, b: int) -> int:
    return a if a > b else b
""",
        "spec": {
            "requires": [],
            "ensures": ["result == max(b, a)"],
            "invariants": []
        },
        "expected_lean": "theorem max_comm (a b : ℕ) : max a b = max b a",
        "difficulty": "medium",
        "expected_tactic": "split_ifs"
    },
    
    {
        "id": "abs_nonneg",
        "name": "Absolute Value Non-negative",
        "level": 4,
        "code": """
def abs_val(x: int) -> int:
    return x if x >= 0 else -x
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
    
    {
        "id": "abs_zero",
        "name": "Absolute Value of Zero",
        "level": 4,
        "code": """
def abs_zero() -> int:
    return 0 if 0 >= 0 else -0
""",
        "spec": {
            "requires": [],
            "ensures": ["result == 0"],
            "invariants": []
        },
        "expected_lean": "theorem abs_zero : |(0 : ℤ)| = 0",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "min_correct",
        "name": "Minimum Function",
        "level": 4,
        "code": """
def min_val(a: int, b: int) -> int:
    return a if a < b else b
""",
        "spec": {
            "requires": [],
            "ensures": ["result <= a", "result <= b"],
            "invariants": []
        },
        "expected_lean": "theorem min_spec (a b : ℕ) : min a b ≤ a ∧ min a b ≤ b",
        "difficulty": "medium",
        "expected_tactic": "split_ifs"
    },
    
    {
        "id": "sign_function",
        "name": "Sign Function",
        "level": 4,
        "code": """
def sign(x: int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
""",
        "spec": {
            "requires": [],
            "ensures": ["result in [-1, 0, 1]"],
            "invariants": []
        },
        "expected_lean": "theorem sign_range (x : ℤ) : sign x ∈ ({-1, 0, 1} : Set ℤ)",
        "difficulty": "medium",
        "expected_tactic": "split_ifs"
    },
    
    {
        "id": "clamp_function",
        "name": "Clamp Function",
        "level": 4,
        "code": """
def clamp(x: int, low: int, high: int) -> int:
    if x < low:
        return low
    elif x > high:
        return high
    else:
        return x
""",
        "spec": {
            "requires": ["low <= high"],
            "ensures": ["result >= low", "result <= high"],
            "invariants": []
        },
        "expected_lean": "theorem clamp_bounds (x low high : ℤ) (h : low ≤ high) : low ≤ clamp x low high ∧ clamp x low high ≤ high",
        "difficulty": "medium",
        "expected_tactic": "split_ifs"
    },
    
    {
        "id": "is_even",
        "name": "Even Number Check",
        "level": 4,
        "code": """
def is_even(n: int) -> bool:
    return n % 2 == 0
""",
        "spec": {
            "requires": [],
            "ensures": ["result == (n % 2 == 0)"],
            "invariants": []
        },
        "expected_lean": "theorem even_iff_mod_two (n : ℕ) : even n ↔ n % 2 = 0",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "is_positive",
        "name": "Positive Check",
        "level": 4,
        "code": """
def is_positive(n: int) -> bool:
    return n > 0
""",
        "spec": {
            "requires": [],
            "ensures": ["result == (n > 0)"],
            "invariants": []
        },
        "expected_lean": "theorem pos_iff (n : ℤ) : n > 0 ↔ 0 < n",
        "difficulty": "easy",
        "expected_tactic": "simp"
    },
    
    {
        "id": "implies_transitive",
        "name": "Implication Transitivity",
        "level": 4,
        "code": """
def implies_trans(p: bool, q: bool, r: bool) -> bool:
    if p and (p implies q) and (q implies r):
        return r
    return False
""",
        "spec": {
            "requires": [],
            "ensures": ["result == r"],
            "invariants": []
        },
        "expected_lean": "theorem implies_trans (p q r : Prop) : p → (p → q) → (q → r) → r",
        "difficulty": "medium",
        "expected_tactic": "tauto"
    },
    
    {
        "id": "not_not_elim",
        "name": "Double Negation Elimination",
        "level": 4,
        "code": """
def not_not(p: bool) -> bool:
    return not (not p)
""",
        "spec": {
            "requires": [],
            "ensures": ["result == p"],
            "invariants": []
        },
        "expected_lean": "theorem not_not (p : Prop) : ¬¬p → p",
        "difficulty": "medium",
        "expected_tactic": "tauto"
    },
    
    # Level 5: PyTorch/Tensors (10 problems)
    {
        "id": "tensor_add_comm",
        "name": "Tensor Addition Commutativity",
        "level": 5,
        "code": """
import torch
def tensor_add_comm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B
""",
        "spec": {
            "requires": ["A.shape == B.shape"],
            "ensures": ["result.shape == A.shape"],
            "invariants": []
        },
        "expected_lean": "theorem tensor_add_comm (A B : Matrix) : A + B = B + A",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "tensor_add_assoc",
        "name": "Tensor Addition Associativity",
        "level": 5,
        "code": """
import torch
def tensor_add_assoc(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return (A + B) + C
""",
        "spec": {
            "requires": ["A.shape == B.shape", "B.shape == C.shape"],
            "ensures": ["result.shape == A.shape"],
            "invariants": []
        },
        "expected_lean": "theorem tensor_add_assoc (A B C : Matrix) : (A + B) + C = A + (B + C)",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "scalar_mul_assoc",
        "name": "Scalar Multiplication Associativity",
        "level": 5,
        "code": """
import torch
def scalar_mul(a: float, b: float, X: torch.Tensor) -> torch.Tensor:
    return a * (b * X)
""",
        "spec": {
            "requires": [],
            "ensures": ["result.shape == X.shape"],
            "invariants": []
        },
        "expected_lean": "theorem scalar_mul_assoc (a b : ℝ) (X : Matrix) : a • (b • X) = (a * b) • X",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "matmul_assoc",
        "name": "Matrix Multiplication Associativity",
        "level": 5,
        "code": """
import torch
def matmul_assoc(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(A, B), C)
""",
        "spec": {
            "requires": ["A.shape[1] == B.shape[0]", "B.shape[1] == C.shape[0]"],
            "ensures": ["result.shape[0] == A.shape[0]", "result.shape[1] == C.shape[1]"],
            "invariants": []
        },
        "expected_lean": "theorem matmul_assoc (A B C : Matrix) : (A * B) * C = A * (B * C)",
        "difficulty": "hard",
        "expected_tactic": "MCTS"
    },
    
    {
        "id": "relu_nonneg",
        "name": "ReLU Non-negativity",
        "level": 5,
        "code": """
import torch
def relu_nonneg(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)
""",
        "spec": {
            "requires": [],
            "ensures": ["all(result >= 0)"],
            "invariants": []
        },
        "expected_lean": "theorem relu_nonneg (x : Matrix) : ∀ i j, relu x i j ≥ 0",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "linear_shape",
        "name": "Linear Layer Shape",
        "level": 5,
        "code": """
import torch
import torch.nn as nn
def linear_shape(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    return layer(x)
""",
        "spec": {
            "requires": ["x.shape[-1] == layer.in_features"],
            "ensures": ["result.shape[-1] == layer.out_features"],
            "invariants": []
        },
        "expected_lean": "theorem linear_shape (x : Vector) (W : Matrix) : shape (W * x) = shape W",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "softmax_sum_one",
        "name": "Softmax Sums to One",
        "level": 5,
        "code": """
import torch
def softmax_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
""",
        "spec": {
            "requires": [],
            "ensures": ["sum(result, dim=-1) == 1"],
            "invariants": []
        },
        "expected_lean": "theorem softmax_sum_one (x : Vector) : ∑ i, softmax x i = 1",
        "difficulty": "hard",
        "expected_tactic": "MCTS"
    },
    
    {
        "id": "transpose_twice",
        "name": "Transpose Twice is Identity",
        "level": 5,
        "code": """
import torch
def transpose_twice(A: torch.Tensor) -> torch.Tensor:
    return A.t().t()
""",
        "spec": {
            "requires": [],
            "ensures": ["result.shape == A.shape"],
            "invariants": []
        },
        "expected_lean": "theorem transpose_transpose (A : Matrix) : (A^T)^T = A",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "dot_product_comm",
        "name": "Dot Product Commutativity",
        "level": 5,
        "code": """
import torch
def dot_product(u: torch.Tensor, v: torch.Tensor) -> float:
    return torch.dot(u, v).item()
""",
        "spec": {
            "requires": ["u.shape == v.shape"],
            "ensures": ["result == torch.dot(v, u).item()"],
            "invariants": []
        },
        "expected_lean": "theorem dot_comm (u v : Vector) : u ⬝ v = v ⬝ u",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
    
    {
        "id": "norm_nonneg",
        "name": "Vector Norm Non-negative",
        "level": 5,
        "code": """
import torch
def vector_norm(v: torch.Tensor) -> float:
    return torch.norm(v).item()
""",
        "spec": {
            "requires": [],
            "ensures": ["result >= 0"],
            "invariants": []
        },
        "expected_lean": "theorem norm_nonneg (v : Vector) : ‖v‖ ≥ 0",
        "difficulty": "hard",
        "expected_tactic": "simp"
    },
]
