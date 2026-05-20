"""List operations for testing."""
from typing import List

def append_lists(xs: List[int], ys: List[int]) -> List[int]:
    """Append two lists."""
    return xs + ys

def list_length(xs: List[int]) -> int:
    """Get length of list."""
    return len(xs)

def sum_list(xs: List[int]) -> int:
    """Sum all elements in list."""
    total = 0
    for x in xs:
        total = total + x
    return total

def reverse_list(xs: List[int]) -> List[int]:
    """Reverse a list."""
    return xs[::-1]
