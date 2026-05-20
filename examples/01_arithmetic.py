"""Simple arithmetic functions for testing."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def square(x: int) -> int:
    """Square a number."""
    return x * x

def sum_of_squares(a: int, b: int) -> int:
    """Compute sum of squares."""
    return square(a) + square(b)
