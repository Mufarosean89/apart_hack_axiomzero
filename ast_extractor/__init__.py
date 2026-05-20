"""
Axiom Zero - AST Extraction Module
Parses Python source code into a normalized intermediate representation (IR)
for formal verification and proof generation.
"""

from .parser import PythonASTParser
from .normalizer import ASTNormalizer
from .ir import (
    NormalizedIR,
    FunctionIR,
    LoopIR,
    ConditionalIR,
    TensorOpKind,
    TypeAnnotationIR,
    StatementIR,
    ExpressionIR
)

__all__ = [
    'extract_ast',
    'parse_to_ir',
    'PythonASTParser',
    'ASTNormalizer',
    'NormalizedIR',
    'FunctionIR',
    'LoopIR',
    'ConditionalIR',
    'TensorOpKind',
    'TypeAnnotationIR',
    'StatementIR',
    'ExpressionIR'
]


def extract_ast(source_code: str, use_tree_sitter: bool = False):
    """
    Extract AST from Python source code.

    Args:
        source_code: Python source code string
        use_tree_sitter: If True, use tree-sitter parser; otherwise use Python's ast module

    Returns:
        Parsed AST tree
    """
    parser = PythonASTParser(use_tree_sitter=use_tree_sitter)
    return parser.parse(source_code)


def parse_to_ir(source_code: str, use_tree_sitter: bool = False) -> NormalizedIR:
    """
    Parse Python source code directly into normalized IR.

    Args:
        source_code: Python source code string
        use_tree_sitter: If True, use tree-sitter parser

    Returns:
        NormalizedIR object containing structured representation
    """
    parser = PythonASTParser(use_tree_sitter=use_tree_sitter)
    ast_tree = parser.parse(source_code)

    normalizer = ASTNormalizer()
    return normalizer.normalize(ast_tree)
