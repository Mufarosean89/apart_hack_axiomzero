"""
AST Parser module for Axiom Zero.
Supports both Python's built-in ast module and tree-sitter for parsing.
"""

import ast
import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PythonASTParser:
    """
    Parser that extracts AST from Python source code.
    Supports both Python's built-in ast module and tree-sitter.
    """

    def __init__(self, use_tree_sitter: bool = False):
        """
        Initialize parser.

        Args:
            use_tree_sitter: If True, attempt to use tree-sitter parser
        """
        self.use_tree_sitter = use_tree_sitter
        self.tree_sitter_available = False

        if use_tree_sitter:
            try:
                from tree_sitter import Language, Parser
                import tree_sitter_python
                self.tree_sitter_available = True
                self.ts_parser = Parser()
                self.ts_parser.set_language(
                    Language(tree_sitter_python.language()))
            except ImportError:
                logger.warning(
                    "tree-sitter not available, falling back to Python ast module"
                )
                self.use_tree_sitter = False
                self.tree_sitter_available = False

    def parse(self, source_code: str, filename: str = "<string>") -> ast.AST:
        """
        Parse Python source code into AST.

        Args:
            source_code: Python source code string
            filename: Source filename for error reporting

        Returns:
            Python AST node

        Raises:
            SyntaxError: If the source code has syntax errors
        """
        if self.use_tree_sitter and self.tree_sitter_available:
            return self._parse_tree_sitter(source_code)
        else:
            return self._parse_python_ast(source_code, filename)

    def _parse_python_ast(self, source_code: str, filename: str) -> ast.AST:
        """
        Parse using Python's built-in ast module.

        Args:
            source_code: Python source code
            filename: Source filename

        Returns:
            AST node
        """
        try:
            return ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            raise SyntaxError(
                f"Syntax error in {filename}: {e.msg} (line {e.lineno})")

    def _parse_tree_sitter(self, source_code: str) -> Any:
        """
        Parse using tree-sitter.

        Args:
            source_code: Python source code

        Returns:
            tree-sitter tree
        """
        if not self.tree_sitter_available:
            raise RuntimeError("tree-sitter is not available")

        source_bytes = source_code.encode('utf-8')
        tree = self.ts_parser.parse(source_bytes)
        return tree

    def parse_file(self, filepath: str) -> ast.AST:
        """
        Parse a Python file.

        Args:
            filepath: Path to Python file

        Returns:
            AST node
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        source_code = path.read_text(encoding='utf-8')
        return self.parse(source_code, filename=str(filepath))

    @staticmethod
    def get_line_number(node: ast.AST) -> int:
        """
        Extract line number from AST node.

        Args:
            node: AST node

        Returns:
            Line number
        """
        return getattr(node, 'lineno', 0)

    @staticmethod
    def dump_ast(tree: ast.AST, indent: int = 0) -> str:
        """
        Pretty print AST for debugging.

        Args:
            tree: AST node
            indent: Indentation level

        Returns:
            Formatted AST string
        """
        result = []

        if isinstance(tree, ast.AST):
            node_name = tree.__class__.__name__
            result.append("  " * indent + node_name)

            for field in tree._fields:
                value = getattr(tree, field, None)
                if isinstance(value, list):
                    result.append("  " * (indent + 1) + f"{field}:")
                    for item in value:
                        result.append(
                            PythonASTParser.dump_ast(item, indent + 2))
                elif isinstance(value, ast.AST):
                    result.append("  " * (indent + 1) + f"{field}:")
                    result.append(PythonASTParser.dump_ast(value, indent + 2))
                else:
                    result.append("  " * (indent + 1) + f"{field}: {value}")
        else:
            result.append("  " * indent + str(tree))

        return "\n".join(result)
