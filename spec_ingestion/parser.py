"""
Specification parser for Python decorator-based annotations.
Extracts @requires, @ensures, @invariant decorators into proof obligations.
"""

import ast
import re
from typing import List, Dict, Optional, Tuple

from .obligations import (
    ProofObligation,
    Precondition,
    Postcondition,
    Invariant,
    ObligationKind,
    ObligationSet
)


class SpecParser:
    """
    Parses formal specifications from Python code.
    Supports decorator-based and inline specification formats.
    """

    def __init__(self):
        """Initialize specification parser."""
        self.spec_decorators = {
            'requires': ObligationKind.PRECONDITION,
            'ensures': ObligationKind.POSTCONDITION,
            'invariant': ObligationKind.INVARIANT,
            'precondition': ObligationKind.PRECONDITION,
            'postcondition': ObligationKind.POSTCONDITION,
        }

    def parse(self, source_code: str, spec_type: str = 'python_decorators',
              source_file: str = None) -> ObligationSet:
        """
        Parse specifications from source code.

        Args:
            source_code: Source code string
            spec_type: Type of specification format
            source_file: Source filename

        Returns:
            ObligationSet with parsed proof obligations
        """
        obligation_set = ObligationSet(source_file=source_file)

        if spec_type == 'python_decorators':
            self._parse_decorators(source_code, obligation_set)
        elif spec_type == 'inline':
            self._parse_inline_specs(source_code, obligation_set)
        elif spec_type == 'docstring':
            self._parse_docstring_specs(source_code, obligation_set)

        return obligation_set

    def _parse_decorators(self, source_code: str, obligation_set: ObligationSet):
        """
        Parse decorator-based specifications.

        Example:
            @requires("x > 0")
            @ensures("result > x")
            def foo(x: int) -> int:
                ...
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse source code: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._parse_function_decorators(node, obligation_set)

    def _parse_function_decorators(self, func_node: ast.FunctionDef,
                                   obligation_set: ObligationSet):
        """Parse decorators on a function definition."""
        func_name = func_node.name
        current_line = func_node.lineno

        # Process decorators in reverse order (bottom to top in source)
        for decorator in reversed(func_node.decorator_list):
            if isinstance(decorator, ast.Call):
                # Get decorator name
                if isinstance(decorator.func, ast.Name):
                    decorator_name = decorator.func.id.lower()
                elif isinstance(decorator.func, ast.Attribute):
                    decorator_name = decorator.func.attr.lower()
                else:
                    continue

                if decorator_name in self.spec_decorators:
                    # Extract specification string
                    if decorator.args:
                        spec_arg = decorator.args[0]
                        if isinstance(spec_arg, ast.Constant):
                            spec_text = spec_arg.value
                            self._create_obligation(
                                obligation_set,
                                self.spec_decorators[decorator_name],
                                spec_text,
                                func_name,
                                current_line
                            )
                        elif isinstance(spec_arg, ast.JoinedStr):
                            # f-string
                            spec_text = self._extract_fstring_value(spec_arg)
                            self._create_obligation(
                                obligation_set,
                                self.spec_decorators[decorator_name],
                                spec_text,
                                func_name,
                                current_line
                            )

    def _parse_inline_specs(self, source_code: str, obligation_set: ObligationSet):
        """
        Parse inline specifications in comments or assert statements.

        Example:
            def foo(x: int) -> int:
                # @requires x > 0
                # @ensures result > x
                assert x > 0  # Precondition
                ...
        """
        lines = source_code.split('\n')
        current_function = None

        for line_num, line in enumerate(lines, 1):
            # Detect function definitions
            func_match = re.match(r'\s*def\s+(\w+)\s*\(', line)
            if func_match:
                current_function = func_match.group(1)

            # Parse inline spec comments
            spec_match = re.search(r'#\s*@(\w+)\s+(.+)', line)
            if spec_match:
                spec_type = spec_match.group(1).lower()
                spec_text = spec_match.group(2).strip()

                if spec_type in self.spec_decorators:
                    self._create_obligation(
                        obligation_set,
                        self.spec_decorators[spec_type],
                        spec_text,
                        current_function,
                        line_num
                    )

            # Parse assert statements as preconditions/postconditions
            assert_match = re.match(
                r'\s*assert\s+(.+?)(?:,\s*(.+))?(?:\s*#\s*(.*))?$', line)
            if assert_match and current_function:
                condition = assert_match.group(1).strip()
                message = assert_match.group(3)

                # Heuristic: asserts at start are preconditions, at end are postconditions
                self._create_obligation(
                    obligation_set,
                    ObligationKind.ASSERTION,
                    condition,
                    current_function,
                    line_num,
                    natural_language=message
                )

    def _parse_docstring_specs(self, source_code: str, obligation_set: ObligationSet):
        """
        Parse specifications from docstrings.

        Example:
            def foo(x: int) -> int:
                '''
                Pre: x > 0
                Post: result > x
                Invariant: x == x@old
                '''
                ...
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                    docstring = node.body[0].value.value
                    self._parse_docstring_content(
                        docstring,
                        node.name,
                        obligation_set,
                        node.lineno
                    )

    def _parse_docstring_content(self, docstring: str, func_name: str,
                                 obligation_set: ObligationSet, line_num: int):
        """Parse specification content from docstring."""
        if not docstring:
            return

        lines = docstring.split('\n')

        for line in lines:
            line = line.strip()

            # Match patterns like "Pre:", "Precondition:", "Requires:"
            pre_match = re.match(
                r'(?:pre(?:condition)?|requires?)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if pre_match:
                self._create_obligation(
                    obligation_set,
                    ObligationKind.PRECONDITION,
                    pre_match.group(1).strip(),
                    func_name,
                    line_num
                )

            # Match patterns like "Post:", "Postcondition:", "Ensures:"
            post_match = re.match(
                r'(?:post(?:condition)?|ensures?)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if post_match:
                self._create_obligation(
                    obligation_set,
                    ObligationKind.POSTCONDITION,
                    post_match.group(1).strip(),
                    func_name,
                    line_num
                )

            # Match patterns like "Invariant:"
            inv_match = re.match(
                r'invariant\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if inv_match:
                self._create_obligation(
                    obligation_set,
                    ObligationKind.INVARIANT,
                    inv_match.group(1).strip(),
                    func_name,
                    line_num
                )

    def _create_obligation(self, obligation_set: ObligationSet, kind: ObligationKind,
                           spec_text: str, func_name: str, line_num: int,
                           natural_language: str = None):
        """Create a proof obligation from specification text."""
        # Parse structured format if present
        lhs, operator, rhs = self._parse_spec_structure(spec_text)

        # Determine priority based on kind
        priority = 1 if kind in [
            ObligationKind.PRECONDITION, ObligationKind.SAFETY] else 2

        # Extract variables
        variables = self._extract_variables(spec_text)

        obligation = ProofObligation(
            kind=kind,
            statement=spec_text,
            natural_language=natural_language,
            function_name=func_name,
            variables=variables,
            priority=priority,
            line_number=line_num,
            lhs=lhs,
            rhs=rhs,
            operator=operator
        )

        obligation_set.add_obligation(obligation)

    def _parse_spec_structure(self, spec_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse specification into structured form.

        Example: "x > 0" -> ("x", ">", "0")
        """
        # Try common patterns
        patterns = [
            (r'(\w+)\s*(==|!=|<=|>=|<|>)\s*(\w+|\d+)', 3),
            (r'(\w+)\s+(in|is|not)\s+(.+)', 3),
            (r'(\w+)\.(\w+)\s*(==|!=|<=|>=|<|>)\s*(\w+|\d+)', 4),
        ]

        for pattern, groups in patterns:
            match = re.search(pattern, spec_text)
            if match:
                if groups == 3:
                    return match.group(1), match.group(2), match.group(3)
                elif groups == 4:
                    lhs = f"{match.group(1)}.{match.group(2)}"
                    return lhs, match.group(3), match.group(4)

        return None, None, None

    def _extract_variables(self, spec_text: str) -> List[str]:
        """Extract variable names from specification text."""
        # Simple regex to find identifiers
        variables = re.findall(r'\b[a-zA-Z_]\w*\b', spec_text)
        # Filter out common keywords
        keywords = {'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while',
                    'True', 'False', 'None', 'result', 'old', 'self'}
        return [v for v in variables if v not in keywords]

    def _extract_fstring_value(self, fstring_node: ast.JoinedStr) -> str:
        """Extract value from f-string AST node."""
        parts = []
        for value in fstring_node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                # Simplified - just mark as variable
                parts.append("{var}")
        return ''.join(parts)
