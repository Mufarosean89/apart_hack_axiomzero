"""
Lean 4 specification parser.
Extracts theorems, lemmas, and specifications from Lean code.
"""

import re
from typing import List, Dict, Optional

from .obligations import (
    ProofObligation,
    TheoremObligation,
    Precondition,
    Postcondition,
    Invariant,
    ObligationKind,
    ObligationSet
)


class LeanSpecParser:
    """
    Parses Lean 4 specifications and theorems.
    Extracts formal proof obligations from Lean code.
    """

    def __init__(self):
        """Initialize Lean parser."""
        self.theorem_pattern = re.compile(
            r'(?:theorem|lemma|example)\s+(\w+)\s*(?:{[^}]*})?\s*(?:\(.*?\))?\s*:\s*(.+?)(?:\s*:=|\s*by|\s*{)',
            re.DOTALL
        )
        self.def_pattern = re.compile(
            r'def\s+(\w+)\s*(?:{[^}]*})?\s*(?:\(.*?\))?\s*(?::\s*(.+?))?=(.+?)(?=\ndef\s|\nclass\s|\Z)',
            re.DOTALL
        )
        self.contract_pattern = re.compile(
            r'(?:requires|ensures|invariant)\s*:\s*(.+?)(?=\n\s*(?:requires|ensures|invariant|def|theorem)|\Z)',
            re.IGNORECASE
        )

    def parse(self, lean_code: str, source_file: str = None) -> ObligationSet:
        """
        Parse Lean specification code.

        Args:
            lean_code: Lean source code
            source_file: Source filename

        Returns:
            ObligationSet with extracted proof obligations
        """
        obligation_set = ObligationSet(source_file=source_file)

        # Parse theorems and lemmas
        self._parse_theorems(lean_code, obligation_set)

        # Parse function contracts
        self._parse_contracts(lean_code, obligation_set)

        # Parse type specifications
        self._parse_type_specs(lean_code, obligation_set)

        return obligation_set

    def _parse_theorems(self, lean_code: str, obligation_set: ObligationSet):
        """
        Extract theorem statements as proof obligations.

        Example:
            theorem matrix_mul_assoc (A B C : Matrix) :
              (A * B) * C = A * (B * C) := by ...
        """
        for match in self.theorem_pattern.finditer(lean_code):
            theorem_name = match.group(1)
            theorem_statement = match.group(2).strip()

            # Extract variables from the statement
            variables = self._extract_lean_variables(theorem_statement)

            # Determine if it's a theorem or lemma
            kind = ObligationKind.THEOREM
            if 'lemma' in match.group(0).lower():
                kind = ObligationKind.LEMMA

            obligation = TheoremObligation(
                kind=kind,
                statement=theorem_statement,
                natural_language=f"Theorem: {theorem_name}",
                variables=variables,
                priority=1,
                tags=['lean_theorem', theorem_name]
            )

            obligation_set.add_obligation(obligation)

    def _parse_contracts(self, lean_code: str, obligation_set: ObligationSet):
        """
        Extract function contracts (requires/ensures).

        Example:
            def safe_div (x y : ℝ) (h : y ≠ 0) : ℝ
              requires y ≠ 0
              ensures result * y = x
        """
        lines = lean_code.split('\n')
        current_def = None

        for i, line in enumerate(lines, 1):
            # Track function definitions
            def_match = re.match(r'\s*def\s+(\w+)', line)
            if def_match:
                current_def = def_match.group(1)

            # Parse contract annotations
            req_match = re.match(
                r'\s*requires\s*:\s*(.+)', line, re.IGNORECASE)
            if req_match and current_def:
                spec_text = req_match.group(1).strip()
                obligation = Precondition(
                    statement=spec_text,
                    natural_language=f"Precondition for {current_def}",
                    function_name=current_def,
                    variables=self._extract_lean_variables(spec_text),
                    priority=1,
                    line_number=i
                )
                obligation_set.add_obligation(obligation)

            ens_match = re.match(r'\s*ensures\s*:\s*(.+)', line, re.IGNORECASE)
            if ens_match and current_def:
                spec_text = ens_match.group(1).strip()
                obligation = Postcondition(
                    statement=spec_text,
                    natural_language=f"Postcondition for {current_def}",
                    function_name=current_def,
                    variables=self._extract_lean_variables(spec_text),
                    priority=2,
                    line_number=i
                )
                obligation_set.add_obligation(obligation)

            inv_match = re.match(
                r'\s*invariant\s*:\s*(.+)', line, re.IGNORECASE)
            if inv_match and current_def:
                spec_text = inv_match.group(1).strip()
                obligation = Invariant(
                    statement=spec_text,
                    natural_language=f"Invariant for {current_def}",
                    function_name=current_def,
                    variables=self._extract_lean_variables(spec_text),
                    priority=2,
                    line_number=i
                )
                obligation_set.add_obligation(obligation)

    def _parse_type_specs(self, lean_code: str, obligation_set: ObligationSet):
        """
        Extract type specifications and constraints.

        Example:
            variable (n m : ℕ)
            hypothesis h : n > 0
        """
        # Parse variable declarations with types
        var_pattern = re.compile(r'variable\s*\((\w+)\s*:\s*(.+?)\)')
        for match in var_pattern.finditer(lean_code):
            var_name = match.group(1)
            var_type = match.group(2).strip()

            # Type constraints become obligations
            if any(op in var_type for op in ['>', '<', '≥', '≤', '≠']):
                obligation = ProofObligation(
                    kind=ObligationKind.TYPE_CONSTRAINT,
                    statement=f"{var_name} : {var_type}",
                    natural_language=f"Type constraint: {var_name} has type {var_type}",
                    variables=[var_name],
                    priority=3,
                    tags=['type_constraint', var_type]
                )
                obligation_set.add_obligation(obligation)

        # Parse hypothesis statements
        hyp_pattern = re.compile(r'hypothesis\s+(\w+)\s*:\s*(.+)')
        for match in hyp_pattern.finditer(lean_code):
            hyp_name = match.group(1)
            hyp_statement = match.group(2).strip()

            obligation = ProofObligation(
                kind=ObligationKind.ASSERTION,
                statement=hyp_statement,
                natural_language=f"Assumption: {hyp_name}",
                variables=self._extract_lean_variables(hyp_statement),
                priority=2,
                tags=['hypothesis', hyp_name]
            )
            obligation_set.add_obligation(obligation)

    def _extract_lean_variables(self, statement: str) -> List[str]:
        """
        Extract variable names from Lean statement.

        Args:
            statement: Lean logical statement

        Returns:
            List of variable names
        """
        # Match Lean identifiers (alphanumeric with underscores)
        # Exclude Lean keywords and symbols
        keywords = {
            'theorem', 'lemma', 'def', 'example', 'variable', 'hypothesis',
            'requires', 'ensures', 'invariant', 'by', 'sorry', 'begin', 'end',
            'have', 'let', 'calc', 'apply', 'exact', 'rw', 'simp', 'ring',
            'forall', 'exists', 'implies', 'and', 'or', 'not', 'eq', 'ne',
            'true', 'false', 'Type', 'Prop', 'Sort',
            '', 'ℤ', 'ℚ', 'ℝ', 'ℂ',  # Unicode types
        }

        # Find all identifiers
        variables = re.findall(r'\b[a-zA-Z_]\w*\b', statement)

        # Filter out keywords and single-letter operators
        filtered = [
            v for v in variables
            if v not in keywords and (len(v) > 1 or v.isupper())
        ]

        return list(set(filtered))  # Remove duplicates

    def parse_file(self, filepath: str) -> ObligationSet:
        """
        Parse Lean specification from file.

        Args:
            filepath: Path to Lean file

        Returns:
            ObligationSet with extracted obligations
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lean_code = f.read()
            return self.parse(lean_code, source_file=filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Lean file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse Lean file: {e}")
