"""
Axiom Zero - Specification Ingestion Module
Parses formal specifications from decorated Python or Lean files into proof obligations.
"""

from .parser import SpecParser
from .obligations import (
    ProofObligation,
    Precondition,
    Postcondition,
    Invariant,
    ObligationKind,
    ObligationSet
)
from .lean_parser import LeanSpecParser

__all__ = [
    'parse_specifications',
    'extract_from_decorators',
    'extract_from_lean_file',
    'SpecParser',
    'LeanSpecParser',
    'ProofObligation',
    'Precondition',
    'Postcondition',
    'Invariant',
    'ObligationKind',
    'ObligationSet'
]


def parse_specifications(source, spec_type: str = 'auto', source_file: str = None) -> ObligationSet:
    """
    Parse formal specifications from various sources.

    Args:
        source: Source code string or file path
        spec_type: 'auto', 'python_decorators', 'lean', 'inline'
        source_file: Source filename for reference

    Returns:
        ObligationSet containing all proof obligations
    """
    if spec_type == 'auto':
        # Auto-detect based on content
        if isinstance(source, str):
            if source.strip().startswith('import') or 'theorem' in source or ('def ' in source and ':=' in source):
                spec_type = 'lean'
            elif '@requires' in source or '@ensures' in source or '@invariant' in source:
                spec_type = 'python_decorators'
            else:
                spec_type = 'inline'
        else:
            spec_type = 'python_decorators'

    if spec_type == 'lean':
        parser = LeanSpecParser()
        return parser.parse(source, source_file)
    else:
        parser = SpecParser()
        return parser.parse(source, spec_type, source_file)


def extract_from_decorators(python_code: str, source_file: str = None) -> ObligationSet:
    """
    Extract specifications from Python decorator annotations.

    Args:
        python_code: Python source with @requires, @ensures, @invariant decorators
        source_file: Source filename

    Returns:
        ObligationSet with extracted proof obligations
    """
    parser = SpecParser()
    return parser.parse(python_code, 'python_decorators', source_file)


def extract_from_lean_file(lean_code: str, source_file: str = None) -> ObligationSet:
    """
    Extract proof obligations from Lean specification file.

    Args:
        lean_code: Lean source code with theorems and specifications
        source_file: Source filename

    Returns:
        ObligationSet with extracted proof obligations
    """
    parser = LeanSpecParser()
    return parser.parse(lean_code, source_file)
