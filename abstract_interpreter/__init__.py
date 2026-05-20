"""
Axiom Zero - Abstract Interpreter Module
Performs symbolic type inference and tensor shape analysis over normalized IR.
Tracks data flow and generates background facts for the proof state.
"""

from .interpreter import AbstractInterpreter
from .shape_analysis import TensorShapeAnalyzer
from .type_inference import TypeInferenceEngine
from .abstract_domain import (
    AbstractValue,
    TensorShape,
    TypeDomain,
    AbstractState
)

__all__ = [
    'run_abstract_interpretation',
    'AbstractInterpreter',
    'TensorShapeAnalyzer',
    'TypeInferenceEngine',
    'AbstractValue',
    'TensorShape',
    'TypeDomain',
    'AbstractState'
]


def run_abstract_interpretation(normalized_ir) -> AbstractState:
    """
    Run abstract interpretation over normalized IR.

    Args:
        normalized_ir: NormalizedIR from AST extraction

    Returns:
        AbstractState containing inferred types, shapes, and data flow facts
    """
    interpreter = AbstractInterpreter()
    return interpreter.analyze(normalized_ir)
