"""
Axiom Zero - Proof Engine Module
RL proof agent environment with Lean 4 integration.
"""

from .proof_state import ProofState, ProofGoal, Variable, GoalStatus
from .tactics import TacticSpace, TacticAction, TacticTemplate, TacticCategory
from .lean_env import LeanEnvironment

__all__ = [
    'ProofState',
    'ProofGoal',
    'Variable',
    'GoalStatus',
    'TacticSpace',
    'TacticAction',
    'TacticTemplate',
    'TacticCategory',
    'LeanEnvironment'
]
