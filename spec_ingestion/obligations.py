"""
Proof obligation data structures for formal specifications.
Defines preconditions, postconditions, invariants, and theorem statements.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class ObligationKind(Enum):
    """Types of proof obligations."""
    PRECONDITION = "precondition"      # @requires - must hold before function execution
    POSTCONDITION = "postcondition"    # @ensures - must hold after function execution
    INVARIANT = "invariant"            # @invariant - must hold throughout loop/class
    THEOREM = "theorem"                # Formal theorem to prove
    LEMMA = "lemma"                    # Supporting lemma
    ASSERTION = "assertion"            # Internal assertion
    TYPE_CONSTRAINT = "type_constraint"  # Type safety constraint
    SHAPE_CONSTRAINT = "shape_constraint"  # Tensor shape constraint
    TERMINATION = "termination"        # Loop/function termination
    SAFETY = "safety"                  # Safety property (no crashes)


@dataclass
class ProofObligation:
    """
    A single proof obligation that must be verified.
    Represents a logical statement that needs formal proof.
    """
    kind: ObligationKind
    statement: str  # Formal logical statement
    natural_language: Optional[str] = None  # Human-readable description
    function_name: Optional[str] = None  # Associated function
    variables: List[str] = field(default_factory=list)  # Variables involved
    assumptions: List[str] = field(default_factory=list)  # Given assumptions
    priority: int = 1  # 1=highest, 5=lowest
    line_number: Optional[int] = None
    source_location: Optional[str] = None

    # For structured obligations
    lhs: Optional[str] = None  # Left-hand side (for equations)
    rhs: Optional[str] = None  # Right-hand side (for equations)
    operator: Optional[str] = None  # Relation operator

    # Metadata
    difficulty: Optional[str] = None  # 'easy', 'medium', 'hard', 'unknown'
    estimated_proof_steps: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'kind': self.kind.value,
            'statement': self.statement,
            'natural_language': self.natural_language,
            'function_name': self.function_name,
            'variables': self.variables,
            'assumptions': self.assumptions,
            'priority': self.priority,
            'line_number': self.line_number,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'operator': self.operator,
            'difficulty': self.difficulty,
            'tags': self.tags
        }

    def __repr__(self):
        return f"ProofObligation({self.kind.value}: {self.statement[:50]}...)"


@dataclass
class Precondition(ProofObligation):
    """Precondition that must hold before function execution."""

    def __post_init__(self):
        self.kind = ObligationKind.PRECONDITION
        self.priority = 1  # Preconditions are always high priority


@dataclass
class Postcondition(ProofObligation):
    """Postcondition that must hold after function execution."""

    def __post_init__(self):
        self.kind = ObligationKind.POSTCONDITION


@dataclass
class Invariant(ProofObligation):
    """Invariant that must hold throughout execution."""
    scope: str = "function"  # 'function', 'loop', 'class'
    loop_variable: Optional[str] = None  # For loop invariants

    def __post_init__(self):
        self.kind = ObligationKind.INVARIANT


@dataclass
class TheoremObligation(ProofObligation):
    """Formal theorem statement."""
    proof_method: Optional[str] = None  # Suggested proof method
    # Other theorems/lemmas needed
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.kind = ObligationKind.THEOREM


@dataclass
class ObligationSet:
    """
    Complete set of proof obligations for a code module.
    Organizes obligations by function and type.
    """
    source_file: Optional[str] = None
    obligations: List[ProofObligation] = field(default_factory=list)

    # Organized by function
    by_function: Dict[str, List[ProofObligation]] = field(default_factory=dict)

    # Organized by type
    preconditions: List[ProofObligation] = field(default_factory=list)
    postconditions: List[ProofObligation] = field(default_factory=list)
    invariants: List[ProofObligation] = field(default_factory=list)
    theorems: List[ProofObligation] = field(default_factory=list)

    # Metadata
    total_obligations: int = 0
    critical_obligations: int = 0  # Priority 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_obligation(self, obligation: ProofObligation):
        """Add a proof obligation."""
        self.obligations.append(obligation)

        # Organize by function
        if obligation.function_name:
            if obligation.function_name not in self.by_function:
                self.by_function[obligation.function_name] = []
            self.by_function[obligation.function_name].append(obligation)

        # Organize by type
        if obligation.kind == ObligationKind.PRECONDITION:
            self.preconditions.append(obligation)
        elif obligation.kind == ObligationKind.POSTCONDITION:
            self.postconditions.append(obligation)
        elif obligation.kind == ObligationKind.INVARIANT:
            self.invariants.append(obligation)
        elif obligation.kind in [ObligationKind.THEOREM, ObligationKind.LEMMA]:
            self.theorems.append(obligation)

        # Update counts
        self.total_obligations = len(self.obligations)
        if obligation.priority == 1:
            self.critical_obligations += 1

    def get_function_obligations(self, function_name: str) -> List[ProofObligation]:
        """Get all obligations for a specific function."""
        return self.by_function.get(function_name, [])

    def get_critical_obligations(self) -> List[ProofObligation]:
        """Get high-priority obligations."""
        return [ob for ob in self.obligations if ob.priority == 1]

    def get_by_kind(self, kind: ObligationKind) -> List[ProofObligation]:
        """Get obligations of a specific kind."""
        return [ob for ob in self.obligations if ob.kind == kind]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_file': self.source_file,
            'total_obligations': self.total_obligations,
            'critical_obligations': self.critical_obligations,
            'obligations': [ob.to_dict() for ob in self.obligations],
            'by_function': {
                func: [ob.to_dict() for ob in obs]
                for func, obs in self.by_function.items()
            },
            'summary': {
                'preconditions': len(self.preconditions),
                'postconditions': len(self.postconditions),
                'invariants': len(self.invariants),
                'theorems': len(self.theorems)
            }
        }

    def summary_str(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Proof Obligation Set",
            f"{'='*60}",
            f"Source: {self.source_file or 'unknown'}",
            f"Total obligations: {self.total_obligations}",
            f"Critical (priority 1): {self.critical_obligations}",
            f"",
            f"Breakdown:",
            f"  Preconditions:  {len(self.preconditions)}",
            f"  Postconditions: {len(self.postconditions)}",
            f"  Invariants:     {len(self.invariants)}",
            f"  Theorems:       {len(self.theorems)}",
        ]

        if self.by_function:
            lines.append(f"\nBy Function:")
            for func_name, obs in self.by_function.items():
                lines.append(f"  {func_name}: {len(obs)} obligations")

        return "\n".join(lines)
