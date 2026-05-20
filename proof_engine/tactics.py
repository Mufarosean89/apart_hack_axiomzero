"""
Tactic definitions and action space for the RL proof agent.
Defines available tactics and their properties.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class TacticCategory(Enum):
    """Categories of tactics."""
    BASIC = "basic"              # intro, exact, apply
    SIMPLIFICATION = "simpl"     # simp, ring, norm_num
    DECISION = "decision"        # omega, decide, linarith
    CONSTRUCTION = "construction"  # have, let, choose
    REASONING = "reasoning"      # cases, induction, by_cases
    LIBRARY = "library"          # External lemmas from Mathlib
    STRUCTURAL = "structural"    # constructor, left, right


@dataclass
class TacticTemplate:
    """
    Template for a tactic with parameter slots.
    Used to generate concrete tactic applications.
    """
    name: str
    category: TacticCategory
    template: str  # Tactic with {placeholders}
    description: str
    parameters: List[Dict[str, str]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)  # When to use
    effects: List[str] = field(default_factory=list)  # What it does

    def instantiate(self, **kwargs) -> str:
        """
        Instantiate tactic template with concrete values.

        Args:
            **kwargs: Parameter values

        Returns:
            Concrete tactic string
        """
        tactic = self.template
        for key, value in kwargs.items():
            tactic = tactic.replace(f"{{{key}}}", str(value))
        return tactic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category.value,
            'template': self.template,
            'description': self.description,
            'parameters': self.parameters,
            'examples': self.examples
        }


@dataclass
class TacticAction:
    """
    A concrete tactic action taken by the agent.
    """
    tactic_name: str
    tactic_string: str  # Full tactic code
    category: TacticCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    goal_id: Optional[int] = None  # Which goal this applies to
    confidence: float = 1.0  # Agent's confidence in this tactic
    predicted_outcome: Optional[str] = None  # Expected result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tactic_name': self.tactic_name,
            'tactic_string': self.tactic_string,
            'category': self.category.value,
            'parameters': self.parameters,
            'goal_id': self.goal_id,
            'confidence': self.confidence
        }


class TacticSpace:
    """
    The action space of available tactics.
    Manages tactic templates and generates valid actions.
    """

    def __init__(self):
        """Initialize tactic space with standard tactics."""
        self.tactics: Dict[str, TacticTemplate] = {}
        self._initialize_basic_tactics()
        self._initialize_simplification_tactics()
        self._initialize_decision_tactics()
        self._initialize_construction_tactics()
        self._initialize_reasoning_tactics()
        self._initialize_structural_tactics()

    def _initialize_basic_tactics(self):
        """Initialize basic tactics."""
        self.tactics['intro'] = TacticTemplate(
            name='intro',
            category=TacticCategory.BASIC,
            template='intro {name}',
            description='Introduce a variable from a ∀ or → goal',
            parameters=[{'name': 'name', 'type': 'identifier',
                         'description': 'Variable name'}],
            examples=['intro x', 'intro h', 'intro n'],
            preconditions=['Goal starts with ∀ or →'],
            effects=['Removes outermost ∀/→', 'Adds variable to context']
        )

        self.tactics['apply'] = TacticTemplate(
            name='apply',
            category=TacticCategory.BASIC,
            template='apply {lemma}',
            description='Apply a lemma or hypothesis to the goal',
            parameters=[{'name': 'lemma', 'type': 'term',
                         'description': 'Lemma or hypothesis name'}],
            examples=['apply h', 'apply Nat.add_comm', 'apply mul_assoc'],
            preconditions=['Goal matches conclusion of lemma'],
            effects=['Creates subgoals for lemma premises']
        )

        self.tactics['exact'] = TacticTemplate(
            name='exact',
            category=TacticCategory.BASIC,
            template='exact {term}',
            description='Provide exact proof term',
            parameters=[{'name': 'term', 'type': 'term',
                         'description': 'Proof term'}],
            examples=['exact h', 'exact rfl', 'exact Nat.zero'],
            preconditions=['Term type matches goal'],
            effects=['Closes goal if term type matches']
        )

        self.tactics['refine'] = TacticTemplate(
            name='refine',
            category=TacticCategory.BASIC,
            template='refine {term}',
            description='Refine goal with partial proof term',
            parameters=[{'name': 'term', 'type': 'term',
                         'description': 'Partial proof term'}],
            examples=['refine ?_', 'refine Eq.trans ?_ ?_'],
            preconditions=['Term partially matches goal'],
            effects=['Creates subgoals for metavariables']
        )

    def _initialize_simplification_tactics(self):
        """Initialize simplification tactics."""
        self.tactics['simp'] = TacticTemplate(
            name='simp',
            category=TacticCategory.SIMPLIFICATION,
            template='simp {args}',
            description='Simplify goal using rewrite rules',
            parameters=[{'name': 'args', 'type': 'optional',
                         'description': 'Additional simp lemmas'}],
            examples=['simp', 'simp [h]', 'simp only [add_comm]'],
            preconditions=['Goal contains reducible expressions'],
            effects=['Applies simplification rules']
        )

        self.tactics['ring'] = TacticTemplate(
            name='ring',
            category=TacticCategory.SIMPLIFICATION,
            template='ring',
            description='Prove equalities in commutative rings',
            parameters=[],
            examples=['ring', 'ring_nf'],
            preconditions=['Goal is ring equality'],
            effects=['Proves or normalizes ring expressions']
        )

        self.tactics['norm_num'] = TacticTemplate(
            name='norm_num',
            category=TacticCategory.SIMPLIFICATION,
            template='norm_num',
            description='Normalize numerical expressions',
            parameters=[],
            examples=['norm_num', 'norm_num at h'],
            preconditions=['Goal contains numerical expressions'],
            effects=['Evaluates numeric computations']
        )

        self.tactics['field_simp'] = TacticTemplate(
            name='field_simp',
            category=TacticCategory.SIMPLIFICATION,
            template='field_simp {args}',
            description='Simplify field expressions',
            parameters=[{'name': 'args', 'type': 'optional',
                         'description': 'Hypotheses to use'}],
            examples=['field_simp', 'field_simp [h]'],
            preconditions=['Goal involves division'],
            effects=['Clears denominators']
        )

    def _initialize_decision_tactics(self):
        """Initialize decision procedures."""
        self.tactics['omega'] = TacticTemplate(
            name='omega',
            category=TacticCategory.DECISION,
            template='omega',
            description='Solve linear arithmetic over integers',
            parameters=[],
            examples=['omega', 'linarith'],
            preconditions=['Goal is linear arithmetic'],
            effects=['Proves or disproves linear constraints']
        )

        self.tactics['linarith'] = TacticTemplate(
            name='linarith',
            category=TacticCategory.DECISION,
            template='linarith {args}',
            description='Linear arithmetic decision procedure',
            parameters=[{'name': 'args', 'type': 'optional',
                         'description': 'Additional hypotheses'}],
            examples=['linarith', 'linarith [h1, h2]'],
            preconditions=['Goal is linear inequality/equality'],
            effects=['Solves linear arithmetic']
        )

        self.tactics['decide'] = TacticTemplate(
            name='decide',
            category=TacticCategory.DECISION,
            template='decide',
            description='Decide decidable propositions by computation',
            parameters=[],
            examples=['decide', 'decide!'],
            preconditions=['Goal is decidable and concrete'],
            effects=['Computes truth value']
        )

    def _initialize_construction_tactics(self):
        """Initialize construction tactics."""
        self.tactics['have'] = TacticTemplate(
            name='have',
            category=TacticCategory.CONSTRUCTION,
            template='have {name} : {type} := {proof}',
            description='Introduce an intermediate lemma',
            parameters=[
                {'name': 'name', 'type': 'identifier', 'description': 'Lemma name'},
                {'name': 'type', 'type': 'type', 'description': 'Lemma type'},
                {'name': 'proof', 'type': 'term', 'description': 'Proof term'}
            ],
            examples=['have h : x > 0 := by linarith',
                      'have : A → B := λ a => ...'],
            preconditions=['Need intermediate result'],
            effects=['Adds hypothesis to context']
        )

        self.tactics['let'] = TacticTemplate(
            name='let',
            category=TacticCategory.CONSTRUCTION,
            template='let {name} : {type} := {value}',
            description='Define a local abbreviation',
            parameters=[
                {'name': 'name', 'type': 'identifier',
                    'description': 'Variable name'},
                {'name': 'type', 'type': 'type', 'description': 'Variable type'},
                {'name': 'value', 'type': 'term', 'description': 'Value'}
            ],
            examples=['let n := 5', 'let f := λ x => x + 1'],
            preconditions=['Need local definition'],
            effects=['Adds definition to context']
        )

    def _initialize_reasoning_tactics(self):
        """Initialize reasoning tactics."""
        self.tactics['cases'] = TacticTemplate(
            name='cases',
            category=TacticCategory.REASONING,
            template='cases {variable}',
            description='Case analysis on inductive type',
            parameters=[{'name': 'variable', 'type': 'identifier',
                         'description': 'Variable to case on'}],
            examples=['cases n', 'cases h', 'cases l'],
            preconditions=['Variable has inductive type'],
            effects=['Creates subgoals for each constructor']
        )

        self.tactics['induction'] = TacticTemplate(
            name='induction',
            category=TacticCategory.REASONING,
            template='induction {variable} with {inductive_hyp}',
            description='Proof by induction',
            parameters=[
                {'name': 'variable', 'type': 'identifier',
                    'description': 'Variable to induct on'},
                {'name': 'inductive_hyp', 'type': 'identifier',
                    'description': 'IH name'}
            ],
            examples=['induction n with ih', 'induction l with hd tl ih'],
            preconditions=['Variable is natural number or list'],
            effects=['Creates base case and inductive step']
        )

        self.tactics['by_cases'] = TacticTemplate(
            name='by_cases',
            category=TacticCategory.REASONING,
            template='by_cases h : {proposition}',
            description='Case analysis on proposition',
            parameters=[
                {'name': 'h', 'type': 'identifier',
                    'description': 'Hypothesis name'},
                {'name': 'proposition', 'type': 'term',
                    'description': 'Proposition to split on'}
            ],
            examples=['by_cases h : x > 0', 'by_cases : P ∨ Q'],
            preconditions=['Need classical case split'],
            effects=['Creates two subgoals: P and ¬P']
        )

    def _initialize_structural_tactics(self):
        """Initialize structural tactics."""
        self.tactics['constructor'] = TacticTemplate(
            name='constructor',
            category=TacticCategory.STRUCTURAL,
            template='constructor',
            description='Apply constructor for inductive type',
            parameters=[],
            examples=['constructor', 'constructor <;> simp'],
            preconditions=['Goal is inductive type with single constructor'],
            effects=['Applies constructor, creates subgoals']
        )

        self.tactics['left'] = TacticTemplate(
            name='left',
            category=TacticCategory.STRUCTURAL,
            template='left',
            description='Prove left side of disjunction',
            parameters=[],
            examples=['left', 'left; exact h'],
            preconditions=['Goal is P ∨ Q'],
            effects=['Reduces goal to P']
        )

        self.tactics['right'] = TacticTemplate(
            name='right',
            category=TacticCategory.STRUCTURAL,
            template='right',
            description='Prove right side of disjunction',
            parameters=[],
            examples=['right', 'right; exact h'],
            preconditions=['Goal is P ∨ Q'],
            effects=['Reduces goal to Q']
        )

    def get_tactic(self, name: str) -> Optional[TacticTemplate]:
        """Get tactic template by name."""
        return self.tactics.get(name)

    def get_tactics_by_category(self, category: TacticCategory) -> List[TacticTemplate]:
        """Get all tactics in a category."""
        return [t for t in self.tactics.values() if t.category == category]

    def get_all_tactics(self) -> List[TacticTemplate]:
        """Get all available tactics."""
        return list(self.tactics.values())

    def suggest_tactics(self, goal_target: str, context: List = None) -> List[TacticTemplate]:
        """
        Suggest relevant tactics based on goal structure.

        Args:
            goal_target: Target type to prove
            context: Available hypotheses

        Returns:
            List of suggested tactics
        """
        suggestions = []

        # Check for implication or forall
        if '→' in goal_target or '∀' in goal_target or '->' in goal_target:
            suggestions.append(self.tactics['intro'])

        # Check for equality
        if '=' in goal_target:
            suggestions.append(self.tactics['ring'])
            suggestions.append(self.tactics['simp'])

        # Check for arithmetic
        if any(op in goal_target for op in ['<', '>', '≤', '≥', '+', '-']):
            suggestions.append(self.tactics['linarith'])
            suggestions.append(self.tactics['omega'])

        # Check for decidable propositions
        if goal_target in ['True', 'False'] or goal_target.startswith('decidable'):
            suggestions.append(self.tactics['decide'])

        # Always suggest basic tactics
        suggestions.append(self.tactics['apply'])
        suggestions.append(self.tactics['exact'])
        suggestions.append(self.tactics['have'])

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """Convert tactic space to dictionary."""
        return {
            'total_tactics': len(self.tactics),
            'tactics': {name: t.to_dict() for name, t in self.tactics.items()},
            'categories': list(set(t.category.value for t in self.tactics.values()))
        }
