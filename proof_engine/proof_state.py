"""
Axiom Zero - Proof State Module
Represents the current state of a proof in Lean 4.
Contains goals, local context, and target types.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class GoalStatus(Enum):
    """Status of a proof goal."""
    OPEN = "open"           # Goal needs to be proved
    SOLVED = "solved"       # Goal has been proved
    FAILED = "failed"       # Tactic application failed
    STUCK = "stuck"         # No progress possible


@dataclass
class Variable:
    """
    A variable in the local context.
    Represents a hypothesis or assumption available for the proof.
    """
    name: str
    var_type: str  # Lean type expression
    value: Optional[str] = None  # Concrete value (if known)
    is_hypothesis: bool = False  # True if this is an assumption

    def to_lean(self) -> str:
        """Convert to Lean declaration."""
        if self.is_hypothesis:
            return f"hypothesis {self.name} : {self.var_type}"
        else:
            return f"variable {self.name} : {self.var_type}"

    def __repr__(self):
        return f"{self.name} : {self.var_type}"


@dataclass
class ProofGoal:
    """
    A single proof goal.
    Represents what needs to be proved given the current context.
    """
    goal_id: int
    target: str  # The type/term to prove
    context: List[Variable] = field(
        default_factory=list)  # Available hypotheses
    status: GoalStatus = GoalStatus.OPEN
    depth: int = 0  # Proof depth
    tactic_history: List[str] = field(default_factory=list)  # Tactics applied

    def add_hypothesis(self, name: str, var_type: str, is_hypothesis: bool = False):
        """Add a variable to the local context."""
        var = Variable(name=name, var_type=var_type,
                       is_hypothesis=is_hypothesis)
        self.context.append(var)

    def apply_tactic(self, tactic: str):
        """Record that a tactic was applied."""
        self.tactic_history.append(tactic)
        self.depth += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'goal_id': self.goal_id,
            'target': self.target,
            'context': [str(v) for v in self.context],
            'status': self.status.value,
            'depth': self.depth,
            'tactic_history': self.tactic_history
        }

    def __repr__(self):
        ctx_str = ", ".join(str(v) for v in self.context[:3])
        if len(self.context) > 3:
            ctx_str += f", ... (+{len(self.context)-3} more)"
        return f"Goal {self.goal_id}: ⊢ {self.target} [{ctx_str}]"


@dataclass
class ProofState:
    """
    Complete proof state.
    Snapshot of what Lean 4's kernel knows at a point in the proof.
    """
    theorem_name: str
    goals: List[ProofGoal] = field(default_factory=list)
    global_context: List[Variable] = field(default_factory=list)

    # Proof metadata
    current_goal_idx: int = 0
    tactic_sequence: List[str] = field(default_factory=list)
    total_tactics_applied: int = 0

    # Status tracking
    is_complete: bool = False
    has_error: bool = False
    error_message: Optional[str] = None

    # For RL agent
    state_vector: Optional[List[float]] = None  # Numerical representation
    reward: float = 0.0

    def __post_init__(self):
        """Create a default trivial goal if none provided."""
        if not self.goals:
            self.goals.append(ProofGoal(goal_id=0, target="True"))

    @property
    def current_goal(self) -> Optional[ProofGoal]:
        """Get the current goal being worked on."""
        if 0 <= self.current_goal_idx < len(self.goals):
            return self.goals[self.current_goal_idx]
        return None

    @property
    def open_goals(self) -> List[ProofGoal]:
        """Get all open (unsolved) goals."""
        return [g for g in self.goals if g.status == GoalStatus.OPEN]

    @property
    def solved_goals(self) -> List[ProofGoal]:
        """Get all solved goals."""
        return [g for g in self.goals if g.status == GoalStatus.SOLVED]

    @property
    def num_open_goals(self) -> int:
        """Number of remaining open goals."""
        return len(self.open_goals)

    def is_proved(self) -> bool:
        """Check if all goals are proved."""
        return len(self.open_goals) == 0 and not self.has_error

    def add_goal(self, target: str, context: List[Variable] = None) -> int:
        """
        Add a new proof goal.

        Args:
            target: Target type to prove
            context: Local context (hypotheses)

        Returns:
            Goal ID
        """
        goal_id = len(self.goals)
        goal = ProofGoal(
            goal_id=goal_id,
            target=target,
            context=context or [],
            depth=self.current_goal.depth + 1 if self.current_goal else 0
        )
        self.goals.append(goal)
        return goal_id

    def solve_goal(self, goal_id: int):
        """Mark a goal as solved."""
        for goal in self.goals:
            if goal.goal_id == goal_id:
                goal.status = GoalStatus.SOLVED
                break

    def fail_goal(self, goal_id: int, error: str = None):
        """Mark a goal as failed."""
        for goal in self.goals:
            if goal.goal_id == goal_id:
                goal.status = GoalStatus.FAILED
                if error:
                    self.error_message = error
                self.has_error = True
                break

    def apply_tactic(self, tactic: str, success: bool = True,
                     new_goals: List[ProofGoal] = None, error: str = None):
        """
        Apply a tactic to the current goal.

        Args:
            tactic: Tactic string
            success: Whether tactic succeeded
            new_goals: New subgoals created
            error: Error message if failed
        """
        if self.current_goal:
            self.current_goal.apply_tactic(tactic)

        self.tactic_sequence.append(tactic)
        self.total_tactics_applied += 1

        if success:
            if self.current_goal:
                self.current_goal.status = GoalStatus.SOLVED

            # Add new subgoals
            if new_goals:
                self.goals.extend(new_goals)
                self.current_goal_idx = len(self.goals) - len(new_goals)
        else:
            if error:
                self.error_message = error
            self.has_error = True
            if self.current_goal:
                self.current_goal.status = GoalStatus.FAILED

    def get_state_features(self) -> Dict[str, Any]:
        """
        Extract features for RL agent observation.

        Returns:
            Dictionary of state features
        """
        current = self.current_goal

        if not current:
            return {
                'num_goals': len(self.goals),
                'num_open': 0,
                'target_complexity': 0,
                'context_size': 0,
                'tactic_count': self.total_tactics_applied,
                'is_complete': self.is_complete
            }

        # Count goal complexity heuristics
        target_complexity = len(current.target.split())
        context_size = len(current.context)

        # Check for common patterns
        has_quantifiers = any(q in current.target for q in [
                              '∀', '∃', 'forall', 'exists'])
        has_implication = '→' in current.target or '->' in current.target
        has_equality = '=' in current.target

        return {
            'num_goals': len(self.goals),
            'num_open': self.num_open_goals,
            'current_goal_idx': self.current_goal_idx,
            'target': current.target,
            'target_complexity': target_complexity,
            'context_size': context_size,
            'context_types': [v.var_type for v in current.context],
            'has_quantifiers': has_quantifiers,
            'has_implication': has_implication,
            'has_equality': has_equality,
            'tactic_count': self.total_tactics_applied,
            'depth': current.depth,
            'is_complete': self.is_proved(),
            'has_error': self.has_error
        }

    def to_lean_script(self) -> str:
        """
        Convert proof state to Lean script.

        Returns:
            Lean 4 code representing the current proof
        """
        lines = [f"theorem {self.theorem_name} :"]

        # Add global context
        for var in self.global_context:
            lines.append(f"  {var.to_lean()}")

        # Add goals
        for i, goal in enumerate(self.goals):
            if i == 0:
                lines.append(f"  {goal.target} :=")
            else:
                lines.append(f"  -- Goal {goal.goal_id}: {goal.target}")

        # Add tactic sequence
        if self.tactic_sequence:
            lines.append("  by")
            for tactic in self.tactic_sequence:
                lines.append(f"    {tactic}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'theorem_name': self.theorem_name,
            'goals': [g.to_dict() for g in self.goals],
            'global_context': [str(v) for v in self.global_context],
            'current_goal_idx': self.current_goal_idx,
            'tactic_sequence': self.tactic_sequence,
            'total_tactics_applied': self.total_tactics_applied,
            'is_complete': self.is_proved(),
            'has_error': self.has_error,
            'error_message': self.error_message,
            'reward': self.reward,
            'features': self.get_state_features()
        }

    def __repr__(self):
        status = "✓ COMPLETE" if self.is_proved() else "IN PROGRESS"
        if self.has_error:
            status = f"✗ ERROR: {self.error_message}"

        return (f"ProofState({self.theorem_name}, "
                f"{self.num_open_goals} open goals, "
                f"{self.total_tactics_applied} tactics, "
                f"{status})")
