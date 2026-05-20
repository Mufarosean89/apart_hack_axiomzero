"""
Lean 4 Environment Interface
Communicates with Lean 4 via subprocess and JSON-RPC.
Provides the step() function for the RL agent.
"""

import logging
import os
import re
import subprocess
import tempfile
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from .proof_state import ProofState, ProofGoal, Variable, GoalStatus
from .tactics import TacticAction, TacticCategory, TacticSpace

logger = logging.getLogger(__name__)


class LeanEnvironment:
    """
    Lean 4 environment for RL proof agent.
    Wraps Lean 4 subprocess and provides step() interface.
    """

    def __init__(self, lean_executable: str = "lean",
                 mathlib_path: str = None,
                 timeout: float = 30.0):
        """
        Initialize Lean 4 environment.

        Args:
            lean_executable: Path to lean executable
            mathlib_path: Path to Mathlib installation
            timeout: Timeout for tactic execution in seconds
        """
        self.lean_executable = lean_executable
        self.mathlib_path = mathlib_path
        self.timeout = timeout
        self.tactic_space = TacticSpace()

        # Lean process (lazy initialization)
        self.lean_process = None
        self.lean_initialized = False

        # Statistics
        self.total_steps = 0
        self.successful_tactics = 0
        self.failed_tactics = 0

    def initialize(self, theorem_statement: str,
                   imports: List[str] = None) -> ProofState:
        """
        Initialize environment with a theorem to prove.

        Args:
            theorem_statement: The theorem statement in Lean
            imports: List of imports (e.g., ['Mathlib.Data.Matrix.Basic'])

        Returns:
            Initial proof state
        """
        # Extract theorem name and type
        theorem_name = self._extract_theorem_name(theorem_statement)
        target_type = self._extract_target_type(theorem_statement)

        # Create initial proof state
        state = ProofState(
            theorem_name=theorem_name,
            goals=[ProofGoal(goal_id=0, target=target_type)]
        )

        # Add imports to global context
        if imports:
            for imp in imports:
                state.global_context.append(
                    Variable(name=f"import_{imp.split('.')[-1]}",
                             var_type=f"import {imp}",
                             is_hypothesis=False)
                )

        self.current_state = state
        return state

    def step(self, tactic: str) -> Tuple[ProofState, float, bool, Dict]:
        """
        Execute one step in the proof environment.

        This is the core RL environment interface:
        - Takes a tactic action
        - Returns new state, reward, done flag, and info

        Args:
            tactic: Tactic string to apply

        Returns:
            Tuple of (new_state, reward, done, info)
        """
        self.total_steps += 1

        # Record tactic
        tactic_action = TacticAction(
            tactic_name=self._extract_tactic_name(tactic),
            tactic_string=tactic,
            category=self._categorize_tactic(tactic)
        )

        try:
            # Execute tactic via Lean
            success, result, error = self._execute_tactic(tactic)

            if success:
                # Parse Lean's response to get new goals
                new_goals = self._parse_lean_response(result)

                # Update proof state
                self._update_state_on_success(tactic_action, new_goals)

                # Calculate reward
                reward = self._calculate_reward(
                    success=True, new_goals=new_goals)
                done = self.current_state.is_proved()

                self.successful_tactics += 1

                info = {
                    'success': True,
                    'new_goals': len(new_goals),
                    'tactic': tactic,
                    'remaining_goals': self.current_state.num_open_goals
                }

            else:
                # Tactic failed
                self._update_state_on_failure(tactic_action, error)

                # Penalty for failed tactic
                reward = self._calculate_reward(success=False, error=error)
                done = True  # Episode ends on failure

                self.failed_tactics += 1

                info = {
                    'success': False,
                    'error': error,
                    'tactic': tactic
                }

        except Exception as e:
            # Unexpected error
            self.current_state.has_error = True
            self.current_state.error_message = str(e)
            reward = -2.0
            done = True
            info = {'success': False, 'error': str(e)}

        return self.current_state, reward, done, info

    def _execute_tactic(self, tactic: str) -> Tuple[bool, str, Optional[str]]:
        """
        Execute a tactic via Lean 4 subprocess.

        Uses lake build or lean --run to compile and check.

        Args:
            tactic: Tactic string

        Returns:
            Tuple of (success, output, error)
        """
        try:
            # Generate Lean file with current proof
            lean_code = self._generate_lean_file(tactic)

            # Write to temporary file (auto-cleaned)
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.lean', delete=False
            ) as tmp:
                tmp.write(lean_code)
                temp_path = Path(tmp.name)

            try:
                # Run Lean
                start_time = time.time()
                result = subprocess.run(
                    [self.lean_executable, "--run", str(temp_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                elapsed = time.time() - start_time

                if result.returncode == 0:
                    return True, result.stdout, None
                else:
                    error_msg = self._parse_lean_error(result.stderr)
                    return False, result.stdout, error_msg
            finally:
                # Clean up temp file and any compiled artifacts
                for suffix in ['', '.ilean', '.olean']:
                    p = temp_path.with_suffix(temp_path.suffix + suffix) if suffix else temp_path
                    if p.exists():
                        p.unlink()

        except subprocess.TimeoutExpired:
            return False, "", f"Tactic execution timed out after {self.timeout}s"
        except Exception as e:
            return False, "", f"Execution error: {str(e)}"

    def _generate_lean_file(self, tactic: str) -> str:
        """
        Generate complete Lean file with current proof state and new tactic.

        Args:
            tactic: New tactic to apply

        Returns:
            Complete Lean code as string
        """
        lines = []

        # Add imports
        for var in self.current_state.global_context:
            if var.var_type.startswith("import "):
                lines.append(var.var_type)

        lines.append("")

        # Add theorem statement
        lines.append(f"theorem {self.current_state.theorem_name} :")
        lines.append(f"  {self.current_state.goals[0].target} :=")

        # Add tactic sequence
        lines.append("  by")
        for prev_tactic in self.current_state.tactic_sequence:
            lines.append(f"    {prev_tactic}")

        # Add new tactic
        lines.append(f"    {tactic}")

        return "\n".join(lines)

    def _parse_lean_response(self, output: str) -> List[ProofGoal]:
        """
        Parse Lean's output to extract remaining goals.

        Args:
            output: Lean's stdout

        Returns:
            List of remaining proof goals
        """
        # In a real implementation, this would parse Lean's JSON-RPC output
        # For now, we'll use a simplified approach

        goals = []

        # Check if proof is complete
        if "proofs generated" in output.lower() or "no goals" in output.lower():
            return goals  # All goals solved

        # Try to extract goal information from output
        # This is simplified - real implementation would parse Lean's infoview protocol
        lines = output.split('\n')
        for line in lines:
            if '⊢' in line or 'goal' in line.lower():
                # Extract target
                target = line.split('⊢')[-1].strip() if '⊢' in line else line
                goal = ProofGoal(
                    goal_id=len(goals),
                    target=target
                )
                goals.append(goal)

        return goals

    def _parse_lean_error(self, stderr: str) -> str:
        """
        Parse Lean error messages.

        Args:
            stderr: Lean's stderr output

        Returns:
            Parsed error message
        """
        # Extract the most relevant error message
        lines = stderr.split('\n')
        for line in lines:
            if 'error' in line.lower():
                return line.strip()

        return stderr.strip() if stderr else "Unknown error"

    def _update_state_on_success(self, tactic_action: TacticAction,
                                 new_goals: List[ProofGoal]):
        """Update proof state after successful tactic."""
        # Apply tactic to current goal
        self.current_state.apply_tactic(
            tactic=tactic_action.tactic_string,
            success=True,
            new_goals=new_goals
        )

        # Move to next goal if current is solved
        if self.current_state.current_goal and \
           self.current_state.current_goal.status == GoalStatus.SOLVED:
            # Find next open goal
            for i, goal in enumerate(self.current_state.goals):
                if goal.status == GoalStatus.OPEN:
                    self.current_state.current_goal_idx = i
                    break

    def _update_state_on_failure(self, tactic_action: TacticAction,
                                 error: Optional[str]):
        """Update proof state after failed tactic."""
        self.current_state.apply_tactic(
            tactic=tactic_action.tactic_string,
            success=False,
            error=error
        )

    def _calculate_reward(self, success: bool, new_goals: List[ProofGoal] = None,
                          error: str = None) -> float:
        """
        Calculate reward for RL agent.

        Reward structure:
        - +1.0 for completing proof
        - +0.1 for making progress (reducing goals)
        - -0.05 per tactic (encourage shorter proofs)
        - -1.0 for tactic failure
        - -0.1 for timeout

        Args:
            success: Whether tactic succeeded
            new_goals: New subgoals created
            error: Error message if failed

        Returns:
            Reward value
        """
        reward = 0.0

        if success:
            # Small cost per tactic
            reward -= 0.05

            # Reward for progress
            if new_goals is not None:
                old_goals = self.current_state.num_open_goals
                new_goal_count = len(new_goals)

                # If we reduced total goals
                if new_goal_count < old_goals:
                    reward += 0.1

            # Check if proof is complete
            if self.current_state.is_proved():
                reward += 1.0  # Large reward for completion
        else:
            # Penalty for failure
            if error and 'timeout' in error.lower():
                reward -= 0.1
            else:
                reward -= 1.0

        return reward

    def _extract_theorem_name(self, statement: str) -> str:
        """Extract theorem name from statement."""
        match = re.search(r'theorem\s+(\w+)', statement)
        if match:
            return match.group(1)
        return "unnamed_theorem"

    def _extract_target_type(self, statement: str) -> str:
        """Extract target type from theorem statement."""
        if ':' in statement:
            parts = statement.split(':', 1)
            if len(parts) > 1:
                return parts[1].strip().rstrip(':=')
        return statement

    def _extract_tactic_name(self, tactic: str) -> str:
        """Extract tactic name from tactic string."""
        return tactic.split()[0] if tactic else "unknown"

    def _categorize_tactic(self, tactic: str) -> TacticCategory:
        """Categorize tactic by name."""
        tactic_name = self._extract_tactic_name(tactic)

        if tactic_name in ['simp', 'ring', 'norm_num']:
            return TacticCategory.SIMPLIFICATION
        elif tactic_name in ['omega', 'linarith', 'decide']:
            return TacticCategory.DECISION
        elif tactic_name in ['have', 'let']:
            return TacticCategory.CONSTRUCTION
        else:
            return TacticCategory.BASIC

    def reset(self, theorem_statement: str = None) -> ProofState:
        """
        Reset environment to initial state.

        Args:
            theorem_statement: New theorem to prove (optional)

        Returns:
            Initial proof state
        """
        if theorem_statement:
            return self.initialize(theorem_statement)
        return self.current_state

    def get_valid_actions(self, state: ProofState = None) -> List[str]:
        """
        Get list of valid actions for current state.

        Args:
            state: Current proof state (uses current_state if None)

        Returns:
            List of valid tactic strings
        """
        if state is None:
            state = self.current_state

        if not state or not state.current_goal:
            return []

        # Get suggested tactics based on goal structure
        suggestions = self.tactic_space.suggest_tactics(
            state.current_goal.target,
            [str(v) for v in state.current_goal.context]
        )

        # Generate concrete tactics
        valid_actions = []
        for tactic_template in suggestions:
            if not tactic_template.parameters:
                # No parameters needed
                valid_actions.append(tactic_template.name)
            else:
                # Generate with placeholder
                valid_actions.append(tactic_template.name)

        return valid_actions

    def close(self):
        """Clean up environment."""
        if self.lean_process:
            self.lean_process.terminate()
            self.lean_process = None

    def __del__(self):
        """Destructor."""
        self.close()
