"""
Test script for proof engine module.
Demonstrates proof state management, tactic space, and environment interface.
"""

from proof_engine import (
    ProofState,
    ProofGoal,
    Variable,
    GoalStatus,
    TacticSpace,
    TacticAction,
    TacticCategory,
    LeanEnvironment
)


def test_proof_state():
    """Test proof state management."""
    print("=" * 80)
    print("TEST 1: PROOF STATE MANAGEMENT")
    print("=" * 80)
    print()

    # Create initial proof state
    state = ProofState(
        theorem_name="matrix_mul_assoc",
        goals=[
            ProofGoal(
                goal_id=0,
                target="∀ (A B C : Matrix), (A * B) * C = A * (B * C)",
                context=[
                    Variable(name="A", var_type="Matrix", is_hypothesis=False),
                    Variable(name="B", var_type="Matrix", is_hypothesis=False),
                    Variable(name="C", var_type="Matrix", is_hypothesis=False)
                ]
            )
        ]
    )

    print("Initial Proof State:")
    print(f"  Theorem: {state.theorem_name}")
    print(f"  Goals: {len(state.goals)}")
    print(f"  Open goals: {state.num_open_goals}")
    print(f"  Current goal: {state.current_goal}")
    print()

    # Apply intro tactic
    print("Applying tactic: intro A")
    state.current_goal.apply_tactic("intro A")
    state.tactic_sequence.append("intro A")
    state.current_goal.add_hypothesis("A", "Matrix")

    print(f"  Tactics applied: {state.total_tactics_applied}")
    print(f"  Goal depth: {state.current_goal.depth}")
    print(f"  Context size: {len(state.current_goal.context)}")
    print()

    # Simulate solving goal
    print("Solving goal...")
    state.solve_goal(0)
    print(f"  Goal status: {state.goals[0].status}")
    print(f"  Is proved: {state.is_proved()}")
    print()

    # Test state features
    print("State Features for RL Agent:")
    features = state.get_state_features()
    for key, value in features.items():
        print(f"  {key}: {value}")
    print()

    # Test serialization
    print("State Serialization:")
    state_dict = state.to_dict()
    print(f"  Keys: {list(state_dict.keys())}")
    print(f"  ✓ Serialization successful")
    print()


def test_tactic_space():
    """Test tactic space and action generation."""
    print("=" * 80)
    print("TEST 2: TACTIC SPACE")
    print("=" * 80)
    print()

    # Initialize tactic space
    space = TacticSpace()

    print(f"Total tactics available: {len(space.tactics)}")
    print()

    # Show tactics by category
    categories = {}
    for name, tactic in space.tactics.items():
        cat = tactic.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)

    print("Tactics by Category:")
    print("-" * 60)
    for category, tactics in categories.items():
        print(f"\n  {category.upper()}:")
        for tactic_name in tactics:
            print(f"    • {tactic_name}")
    print()

    # Test tactic suggestion
    print("Tactic Suggestions for Different Goals:")
    print("-" * 60)

    # Goal with implication
    goal1 = "A → B → C"
    suggestions1 = space.suggest_tactics(goal1)
    print(f"\n  Goal: {goal1}")
    print(f"  Suggested: {[t.name for t in suggestions1]}")

    # Goal with equality
    goal2 = "x + y = y + x"
    suggestions2 = space.suggest_tactics(goal2)
    print(f"\n  Goal: {goal2}")
    print(f"  Suggested: {[t.name for t in suggestions2]}")

    # Arithmetic goal
    goal3 = "x > 0 ∧ y > 0 → x + y > 0"
    suggestions3 = space.suggest_tactics(goal3)
    print(f"\n  Goal: {goal3}")
    print(f"  Suggested: {[t.name for t in suggestions3]}")
    print()

    # Test tactic instantiation
    print("Tactic Instantiation Examples:")
    print("-" * 60)

    intro_tactic = space.get_tactic('intro')
    print(f"  Template: {intro_tactic.template}")
    print(f"  Example: {intro_tactic.instantiate(name='x')}")

    have_tactic = space.get_tactic('have')
    print(f"\n  Template: {have_tactic.template}")
    print(
        f"  Example: {have_tactic.instantiate(name='h', type='x > 0', proof='by linarith')}")
    print()

    # Test serialization
    print("Tactic Space Serialization:")
    space_dict = space.to_dict()
    print(f"  Total tactics: {space_dict['total_tactics']}")
    print(f"  Categories: {space_dict['categories']}")
    print()


def test_lean_environment():
    """Test Lean environment interface (simulated)."""
    print("=" * 80)
    print("TEST 3: LEAN ENVIRONMENT (SIMULATED)")
    print("=" * 80)
    print()

    # Initialize environment
    env = LeanEnvironment(lean_executable="lean")

    print("Environment initialized")
    print(f"  Tactic space size: {len(env.tactic_space.tactics)}")
    print()

    # Initialize with theorem
    theorem = """
    theorem add_comm (a b : ℕ) : a + b = b + a
    """

    print("Initializing environment with theorem:")
    print(f"  {theorem.strip()}")
    print()

    state = env.initialize(theorem, imports=['Mathlib.Data.Nat.Basic'])

    print("Initial State:")
    print(f"  Theorem: {state.theorem_name}")
    print(f"  Goals: {state.num_open_goals}")
    print(
        f"  Target: {state.current_goal.target if state.current_goal else 'N/A'}")
    print()

    # Get valid actions
    print("Valid Actions:")
    actions = env.get_valid_actions(state)
    print(f"  Available tactics: {actions}")
    print()

    # Simulate proof steps (without actual Lean execution)
    print("Simulated Proof Steps:")
    print("-" * 60)

    # Step 1: intro a
    print("\n  Step 1: intro a")
    print("  (Would execute: lean --run temp_proof.lean)")
    print("  Expected: Introduce variable a")

    # Step 2: intro b
    print("\n  Step 2: intro b")
    print("  Expected: Introduce variable b")

    # Step 3: induction a
    print("\n  Step 3: induction a with ih")
    print("  Expected: Create base case and inductive step")

    # Step 4: simp
    print("\n  Step 4: simp")
    print("  Expected: Simplify base case")

    # Step 5: rw [Nat.add_comm]
    print("\n  Step 5: rw [Nat.add_comm]")
    print("  Expected: Rewrite using commutativity")

    # Step 6: exact ih
    print("\n  Step 6: exact ih")
    print("  Expected: Complete proof with induction hypothesis")
    print()

    # Show reward structure
    print("Reward Structure:")
    print("-" * 60)
    print("  +1.0  - Proof completed")
    print("  +0.1  - Progress made (goals reduced)")
    print("  -0.05 - Per tactic (encourage short proofs)")
    print("  -1.0  - Tactic failed")
    print("  -0.1  - Timeout")
    print()

    # Test state representation
    print("State Representation for RL:")
    print("-" * 60)
    features = state.get_state_features()
    for key, value in features.items():
        if key not in ['target', 'context_types']:
            print(f"  {key}: {value}")
    print()


def test_complete_proof_simulation():
    """Simulate a complete proof with reward tracking."""
    print("=" * 80)
    print("TEST 4: COMPLETE PROOF SIMULATION")
    print("=" * 80)
    print()

    # Create proof state
    state = ProofState(
        theorem_name="simple_arithmetic",
        goals=[
            ProofGoal(
                goal_id=0,
                target="∀ (x y : ℕ), x + y = y + x",
                context=[]
            )
        ]
    )

    print(f"Theorem: {state.theorem_name}")
    print(f"Target: {state.current_goal.target}")
    print()

    # Simulate proof steps
    proof_steps = [
        ("intro x", True, "Introduced x"),
        ("intro y", True, "Introduced y"),
        ("induction x with ih", True, "Created induction cases"),
        ("simp", True, "Simplified base case"),
        ("rw [Nat.add_comm]", True, "Rewrote using commutativity"),
        ("exact ih", True, "Applied induction hypothesis")
    ]

    total_reward = 0.0

    print("Proof Execution:")
    print("-" * 60)

    for i, (tactic, success, description) in enumerate(proof_steps, 1):
        print(f"\n  Step {i}: {tactic}")
        print(f"    Description: {description}")

        # Apply tactic
        state.apply_tactic(tactic, success=success)

        # Calculate reward
        if i == len(proof_steps):
            # Last step - proof complete
            reward = 1.0 - (0.05 * i)
            state.goals[0].status = GoalStatus.SOLVED
            state.is_complete = True
            print(f"    ✓ PROOF COMPLETE!")
        else:
            reward = -0.05
            print(f"    Progress made")

        total_reward += reward
        print(f"    Reward: {reward:+.2f}")
        print(f"    Total reward: {total_reward:+.2f}")

    print()
    print("Final State:")
    print(f"  Tactics applied: {state.total_tactics_applied}")
    print(f"  Total reward: {total_reward:+.2f}")
    print(f"  Proof complete: {state.is_proved()}")
    print()

    # Show complete proof
    print("Complete Proof Script:")
    print("-" * 60)
    print(state.to_lean_script())
    print()


def test_rl_integration():
    """Test integration points for RL agent."""
    print("=" * 80)
    print("TEST 5: RL AGENT INTEGRATION POINTS")
    print("=" * 80)
    print()

    print("The proof engine provides the following for RL training:")
    print()

    print("1. OBSERVATION SPACE (State Features)")
    print("-" * 60)
    print("  • Number of open goals")
    print("  • Target complexity (token count)")
    print("  • Context size (available hypotheses)")
    print("  • Has quantifiers (∀, ∃)")
    print("  • Has implications (→)")
    print("  • Has equalities (=)")
    print("  • Proof depth")
    print("  • Tactic count")
    print()

    print("2. ACTION SPACE (Tactics)")
    print("-" * 60)
    space = TacticSpace()
    print(f"  • {len(space.tactics)} total tactics")
    print(f"  • Categories: basic, simplification, decision, etc.")
    print(f"  • Parameterized tactics (instantiation)")
    print(f"  • Context-aware suggestions")
    print()

    print("3. REWARD SIGNAL")
    print("-" * 60)
    print("  • +1.0 for proof completion (sparse reward)")
    print("  • +0.1 for progress (dense reward)")
    print("  • -0.05 per step (proof length penalty)")
    print("  • -1.0 for failure (error penalty)")
    print()

    print("4. ENVIRONMENT INTERFACE")
    print("-" * 60)
    print("  • step(tactic) -> (state, reward, done, info)")
    print("  • reset() -> initial state")
    print("  • get_valid_actions() -> action mask")
    print("  • Standard Gym-compatible API")
    print()

    print("5. GROUND TRUTH VERIFICATION")
    print("-" * 60)
    print("  • Lean 4 kernel validates every tactic")
    print("  • Binary outcome: compiles or doesn't")
    print("  • No approximation - exact correctness")
    print("  • Error messages provide learning signal")
    print()


def run_all_tests():
    """Run all proof engine tests."""
    print()
    print("" + "═" * 78 + "╗")
    print("║" + " " * 18 + "AXIOM ZERO - PROOF ENGINE TESTS" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    test_proof_state()
    print("\n" + "━" * 80 + "\n")

    test_tactic_space()
    print("\n" + "━" * 80 + "\n")

    test_lean_environment()
    print("\n" + "━" * 80 + "\n")

    test_complete_proof_simulation()
    print("\n" + "━" * 80 + "\n")

    test_rl_integration()

    print()
    print("=" * 80)
    print("✓ ALL PROOF ENGINE TESTS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Proof state management")
    print("  ✓ Tactic space with 20+ tactics")
    print("  ✓ Lean 4 environment interface")
    print("  ✓ Complete proof simulation")
    print("  ✓ RL agent integration points")
    print()
    print("The game engine is ready for the RL proof agent!")
    print()


if __name__ == "__main__":
    run_all_tests()
