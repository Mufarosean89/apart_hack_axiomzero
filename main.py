#!/usr/bin/env python3
"""
Axiom Zero - Main Entry Point
Complete pipeline: Python/PyTorch → Verified Lean 4
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main entry point for Axiom Zero."""
    parser = argparse.ArgumentParser(
        description="Axiom Zero: AlphaZero-style proof automation for Python/PyTorch"
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input Python file or code string"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.lean",
        help="Output Lean 4 file (default: output.lean)"
    )

    parser.add_argument(
        "--mode",
        choices=["parse", "compile", "prove", "full"],
        default="full",
        help="Pipeline mode (default: full)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AXIOM ZERO - Python to Verified Lean 4 Compiler")
    print("=" * 70)
    print()

    # Step 1: Parse input
    print("[1/4] Parsing input...")
    try:
        from ast_extractor import parse_to_ir

        input_path = Path(args.input)
        if input_path.exists():
            code = input_path.read_text()
            print(f"   Loaded: {args.input}")
        else:
            code = args.input
            print(f"   Using code string")

        ir = parse_to_ir(code)
        print(
            f"   ✓ Parsed: {ir.total_functions} functions, {ir.total_loops} loops")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1

    if args.mode == "parse":
        print("\n✓ Parse mode complete")
        return 0

    # Step 2: Abstract interpretation
    print("\n[2/4] Running abstract interpretation...")
    try:
        from abstract_interpreter import run_abstract_interpretation

        abstract_state = run_abstract_interpretation(ir)
        print(f"   ✓ Analyzed: {len(abstract_state.function_envs)} functions")
        print(f"   ✓ Facts: {len(abstract_state.shape_facts)} shape facts")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1

    if args.mode == "compile":
        print("\n✓ Compile mode complete (without Lean generation)")
        return 0

    # Step 3: Generate Lean skeleton
    print("\n[3/4] Generating Lean 4 skeleton...")
    try:
        from compiler.ir_to_lean import IRtoLeanCompiler

        compiler = IRtoLeanCompiler()
        skeleton = compiler.compile(ir)

        print(f"   ✓ Generated: {skeleton.total_functions} declarations")
        print(
            f"   ✓ Holes: {skeleton.total_holes} ({skeleton.simple_holes} simple, {skeleton.complex_holes} complex)")

        # Write skeleton
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(skeleton.to_string())
        print(f"   ✓ Written to: {args.output}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if args.mode == "prove":
        print("\n✓ Prove mode complete (skeleton generated, hole filling requires Lean 4)")
        return 0

    # Step 4: Hole filling (optional)
    print("\n[4/4] Hole filling...")
    print("   Note: Full proof generation requires Lean 4 installation")
    print("   Install with: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
    print()
    print("   Skeleton with holes generated:")
    print(
        f"   • Simple holes ({skeleton.simple_holes}): Auto-fill with simp/ring/omega")
    print(
        f"   • Complex holes ({skeleton.complex_holes}): Require MCTS + RL agent")
    print()

    print("=" * 70)
    print("✓ COMPILATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Output: {args.output}")
    print(f"Functions: {skeleton.total_functions}")
    print(f"Proof holes: {skeleton.total_holes}")
    print()
    print("Next steps:")
    print("  1. Install Lean 4: https://leanprover.github.io")
    print("  2. Run: lean --run {args.output}")
    print("  3. Fill holes manually or use RL agent")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
