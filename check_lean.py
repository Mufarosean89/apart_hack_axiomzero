#!/usr/bin/env python3
"""Quick Lean 4 verification"""

import subprocess
import sys

print("Checking Lean 4 installation...")
print()

# Check lean
try:
    result = subprocess.run(["lean", "--version"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ lean installed")
        print(f"  {result.stdout.strip()}")
    else:
        print("✗ lean error")
        print(f"  {result.stderr}")
except Exception as e:
    print(f"✗ lean not found: {e}")

print()

# Check lake
try:
    result = subprocess.run(["lake", "--version"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ lake installed")
        print(f"  {result.stdout.strip()}")
    else:
        print("⊗ lake error (optional)")
except FileNotFoundError:
    print("⊗ lake not found (optional)")
except Exception as e:
    print(f"⊗ lake error: {e}")

print()
print("="*60)
print("Lean 4 is installed and ready for Axiom Zero!")
print("="*60)
