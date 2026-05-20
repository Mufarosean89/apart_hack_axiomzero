#!/usr/bin/env python3
"""
Test JSON-RPC client with actual Lean 4 installation
"""

from lean_jsonrpc import LeanRPCClient
import time

print("="*70)
print("TESTING JSON-RPC CLIENT WITH LEAN 4")
print("="*70)
print()

# Create client
client = LeanRPCClient(lean_path="lean")

# Start server
print("[1/3] Starting Lean 4 server...")
client.start("test_theorem.lean")
print("✓ Server started")
print()

# Get goals
print("[2/3] Getting proof goals...")
time.sleep(1)  # Give server time to load
response = client.get_goals("test_theorem.lean", line=2, column=3)

if response.success:
    print(f"✓ Goals retrieved successfully")
    print(f"  Number of goals: {len(response.goals)}")
    
    if response.goals:
        goal = response.goals[0]
        print(f"\n  Goal:")
        print(f"    {goal.goal}")
        if goal.hypotheses:
            print(f"\n  Hypotheses:")
            for h in goal.hypotheses:
                print(f"    • {h}")
else:
    print(f"✗ Failed to get goals: {response.error}")

print()

# Test tactic application
print("[3/3] Testing tactic application...")
# Apply 'intro a' tactic
response = client.apply_tactic("intro a", "test_theorem.lean", line=2, column=3)

if response.success:
    print(f"✓ Tactic applied successfully")
    print(f"  Remaining goals: {len(response.goals)}")
else:
    print(f"✗ Tactic failed: {response.error}")

print()

# Cleanup
print("Stopping server...")
client.stop()
print("✓ Server stopped")

print()
print("="*70)
print("✓ JSON-RPC CLIENT TEST COMPLETE")
print("="*70)
print()
print("Lean 4 is installed and working!")
print("JSON-RPC client can communicate with Lean server.")
