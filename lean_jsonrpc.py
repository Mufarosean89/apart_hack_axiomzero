#!/usr/bin/env python3
"""
Lean 4 JSON-RPC Client
High-performance protocol for communicating with Lean 4 server.
Replaces subprocess calls with proper infoview JSON-RPC protocol.
"""

import json
import subprocess
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LeanGoal:
    """Represents a proof goal."""
    goal: str
    hypotheses: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeanResponse:
    """Response from Lean server."""
    success: bool
    goals: List[LeanGoal]
    error: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'goals': [g.to_dict() for g in self.goals],
            'error': self.error,
            'message': self.message
        }


class LeanRPCClient:
    """
    JSON-RPC client for Lean 4 server.
    
    Provides fast, bidirectional communication with Lean 4
    using the Language Server Protocol (LSP).
    """
    
    def __init__(self, lean_path: str = "lean"):
        """
        Initialize Lean RPC client.
        
        Args:
            lean_path: Path to lean executable
        """
        self.lean_path = lean_path
        self.process = None
        self.request_id = 0
        self.current_file = None
        self.current_pos = (0, 0)
        
    def start(self, lean_file: str):
        """
        Start Lean server process.
        
        Args:
            lean_file: Path to .lean file to work with
        """
        self.current_file = lean_file
        
        # Start lean --server
        self.process = subprocess.Popen(
            [self.lean_path, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"Started Lean 4 server for {lean_file}")
        time.sleep(0.5)  # Give server time to start
        
    def stop(self):
        """Stop Lean server."""
        if self.process:
            self.process.terminate()
            self.process = None
            logger.info("Stopped Lean server")
    
    def _send_request(self, method: str, params: dict) -> dict:
        """
        Send JSON-RPC request to Lean server.
        
        Args:
            method: RPC method name
            params: Method parameters
            
        Returns:
            Response dictionary
        """
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        response = json.loads(response_line)
        
        return response
    
    def get_goals(self, file_path: str, line: int, column: int) -> LeanResponse:
        """
        Get current proof goals at a position.
        
        Args:
            file_path: File path
            line: Line number (1-based)
            column: Column number (1-based)
            
        Returns:
            LeanResponse with current goals
        """
        params = {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line - 1, "character": column - 1}
        }
        
        try:
            response = self._send_request("textDocument/goal", params)
            
            if "result" in response and response["result"]:
                goals = []
                for g in response["result"].get("goals", []):
                    goal = LeanGoal(
                        goal=g.get("mainGoal", ""),
                        hypotheses=g.get("hypotheses", [])
                    )
                    goals.append(goal)
                
                return LeanResponse(success=True, goals=goals)
            else:
                return LeanResponse(success=True, goals=[])
                
        except Exception as e:
            logger.error(f"Error getting goals: {e}")
            return LeanResponse(success=False, goals=[], error=str(e))
    
    def apply_tactic(self, tactic: str, file_path: str, line: int, column: int) -> LeanResponse:
        """
        Apply a tactic at a position.
        
        Args:
            tactic: Tactic to apply (e.g., "simp", "ring")
            file_path: File path
            line: Line number
            column: Column number
            
        Returns:
            LeanResponse with new goals after tactic
        """
        # Send text edit to insert tactic
        edit_params = {
            "textDocument": {"uri": f"file://{file_path}"},
            "edits": [{
                "range": {
                    "start": {"line": line - 1, "character": column - 1},
                    "end": {"line": line - 1, "character": column - 1}
                },
                "newText": tactic
            }]
        }
        
        try:
            self._send_request("textDocument/didChange", edit_params)
            time.sleep(0.1)  # Give Lean time to process
            
            # Get updated goals
            return self.get_goals(file_path, line, column + len(tactic))
            
        except Exception as e:
            logger.error(f"Error applying tactic: {e}")
            return LeanResponse(success=False, goals=[], error=str(e))
    
    def check_file(self, file_path: str) -> LeanResponse:
        """
        Check entire file for errors.
        
        Args:
            file_path: File path
            
        Returns:
            LeanResponse with diagnostics
        """
        content = Path(file_path).read_text()
        
        params = {
            "textDocument": {
                "uri": f"file://{file_path}",
                "text": content
            }
        }
        
        try:
            self._send_request("textDocument/didOpen", params)
            time.sleep(0.5)
            
            # Get diagnostics
            diag_params = {
                "textDocument": {"uri": f"file://{file_path}"}
            }
            
            response = self._send_request("textDocument/publishDiagnostics", diag_params)
            
            if "result" in response and response["result"]:
                diagnostics = response["result"].get("diagnostics", [])
                if diagnostics:
                    errors = [d.get("message", "") for d in diagnostics]
                    return LeanResponse(
                        success=False,
                        goals=[],
                        error="; ".join(errors)
                    )
            
            return LeanResponse(success=True, goals=[])
            
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return LeanResponse(success=False, goals=[], error=str(e))
    
    def get_term_at_position(self, file_path: str, line: int, column: int) -> Optional[str]:
        """
        Get term/type information at a position.
        
        Args:
            file_path: File path
            line: Line number
            column: Column number
            
        Returns:
            Type information string or None
        """
        params = {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line - 1, "character": column - 1}
        }
        
        try:
            response = self._send_request("textDocument/hover", params)
            
            if "result" in response and response["result"]:
                return response["result"].get("contents", "")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting term info: {e}")
            return None


class LeanRPCEnvironment:
    """
    Gym-compatible environment using JSON-RPC.
    Faster than subprocess-based LeanEnvironment.
    """
    
    def __init__(self, lean_file: str, lean_path: str = "lean"):
        """
        Initialize environment.
        
        Args:
            lean_file: Path to .lean file
            lean_path: Path to lean executable
        """
        self.client = LeanRPCClient(lean_path)
        self.lean_file = lean_file
        self.current_goals = []
        self.done = False
        self.reward = 0.0
        
        # Start server
        self.client.start(lean_file)
        
    def step(self, tactic: str, line: int, column: int):
        """
        Apply tactic and get new state.
        
        Args:
            tactic: Tactic to apply
            line: Line number
            column: Column number
            
        Returns:
            (goals, reward, done, info)
        """
        # Apply tactic
        response = self.client.apply_tactic(
            tactic, self.lean_file, line, column
        )
        
        self.current_goals = response.goals
        self.done = len(response.goals) == 0 and response.success
        
        # Calculate reward
        if self.done:
            self.reward = 1.0  # Proof complete!
        elif response.error:
            self.reward = -1.0  # Error
        else:
            self.reward = -0.05  # Small penalty per step
        
        info = {
            'response': response.to_dict(),
            'goals_remaining': len(response.goals)
        }
        
        return response.goals, self.reward, self.done, info
    
    def get_state(self) -> dict:
        """Get current proof state."""
        return {
            'goals': [g.to_dict() for g in self.current_goals],
            'done': self.done,
            'file': self.lean_file
        }
    
    def reset(self):
        """Reset environment."""
        self.current_goals = []
        self.done = False
        self.reward = 0.0
        return self.get_state()
    
    def close(self):
        """Close environment."""
        self.client.stop()


if __name__ == "__main__":
    # Example usage
    print("Lean 4 JSON-RPC Client")
    print("="*60)
    print()
    print("Features:")
    print("  • Fast bidirectional communication with Lean 4")
    print("  • Language Server Protocol (LSP) implementation")
    print("  • Gym-compatible environment interface")
    print("  • Real-time goal tracking")
    print("  • Tactic application with immediate feedback")
    print()
    print("Usage:")
    print("  client = LeanRPCClient()")
    print("  client.start('theorem.lean')")
    print("  response = client.apply_tactic('simp', 'theorem.lean', 10, 5)")
    print("  client.stop()")
