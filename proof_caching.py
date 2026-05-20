#!/usr/bin/env python3
"""
Proof Caching & Transfer Learning
Caches solved proofs and reuses tactics for similar problems.
"""

import json
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CachedProof:
    """A cached proof with metadata."""
    theorem_id: str
    theorem_statement: str
    proof_tactics: List[str]
    proof_length: int
    search_time: float
    difficulty: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            'theorem_id': self.theorem_id,
            'theorem_statement': self.theorem_statement,
            'proof_tactics': self.proof_tactics,
            'proof_length': self.proof_length,
            'search_time': self.search_time,
            'difficulty': self.difficulty,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'CachedProof':
        return CachedProof(**d)


class ProofCache:
    """
    Cache for solved proofs with similarity-based retrieval.
    
    Features:
    - Fast lookup by theorem ID
    - Similarity search for new theorems
    - Tactic pattern extraction
    - Transfer learning suggestions
    """
    
    def __init__(self, cache_file: str = "proof_cache.json"):
        """
        Initialize proof cache.
        
        Args:
            cache_file: Path to cache file
        """
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, CachedProof] = {}
        self.tactic_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing cache
        self._load()
    
    def add_proof(self, proof: CachedProof):
        """
        Add proof to cache.
        
        Args:
            proof: CachedProof to cache
        """
        self.cache[proof.theorem_id] = proof
        
        # Extract tactic patterns
        pattern_key = self._extract_pattern(proof.proof_tactics)
        self.tactic_patterns[pattern_key].append(proof.theorem_id)
        
        # Save to disk
        self._save()
        
        logger.info(f"Cached proof: {proof.theorem_id} ({proof.proof_length} tactics)")
    
    def get_proof(self, theorem_id: str) -> Optional[CachedProof]:
        """
        Get cached proof by ID.
        
        Args:
            theorem_id: Theorem identifier
            
        Returns:
            CachedProof or None
        """
        return self.cache.get(theorem_id)
    
    def find_similar_proofs(
        self,
        theorem_statement: str,
        top_k: int = 5
    ) -> List[Tuple[float, CachedProof]]:
        """
        Find similar proofs using statement similarity.
        
        Args:
            theorem_statement: New theorem statement
            top_k: Number of similar proofs to return
            
        Returns:
            List of (similarity_score, CachedProof) tuples
        """
        similarities = []
        
        for theorem_id, proof in self.cache.items():
            similarity = self._compute_similarity(
                theorem_statement,
                proof.theorem_statement
            )
            similarities.append((similarity, proof))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:top_k]
    
    def suggest_tactics(
        self,
        theorem_statement: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Suggest tactics based on similar proofs.
        
        Args:
            theorem_statement: New theorem statement
            top_k: Number of suggestions
            
        Returns:
            List of suggested tactics
        """
        similar = self.find_similar_proofs(theorem_statement, top_k=top_k)
        
        if not similar:
            return []
        
        # Aggregate tactics from similar proofs
        tactic_scores = defaultdict(float)
        
        for similarity, proof in similar:
            for i, tactic in enumerate(proof.proof_tactics):
                # Weight by similarity and position (earlier tactics more important)
                weight = similarity * (1.0 / (i + 1))
                tactic_scores[tactic] += weight
        
        # Sort by score
        sorted_tactics = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [tactic for tactic, score in sorted_tactics[:top_k]]
    
    def get_common_patterns(self) -> Dict[str, int]:
        """
        Get common tactic patterns across proofs.
        
        Returns:
            Dictionary of pattern -> count
        """
        patterns = {}
        
        for pattern_key, theorem_ids in self.tactic_patterns.items():
            patterns[pattern_key] = len(theorem_ids)
        
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {'size': 0}
        
        avg_length = sum(p.proof_length for p in self.cache.values()) / len(self.cache)
        avg_time = sum(p.search_time for p in self.cache.values()) / len(self.cache)
        
        by_difficulty = defaultdict(int)
        for proof in self.cache.values():
            by_difficulty[proof.difficulty] += 1
        
        return {
            'size': len(self.cache),
            'avg_proof_length': avg_length,
            'avg_search_time': avg_time,
            'by_difficulty': dict(by_difficulty),
            'num_patterns': len(self.tactic_patterns),
        }
    
    def _extract_pattern(self, tactics: List[str]) -> str:
        """
        Extract abstract pattern from tactic sequence.
        
        Args:
            tactics: List of tactics
            
        Returns:
            Pattern string
        """
        # Abstract away specific parameters
        pattern = []
        for tactic in tactics:
            # Remove parameters (e.g., "rw [Nat.add_comm]" -> "rw")
            base = tactic.split()[0] if tactic else tactic
            pattern.append(base)
        
        return "->".join(pattern)
    
    def _compute_similarity(self, stmt1: str, stmt2: str) -> float:
        """
        Compute similarity between two theorem statements.
        
        Simple token-based similarity (can be enhanced with embeddings).
        
        Args:
            stmt1: First statement
            stmt2: Second statement
            
        Returns:
            Similarity score [0, 1]
        """
        # Tokenize
        tokens1 = set(stmt1.lower().split())
        tokens2 = set(stmt2.lower().split())
        
        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                
                for theorem_id, proof_dict in data.items():
                    self.cache[theorem_id] = CachedProof.from_dict(proof_dict)
                
                logger.info(f"Loaded {len(self.cache)} proofs from cache")
                
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def _save(self):
        """Save cache to disk."""
        try:
            data = {
                theorem_id: proof.to_dict()
                for theorem_id, proof in self.cache.items()
            }
            
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache: {e}")


class TransferLearningAgent:
    """
    Uses cached proofs to accelerate new proof search.
    """
    
    def __init__(self, cache: ProofCache):
        """
        Initialize transfer learning agent.
        
        Args:
            cache: Proof cache
        """
        self.cache = cache
    
    def initialize_search(self, theorem_statement: str) -> Dict[str, Any]:
        """
        Initialize MCTS search with knowledge from similar proofs.
        
        Args:
            theorem_statement: New theorem statement
            
        Returns:
            Search initialization parameters
        """
        # Get similar proofs
        similar = self.cache.find_similar_proofs(theorem_statement, top_k=5)
        
        if not similar:
            return {'use_default': True}
        
        # Aggregate tactics
        suggested_tactics = self.cache.suggest_tactics(theorem_statement, top_k=10)
        
        # Get common patterns
        patterns = self.cache.get_common_patterns()
        top_patterns = list(patterns.keys())[:3]
        
        # Compute prior probabilities for MCTS
        tactic_priors = {}
        total_score = 0
        
        for i, tactic in enumerate(suggested_tactics):
            score = 1.0 / (i + 1)  # Higher score for more common tactics
            tactic_priors[tactic] = score
            total_score += score
        
        # Normalize
        for tactic in tactic_priors:
            tactic_priors[tactic] /= total_score
        
        return {
            'use_default': False,
            'suggested_tactics': suggested_tactics[:5],
            'tactic_priors': tactic_priors,
            'top_patterns': top_patterns,
            'num_similar_proofs': len(similar),
        }
    
    def adapt_strategy(
        self,
        theorem_statement: str,
        current_depth: int,
        failed_tactics: List[str]
    ) -> Dict[str, Any]:
        """
        Adapt search strategy based on cache knowledge.
        
        Args:
            theorem_statement: Current theorem
            current_depth: Current search depth
            failed_tactics: Tactics that have failed
            
        Returns:
            Adapted strategy
        """
        # Get suggestions
        suggestions = self.cache.suggest_tactics(theorem_statement, top_k=10)
        
        # Remove failed tactics
        valid_suggestions = [t for t in suggestions if t not in failed_tactics]
        
        # If deep in search, try different approach
        if current_depth > 10:
            # Look for shorter proofs in cache
            short_proofs = [
                p for p in self.cache.cache.values()
                if p.proof_length < 5
            ]
            
            if short_proofs:
                # Prioritize tactics from short proofs
                short_tactics = []
                for proof in short_proofs:
                    short_tactics.extend(proof.proof_tactics[:3])
                
                valid_suggestions = list(dict.fromkeys(short_tactics))  # Unique, preserve order
        
        return {
            'next_tactics': valid_suggestions[:3],
            'should_backtrack': current_depth > 15,
            'alternative_strategies': len(valid_suggestions) > 3,
        }


class ProofCachingSystem:
    """
    Complete proof caching system with transfer learning.
    """
    
    def __init__(self, cache_file: str = "proof_cache.json"):
        """
        Initialize caching system.
        
        Args:
            cache_file: Path to cache file
        """
        self.cache = ProofCache(cache_file=cache_file)
        self.transfer_agent = TransferLearningAgent(self.cache)
    
    def solve_with_caching(
        self,
        theorem_id: str,
        theorem_statement: str,
        solver_func
    ) -> Dict[str, Any]:
        """
        Solve theorem with caching support.
        
        Args:
            theorem_id: Theorem identifier
            theorem_statement: Theorem statement
            solver_func: Function to solve theorem
            
        Returns:
            Solution result
        """
        # Check cache first
        cached = self.cache.get_proof(theorem_id)
        
        if cached:
            logger.info(f"Cache hit: {theorem_id}")
            return {
                'source': 'cache',
                'proof_tactics': cached.proof_tactics,
                'proof_length': cached.proof_length,
                'search_time': 0,  # Instant from cache
                'success': True,
            }
        
        # Initialize with transfer learning
        init_params = self.transfer_agent.initialize_search(theorem_statement)
        
        # Solve theorem
        start_time = time.time()
        result = solver_func(theorem_statement, init_params)
        search_time = time.time() - start_time
        
        if result['success']:
            # Cache the proof
            proof = CachedProof(
                theorem_id=theorem_id,
                theorem_statement=theorem_statement,
                proof_tactics=result['proof_tactics'],
                proof_length=len(result['proof_tactics']),
                search_time=search_time,
                difficulty=result.get('difficulty', 'medium'),
                metadata=result.get('metadata', {}),
            )
            
            self.cache.add_proof(proof)
            
            logger.info(
                f"Proof solved and cached: {theorem_id} "
                f"({len(result['proof_tactics'])} tactics, {search_time:.2f}s)"
            )
        
        return {
            'source': 'solved',
            **result,
            'search_time': search_time,
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_statistics()
    
    def export_cache(self, filepath: str = None):
        """Export cache to JSON file."""
        filepath = filepath or "proof_cache_export.json"
        
        data = {
            'statistics': self.cache.get_statistics(),
            'proofs': {
                tid: proof.to_dict()
                for tid, proof in self.cache.cache.items()
            },
            'patterns': self.cache.get_common_patterns(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Cache exported to: {filepath}")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("PROOF CACHING & TRANSFER LEARNING")
    print("="*70)
    print()
    
    # Initialize caching system
    system = ProofCachingSystem()
    
    # Simulate solving some theorems
    theorems = [
        {
            'id': 'add_comm',
            'statement': '∀ (a b : ℕ), a + b = b + a',
            'difficulty': 'easy'
        },
        {
            'id': 'mul_one',
            'statement': '∀ (n : ℕ), n * 1 = n',
            'difficulty': 'easy'
        },
        {
            'id': 'list_append',
            'statement': '∀ (xs ys : List ℕ), length (xs ++ ys) = length xs + length ys',
            'difficulty': 'medium'
        },
    ]
    
    # Simulate solver
    def mock_solver(statement: str, init_params: dict) -> dict:
        import random
        return {
            'success': True,
            'proof_tactics': ['intro', 'induction', 'simp', 'ring'],
            'difficulty': 'medium',
        }
    
    # Solve and cache
    for theorem in theorems:
        result = system.solve_with_caching(
            theorem['id'],
            theorem['statement'],
            mock_solver
        )
        
        print(f"✓ {theorem['id']}: {result['source']} ({result.get('proof_length', 0)} tactics)")
    
    # Get statistics
    stats = system.get_cache_stats()
    
    print(f"\nCache Statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Avg Proof Length: {stats['avg_proof_length']:.1f}")
    print(f"  Avg Search Time: {stats['avg_search_time']:.2f}s")
    print(f"  Patterns: {stats['num_patterns']}")
    
    # Test transfer learning
    print(f"\nTransfer Learning Demo:")
    new_theorem = "∀ (x y z : ℕ), (x + y) + z = x + (y + z)"
    
    init = system.transfer_agent.initialize_search(new_theorem)
    print(f"  New theorem: {new_theorem}")
    print(f"  Suggested tactics: {init.get('suggested_tactics', [])}")
    print(f"  Similar proofs: {init.get('num_similar_proofs', 0)}")
    
    print("\n" + "="*70)
    print("✓ PROOF CACHING SYSTEM COMPLETE")
    print("="*70)
