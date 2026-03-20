#!/usr/bin/env python3
"""
RAVANA v2 — Training Entry Point
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

from core import Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager
from training.pipeline import TrainingPipeline, TrainingConfig


def main():
    """Execute Phase A training."""
    # Create components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95,
        min_dissonance=0.15,
        max_identity=0.95,
        min_identity=0.10,
        dissonance_target=0.45,
        identity_target=0.65,
    ))
    
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    
    # Create state manager (wires everything together)
    manager = StateManager(governor, resolution, identity)
    
    # Create and run pipeline
    config = TrainingConfig(
        total_episodes=1000,  # Start small for testing
        log_interval=100,
        debug_first_n=20,
    )
    
    pipeline = TrainingPipeline(manager, config)
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
