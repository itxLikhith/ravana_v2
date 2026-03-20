#!/usr/bin/env python3
"""
RAVANA v2 — Meta² Training Entry Point
Live training with Meta²-Cognition layer active.

When systematic failures detected, Meta²:
  1. Issues epistemic critique
  2. Expands hypothesis space
  3. Logs epiphany events
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

sys.path.insert(0, '/home/workspace/ravana_v2')

from core.governor import Governor, GovernorConfig
from core.resolution import ResolutionEngine
from core.identity import IdentityEngine
from core.state import StateManager
from core.hypothesis_generation import HypothesisGenerator, HypothesisType, GenerationConfig
from core.meta2_cognition import Meta2CognitionEngine, Meta2Config
from core.meta2_integration import Meta2IntegratedGenerator, Meta2GenerationConfig


@dataclass
class Meta2TrainingConfig:
    """Training configuration with Meta² enabled."""
    total_episodes: int = 1000
    log_interval: int = 50
    checkpoint_interval: int = 500
    debug_first_n: int = 20
    
    # Scenario injection: periodically force "impossible" situations
    # to trigger Meta² epiphanies
    inject_adversarial_scenarios: bool = True
    adversarial_interval: int = 100  # Every N episodes, inject surprise
    
    # Difficulty schedule
    initial_difficulty: float = 0.3
    max_difficulty: float = 0.9
    difficulty_ramp_episodes: int = 500


class Meta2TrainingPipeline:
    """
    Training pipeline with live Meta²-Cognition.
    
    Key difference from base pipeline: Meta² monitors the hypothesis
    generator and intervenes when systematic failure is detected.
    """
    
    def __init__(self, state_manager, meta2_generator, config: Meta2TrainingConfig = None):
        self.manager = state_manager
        self.meta2_gen = meta2_generator
        self.config = config or Meta2TrainingConfig()
        
        # Track epiphanies and critiques
        self.epiphanies: List[Dict] = []
        self.critiques: List[Dict] = []
        self.space_expansions: List[Dict] = []
        
        # Episode tracking for scenario injection
        self.episode_counter: int = 0
        
        # Output
        self.output_dir = Path("/home/workspace/ravana_v2/results")
        self.output_dir.mkdir(exist_ok=True)
    
    def _compute_difficulty(self, episode: int) -> float:
        """Adaptive difficulty with occasional surprises."""
        base_difficulty = self.config.initial_difficulty
        
        if episode >= self.config.difficulty_ramp_episodes:
            base_difficulty = self.config.max_difficulty
        else:
            progress = episode / self.config.difficulty_ramp_episodes
            base_difficulty = self.config.initial_difficulty + \
                (self.config.max_difficulty - self.config.initial_difficulty) * progress
        
        # Inject adversarial difficulty spikes to trigger Meta²
        if self.config.inject_adversarial_scenarios:
            if episode % self.config.adversarial_interval in [90, 91, 92]:
                # Sudden spike that current hypotheses can't explain
                return min(1.0, base_difficulty + 0.3)
        
        return base_difficulty
    
    def _simulate_outcome(self, difficulty: float, episode: int) -> bool:
        """Simulate episode outcome with occasional systematic failures."""
        # Base success rate
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        
        # Inject systematic failure patterns during adversarial windows
        # This is designed to trigger Meta²'s "space inadequate" detection
        if self.config.inject_adversarial_scenarios:
            if episode % self.config.adversarial_interval == 91:
                # Force failure: reality doesn't match any hypothesis
                return False
            if episode % self.config.adversarial_interval == 92:
                # Another forced failure
                return False
        
        return np.random.random() < max(0.1, min(0.9, success_rate))
    
    def _get_current_context(self) -> Dict[str, float]:
        """Extract current state for Meta² monitoring."""
        return {
            'dissonance': self.manager.state.dissonance,
            'identity': self.manager.state.identity,
            'wisdom': self.manager.state.accumulated_wisdom,
        }
    
    def train(self) -> Dict[str, Any]:
        """Execute Meta²-enabled training run."""
        print(f"=" * 70)
        print(f"RAVANA v2 — Meta² Training (Phase I² Live Integration)")
        print(f"Total episodes: {self.config.total_episodes:,}")
        print(f"Meta²-Cognition: ACTIVE")
        print(f"Epiphany trigger: Space inadequacy detection")
        print(f"=" * 70)
        
        start_time = time.time()
        
        for episode in range(self.config.total_episodes):
            self.episode_counter = episode
            
            # Compute difficulty (with adversarial injection)
            difficulty = self._compute_difficulty(episode)
            
            # Simulate outcome
            correctness = self._simulate_outcome(difficulty, episode)
            
            # Execute cognitive step (GOVERNOR-GATED)
            debug = episode < self.config.debug_first_n
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=debug
            )
            
            # Update Meta² with current state
            context = self._get_current_context()
            prediction_error = abs(
                step_record['post_dissonance'] - step_record['pre_dissonance']
            ) if 'post_dissonance' in step_record else 0.0
            
            # Monitor Meta² state
            meta2_status = self.meta2_gen.get_meta2_status()
            
            # Check for epiphany
            if meta2_status.get('epiphanies', 0) > len(self.epiphanies):
                new_epiphany = {
                    'episode': episode,
                    'trigger': 'space_inadequacy_detected',
                    'prev_space': meta2_status.get('type_distribution', {}),
                }
                self.epiphanies.append(new_epiphany)
                print(f"\n  🎆 EPIPHANY at EP{episode}: Meta² expanded hypothesis space!")
            
            # Periodic logging
            if (episode + 1) % self.config.log_interval == 0:
                self._log_progress(episode + 1, step_record, difficulty, meta2_status)
            
            # Hard assertions (debug only)
            if debug:
                self._assert_state_valid()
        
        # Final summary
        elapsed = time.time() - start_time
        summary = self._generate_summary(elapsed)
        
        return summary
    
    def _log_progress(self, episode: int, record: Dict, difficulty: float, meta2_status: Dict):
        """Log training progress with Meta² metrics."""
        state = self.manager.state
        
        # Check for recent critiques
        recent_critiques = sum(1 for c in self.critiques 
                              if c['episode'] > episode - self.config.log_interval)
        
        critique_indicator = f"[{recent_critiques} critiques]" if recent_critiques > 0 else ""
        
        print(f"EP{episode:,}/{self.config.total_episodes:,} | "
              f"D={state.dissonance:.3f} | "
              f"I={state.identity:.3f} | "
              f"W={state.accumulated_wisdom:.2f} | "
              f"Mode:{record['mode'][:3]} | "
              f"Diff:{difficulty:.2f} | "
              f"Meta²:{meta2_status.get('total_hypotheses', 0)}h {critique_indicator}")
    
    def _assert_state_valid(self):
        """Hard assertions for debugging."""
        state = self.manager.state
        config = self.manager.governor.config
        
        assert state.dissonance <= config.max_dissonance + 0.01, \
            f"DISSONANCE CEILING BREACH: {state.dissonance} > {config.max_dissonance}"
        
        assert state.dissonance >= config.min_dissonance - 0.01, \
            f"DISSONANCE FLOOR BREACH: {state.dissonance} < {config.min_dissonance}"
        
        assert state.identity >= config.min_identity - 0.01, \
            f"IDENTITY FLOOR BREACH: {state.identity} < {config.min_identity}"
    
    def _generate_summary(self, elapsed: float) -> Dict[str, Any]:
        """Generate training summary with Meta² metrics."""
        status = self.manager.get_status()
        meta2_status = self.meta2_gen.get_meta2_status()
        clamp_metrics = self.manager.governor.get_clamp_metrics()
        
        summary = {
            "total_episodes": self.config.total_episodes,
            "elapsed_seconds": elapsed,
            "final_state": status["state"],
            "governor_stats": status["governor"],
            "resolution_stats": status["resolution"],
            "identity_stats": status["identity"],
            "clamp_metrics": clamp_metrics,
            # Meta²-specific metrics
            "meta2_metrics": {
                "total_hypotheses_generated": meta2_status.get('total_generated', 0),
                "currently_active": meta2_status.get('currently_active', 0),
                "hypothesis_type_distribution": meta2_status.get('type_distribution', {}),
                "epiphanies_observed": len(self.epiphanies),
                "critiques_issued": len(self.critiques),
            },
            "epiphany_events": self.epiphanies,
        }
        
        # Save main summary
        with open(self.output_dir / "meta2_training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate report
        report = self._generate_report(summary, elapsed)
        with open(self.output_dir / "meta2_training_report.txt", "w") as f:
            f.write(report)
        
        # Print summary
        print(f"\n{'=' * 70}")
        print(f"Meta² Training complete: {elapsed:.1f}s")
        print(f"Final: D={self.manager.state.dissonance:.3f} I={self.manager.state.identity:.3f}")
        print(f"Epiphanies: {len(self.epiphanies)}")
        print(f"Hypotheses generated: {meta2_status.get('total_generated', 0)}")
        print(f"{'=' * 70}")
        print(report)
        
        return summary
    
    def _generate_report(self, summary: Dict, elapsed: float) -> str:
        """Generate human-readable Meta² training report."""
        lines = [
            "=" * 70,
            "META²-COGNITION TRAINING REPORT",
            "=" * 70,
            "",
            f"Total Episodes: {summary['total_episodes']:,}",
            f"Training Time: {elapsed:.1f}s",
            "",
            "FINAL STATE:",
            f"  Dissonance: {summary['final_state'].get('dissonance', 0):.3f}",
            f"  Identity: {summary['final_state'].get('identity', 0):.3f}",
            f"  Wisdom: {summary['final_state'].get('wisdom', 0):.2f}",
            "",
            "META²-COGNITION METRICS:",
            f"  Hypotheses generated: {summary['meta2_metrics']['total_hypotheses_generated']}",
            f"  Currently active: {summary['meta2_metrics']['currently_active']}",
            f"  Epiphanies observed: {summary['meta2_metrics']['epiphanies_observed']}",
            f"  Critiques issued: {summary['meta2_metrics']['critiques_issued']}",
            "",
            "HYPOTHESIS TYPE DISTRIBUTION:",
        ]
        
        for htype, count in summary['meta2_metrics']['hypothesis_type_distribution'].items():
            lines.append(f"  {htype}: {count}")
        
        if summary['epiphany_events']:
            lines.extend([
                "",
                "EPIPHANY EVENTS:",
            ])
            for ep in summary['epiphany_events']:
                lines.append(f"  Episode {ep['episode']}: {ep['trigger']}")
        else:
            lines.extend([
                "",
                "EPIPHANY EVENTS: None (hypothesis space adequate)",
            ])
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def main():
    """Execute Meta²-enabled training."""
    print("Initializing Meta² Training Pipeline...")
    
    # Create base components
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
    
    # Create state manager
    manager = StateManager(governor, resolution, identity)
    
    # Create Meta²-integrated hypothesis generator
    base_config = GenerationConfig()
    meta2_config = Meta2Config(
        space_inadequacy_threshold=0.65,
        sustained_failure_window=30,
        failure_rate_threshold=0.50,
    )
    gen_config = Meta2GenerationConfig()
    
    # Create base generator and Meta² engine
    base_gen = HypothesisGenerator(config=base_config)
    meta2_engine = Meta2CognitionEngine(config=meta2_config)
    
    meta2_gen = Meta2IntegratedGenerator(
        base_generator=base_gen,
        meta2_engine=meta2_engine,
        config=gen_config
    )
    
    # Create and run Meta² pipeline
    config = Meta2TrainingConfig(
        total_episodes=500,  # Shorter run for demo
        log_interval=50,
        debug_first_n=10,
        inject_adversarial_scenarios=True,
        adversarial_interval=100,
    )
    
    pipeline = Meta2TrainingPipeline(manager, meta2_gen, config)
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
