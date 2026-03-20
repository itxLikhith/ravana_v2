#!/usr/bin/env python3
"""
RAVANA v2 — The "Beautiful Lie" Test
Does RAVANA prefer simple truth or seductive overfit?

Setup:
- True model: linear boundary (simple, stable)
- Beautiful lie: 5th-order polynomial that fits noise better short-term
- Over time: simple model wins (stable), complex model fails (overfits)

If RAVANA passes:
- ✅ Rejects complex model despite better short-term fit
- ✅ Chooses stable simple truth
- ✅ Occam penalty protects against overfitting

If RAVANA fails:
- ❌ Chases complex overfitting model
- ❌ Becomes "conspiracy theorist" (fitting noise with elaborate stories)
- ❌ No true Occam discipline

This is the definitive test of epistemic maturity.
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from core import (
    Governor, GovernorConfig,
    ResolutionEngine,
    IdentityEngine,
    StateManager,
    StrategyWithLearning, StrategyLearningLayer, LearningConfig,
    NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics,
    LearnedWorldModel, WorldModelConfig,
    BeliefReasoner, BeliefConfig,
    HypothesisGenerator, GenerationConfig, HypothesisType,
    SurgicalProbeSelector, SurgicalProbeConfig,
    OccamLayer, OccamConfig, DisciplinedBeliefSystem
)


@dataclass
class BeautifulLieConfig:
    """Configuration for the Beautiful Lie test."""
    # True world: simple linear boundary
    true_boundary_intercept: float = 0.70
    true_boundary_slope: float = 0.0001  # Very slow drift
    
    # Beautiful lie: 5th order polynomial with overfit coefficients
    lie_poly_coeffs: List[float] = field(default_factory=lambda: [
        0.75,      # constant (close to true)
        0.001,     # linear term (small drift)
        0.0005,    # quadratic (small curve)
        -0.0001,   # cubic (oscillation)
        0.00001,   # quartic (faster oscillation)
        -0.000001  # quintic (complex wiggle)
    ])
    
    # Test parameters
    total_episodes: int = 300
    log_interval: int = 50
    
    # Hypothesis seeds
    simple_hypothesis_seed: bool = True
    complex_hypothesis_seed: bool = True


class BeautifulLieWorld:
    """
    🎭 THE DECEPTION:
    - True boundary: simple linear (easy to learn, stable long-term)
    - Beautiful lie: complex polynomial (fits short-term noise better)
    
    Initially, the complex model looks better (overfits noise).
    Eventually, the simple model wins (generalizes correctly).
    """
    
    def __init__(self, config: BeautifulLieConfig):
        self.config = config
        self.episode = 0
        
    def get_true_boundary(self, episode: int) -> float:
        """True boundary: simple linear with tiny drift."""
        return (
            self.config.true_boundary_intercept + 
            self.config.true_boundary_slope * episode
        )
    
    def get_lie_boundary(self, episode: int) -> float:
        """Beautiful lie: complex polynomial that overfits."""
        coeffs = self.config.lie_poly_coeffs
        x = episode / 100.0  # Normalized episode
        
        # 5th order polynomial
        lie = (coeffs[0] + 
               coeffs[1] * x + 
               coeffs[2] * x**2 + 
               coeffs[3] * x**3 + 
               coeffs[4] * x**4 + 
               coeffs[5] * x**5)
        
        return np.clip(lie, 0.50, 0.95)
    
    def observe_with_noise(self, true_val: float, noise_std: float = 0.03) -> float:
        """True boundary with observation noise."""
        return np.clip(true_val + np.random.normal(0, noise_std), 0.0, 1.0)
    
    def get_observation(self, episode: int) -> Tuple[float, float, float]:
        """
        Returns: (observed_boundary, true_boundary, lie_boundary)
        """
        true_b = self.get_true_boundary(episode)
        lie_b = self.get_lie_boundary(episode)
        observed = self.observe_with_noise(true_b)
        
        return observed, true_b, lie_b


class BeautifulLiePipeline:
    """
    🧪 TEST PIPELINE:
    RAVANA must choose between simple truth and beautiful lie.
    """
    
    def __init__(
        self,
        manager: StateManager,
        env: BeautifulLieWorld,
        config: BeautifulLieConfig
    ):
        self.manager = manager
        self.env = env
        self.config = config
        
        # Components
        self.world_model = LearnedWorldModel(WorldModelConfig())
        self.probe_selector = SurgicalProbeSelector(SurgicalProbeConfig())
        
        # Disciplined belief system with Occam layer
        belief = BeliefReasoner()
        generator = HypothesisGenerator(GenerationConfig())
        self.disciplined_belief = DisciplinedBeliefSystem(
            belief_reasoner=belief,
            hypothesis_generator=generator,
            occam_config=OccamConfig(
                lambda_penalty=0.4,  # Moderate skepticism
                max_hypotheses=4,
                min_evidence_before_penalty=8
            )
        )
        
        # Tracking
        self.selection_history: List[Dict] = []
        self.truth_tracking: List[float] = []
        self.lie_tracking: List[float] = []
        self.occam_scores: Dict[str, List[float]] = {'simple': [], 'complex': []}
        
        # Results
        self.output_dir = Path("/home/workspace/ravana_v2/results")
        self.output_dir.mkdir(exist_ok=True)
        
    def _simulate_outcome(self, difficulty: float) -> bool:
        """Simulate episode outcome."""
        base_success = 0.65
        success_rate = base_success - (difficulty - 0.3) * 0.3
        return np.random.random() < max(0.2, success_rate)
    
    def train(self) -> Dict[str, Any]:
        """Run the Beautiful Lie test."""
        print("=" * 70)
        print("🎭 THE BEAUTIFUL LIE TEST")
        print("=" * 70)
        print("\nSetup:")
        print(f"  True model: linear boundary (simple, stable)")
        print(f"  Beautiful lie: 5th-order polynomial (complex, seductive)")
        print(f"  Test: Does RAVANA prefer simple truth or complex overfit?")
        print()
        print("🎯 Pass if RAVANA:")
        print("   ✅ Rejects complex model despite short-term fit")
        print("   ✅ Chooses stable simple truth")
        print("   ✅ Occam penalty protects against overfitting")
        print()
        print("🚨 Fail if RAVANA:")
        print("   ❌ Chases complex overfitting model")
        print("   ❌ Becomes 'conspiracy theorist'")
        print("   ❌ No true Occam discipline")
        print("=" * 70)
        
        # Seed hypotheses
        if self.config.simple_hypothesis_seed:
            # Simple hypothesis: constant or slow drift
            self._seed_simple_hypothesis()
        
        if self.config.complex_hypothesis_seed:
            # Complex hypothesis: time-varying with structure
            self._seed_complex_hypothesis()
        
        # Run episodes
        for episode in range(self.config.total_episodes):
            # Get world observation
            observed, true_b, lie_b = self.env.get_observation(episode)
            
            # Step environment (updates hidden state)
            self.env.episode = episode
            
            # Get context for decision
            difficulty = self._compute_difficulty(episode)
            correctness = self._simulate_outcome(difficulty)
            
            # Execute cognitive step
            debug = episode < 20
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=debug
            )
            
            # Score hypotheses with Occam discipline
            scores = self.disciplined_belief.score_all_hypotheses(episode)
            
            # Select best (with pruning)
            best = self.disciplined_belief.select_best(scores)
            
            # Track truth vs lie
            self.truth_tracking.append(true_b)
            self.lie_tracking.append(lie_b)
            
            # Track selection
            if best:
                self.selection_history.append({
                    'episode': episode,
                    'selected_hypothesis': best.hypothesis_id,
                    'selected_complexity': best.complexity,
                    'selected_score': best.occam_score,
                    'true_boundary': true_b,
                    'lie_boundary': lie_b,
                    'penalty_applied': best.penalty_applied
                })
                
                # Track by complexity type
                if best.complexity < 0.3:
                    self.occam_scores['simple'].append(best.occam_score)
                else:
                    self.occam_scores['complex'].append(best.occam_score)
            
            # Periodic logging
            if (episode + 1) % self.config.log_interval == 0:
                self._log_progress(episode + 1, best, true_b, lie_b, scores)
            
            # Generate new hypothesis if needed
            if self.disciplined_belief.should_generate_new(scores, episode):
                new_hyp = self.disciplined_belief.generator.generate_parametric_hypothesis(
                    trigger_type='underperforming_models',
                    context={'episode': episode, 'uncertainty': self.manager.state.dissonance}
                )
                if new_hyp:
                    print(f"\n🌱 EP{episode:03d} | Generated new hypothesis: {new_hyp['type']}")
        
        # Final analysis
        return self._analyze_results()
    
    def _seed_simple_hypothesis(self):
        """Seed a simple (true) hypothesis."""
        # Simple constant model
        simple_hyp = {
            'id': 'simple_linear',
            'type': 'CONSTANT_SLOW_DRIFT',
            'boundary_estimate': 0.70,
            'confidence': 0.6,
            'uncertainty': 0.25,
            'complexity_score': 0.15,  # Low complexity
            'birth_episode': 0,
            'evidence_count': 0
        }
        self.disciplined_belief.belief.hypotheses.append(simple_hyp)
        print("  🌱 Seeded simple hypothesis: linear drift (complexity=0.15)")
    
    def _seed_complex_hypothesis(self):
        """Seed a complex (lie) hypothesis."""
        # Complex polynomial model
        complex_hyp = {
            'id': 'complex_poly',
            'type': 'POLYNOMIAL_OSCILLATION',
            'boundary_estimate': 0.75,
            'confidence': 0.6,
            'uncertainty': 0.25,
            'complexity_score': 0.65,  # High complexity
            'birth_episode': 0,
            'evidence_count': 0
        }
        self.disciplined_belief.belief.hypotheses.append(complex_hyp)
        print("  🌱 Seeded complex hypothesis: 5th-order poly (complexity=0.65)")
    
    def _compute_difficulty(self, episode: int) -> float:
        """Adaptive difficulty."""
        base = 0.5
        cycle = 0.1 * np.sin(2 * np.pi * episode / 100)
        return np.clip(base + cycle, 0.3, 0.8)
    
    def _log_progress(
        self,
        episode: int,
        best: Any,
        true_b: float,
        lie_b: float,
        scores: List
    ):
        """Log progress with Occam analysis."""
        state = self.manager.state
        
        print(f"\nEP{episode:03d} | D={state.dissonance:.3f} | I={state.identity:.3f}")
        
        if best:
            print(f"   Selected: {best.hypothesis_id[:15]:15} | "
                  f"Complexity={best.complexity:.2f} | "
                  f"Score={best.occam_score:.3f} | "
                  f"Penalty={best.penalty_applied:.3f}")
        
        # Show all hypotheses
        print(f"   Hypotheses ({len(scores)}):")
        for s in sorted(scores, key=lambda x: x.occam_score, reverse=True)[:3]:
            simple_marker = "✓" if s.complexity < 0.3 else " "
            print(f"      [{simple_marker}] {s.hypothesis_id[:20]:20} | "
                  f"complexity={s.complexity:.2f} | "
                  f"score={s.occam_score:.3f}")
        
        # Show truth vs lie
        simple_selected = best and best.complexity < 0.3
        truth_marker = "🟢" if simple_selected else "⚪"
        print(f"   {truth_marker} True={true_b:.3f} | Lie={lie_b:.3f} | "
              f"Prefers: {'SIMPLE ✓' if simple_selected else 'COMPLEX'}")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze test results."""
        print("\n" + "=" * 70)
        print("🎭 BEAUTIFUL LIE TEST — FINAL ANALYSIS")
        print("=" * 70)
        
        # Count selections by type
        simple_selections = sum(
            1 for h in self.selection_history 
            if h['selected_complexity'] < 0.3
        )
        complex_selections = len(self.selection_history) - simple_selections
        
        # Calculate preference over time
        early_episodes = self.selection_history[:50]
        late_episodes = self.selection_history[-50:]
        
        early_simple_pref = sum(
            1 for h in early_episodes if h['selected_complexity'] < 0.3
        ) / max(1, len(early_episodes))
        
        late_simple_pref = sum(
            1 for h in late_episodes if h['selected_complexity'] < 0.3
        ) / max(1, len(late_episodes))
        
        # Calculate average scores
        avg_simple_score = np.mean(self.occam_scores['simple']) if self.occam_scores['simple'] else 0
        avg_complex_score = np.mean(self.occam_scores['complex']) if self.occam_scores['complex'] else 0
        
        # Verdict
        print(f"\n📊 RESULTS:")
        print(f"   Simple hypothesis selections: {simple_selections} ({100*simple_selections/max(1,len(self.selection_history)):.1f}%)")
        print(f"   Complex hypothesis selections: {complex_selections} ({100*complex_selections/max(1,len(self.selection_history)):.1f}%)")
        print(f"   Early preference for simple: {100*early_simple_pref:.1f}%")
        print(f"   Late preference for simple: {100*late_simple_pref:.1f}%")
        print(f"   Avg Occam score (simple): {avg_simple_score:.3f}")
        print(f"   Avg Occam score (complex): {avg_complex_score:.3f}")
        
        # The key metric: does preference for simple INCREASE over time?
        learning_simple = late_simple_pref > early_simple_pref
        
        print(f"\n🎯 VERDICT:")
        if late_simple_pref > 0.6 and learning_simple:
            print("   🟢 PASS — RAVANA prefers simple truth over complex lie")
            print("   ✅ Occam discipline working correctly")
            verdict = "PASS"
        elif late_simple_pref < 0.4:
            print("   🔴 FAIL — RAVANA seduced by complex overfit")
            print("   ❌ Conspiracy theorist mode activated")
            verdict = "FAIL"
        else:
            print("   🟡 PARTIAL — RAVANA undecided, needs more evidence")
            print("   ⚠️  Discipline present but not decisive")
            verdict = "PARTIAL"
        
        # Save results
        results = {
            'verdict': verdict,
            'simple_preference_early': early_simple_pref,
            'simple_preference_late': late_simple_pref,
            'simple_selections': simple_selections,
            'complex_selections': complex_selections,
            'avg_occam_score_simple': avg_simple_score,
            'avg_occam_score_complex': avg_complex_score,
            'selection_history': self.selection_history,
            'occam_discipline': self.disciplined_belief.occam.get_discipline_status()
        }
        
        with open(self.output_dir / "beautiful_lie_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {self.output_dir / 'beautiful_lie_results.json'}")
        print("=" * 70)
        
        return results


def main():
    """Run the Beautiful Lie Test."""
    # Create base components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95,
        min_dissonance=0.15,
        max_identity=0.95,
        min_identity=0.10,
    ))
    
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Create Beautiful Lie world
    world_config = BeautifulLieConfig()
    world = BeautifulLieWorld(world_config)
    
    # Create pipeline
    pipeline = BeautifulLiePipeline(manager, world, world_config)
    
    # Run test
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()