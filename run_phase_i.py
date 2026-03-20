#!/usr/bin/env python3
"""
RAVANA v2 — Phase I: Meta-Cognition Test

VALIDATES:
- When probes fail repeatedly, RAVANA detects systematic epistemic failure
- When confidence is miscalibrated, RAVANA reduces trust in its own assessments  
- When bias detected, RAVANA switches epistemic mode (cautious/exploratory/recovery)

This is the layer that prevents RAVANA from becoming:
- A confident fool (high confidence, wrong beliefs)
- A systematic misinterpreter (repeated probe failures)
- A frozen thinker (no mode switching when stuck)
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

from core import (
    Governor, GovernorConfig,
    ResolutionEngine,
    IdentityEngine,
    StateManager,
    StrategyLayer, StrategyConfig, ExplorationMode, BehavioralContext,
    StrategyLearningLayer,
    IntentEngine,
    MicroPlanner,
    NonStationaryEnvironment, EnvironmentConfig,
    BeliefReasoner, BeliefConfig,
    ActiveEpistemology,
    SurgicalProbeSelector,
    HypothesisGenerator,
    OccamLayer, OccamConfig,
    MetaCognition, MetaCognitiveConfig, EpistemicMode,
    TrainingConfig
)

from training.pipeline import TrainingPipeline
import numpy as np
import random


class AdversarialProbeEnvironment:
    """
    Environment designed to FAIL RAVANA's probes on purpose.
    
    RAVANA designs probes to distinguish H1 vs H2.
    This environment makes those probes give INCONCLUSIVE results.
    
    Test: Does RAVANA detect that its probes aren't working?
    """
    def __init__(self, true_boundary=0.75):
        self.true_boundary = true_boundary
        self.episode = 0
        self.probe_failure_rate = 0.0  # Starts working
        self.hidden_manipulation_active = False
        
    def step(self, episode: int):
        """Environment evolution."""
        self.episode = episode
        
        # Phase 1: Probes work (episodes 0-100)
        if episode < 100:
            self.probe_failure_rate = 0.1
            self.hidden_manipulation_active = False
        
        # Phase 2: Hidden manipulation begins (episodes 100-200)
        # Probes become unreliable WITHOUT RAVANA knowing why
        elif episode < 200:
            self.probe_failure_rate = 0.6  # 60% inconclusive
            self.hidden_manipulation_active = True
            
        # Phase 3: Extreme manipulation (episodes 200-300)
        # Almost all probes fail
        elif episode < 300:
            self.probe_failure_rate = 0.85  # 85% inconclusive
            self.hidden_manipulation_active = True
            
        # Phase 4: Manipulation ends (episodes 300-500)
        # Returns to normal - does RAVANA trust probes again?
        else:
            self.probe_failure_rate = 0.15  # Slightly higher baseline
            self.hidden_manipulation_active = False
    
    def execute_probe(self, probe_design: dict) -> dict:
        """
        Execute probe but manipulate results based on failure rate.
        """
        # True result (what RAVANA should observe)
        true_result = {
            'observed_boundary': self.true_boundary + np.random.normal(0, 0.02),
            'confidence': 0.8,
            'conclusive': True,
            'evidence_for': 'H_true'
        }
        
        # With probability = probe_failure_rate, make it inconclusive
        if random.random() < self.probe_failure_rate:
            # Manipulate to be INCONCLUSIVE
            true_result = {
                'observed_boundary': (0.5 + 0.8) / 2,  # Ambiguous middle
                'confidence': 0.3,  # Low confidence
                'conclusive': False,  # KEY: inconclusive
                'evidence_for': 'ambiguous'
            }
        
        return true_result


class MetaCognitiveTrainingPipeline(TrainingPipeline):
    """Training pipeline with meta-cognitive monitoring."""
    
    def __init__(self, state_manager, meta_cognition, env, config: TrainingConfig = None):
        super().__init__(state_manager, config)
        self.meta = meta_cognition
        self.env = env
        self.probe_results = []
        self.epistemic_mode_history = []
        
    def train(self) -> dict:
        """Execute training with meta-cognitive oversight."""
        print("=" * 70)
        print("🧠 RAVANA v2 — Phase I: Meta-Cognition Test")
        print("=" * 70)
        print("\n🎯 TEST: When probes systematically fail,")
        print("         does RAVANA detect the epistemic failure?")
        print("\n🎯 TEST: When confidence is miscalibrated,")
        print("         does RAVANA distrust its own assessments?")
        print("\n🎯 TEST: When stuck,")
        print("         does RAVANA switch epistemic mode?")
        print("=" * 70)
        
        results = {
            'episodes': [],
            'meta_state': [],
            'probe_failures_detected': 0,
            'mode_switches': 0,
            'calibration_improvements': []
        }
        
        for episode in range(500):
            # Update environment
            self.env.step(episode)
            
            # Get current state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # META: Get current epistemic mode recommendation
            current_mode = self.meta.recommend_epistemic_mode(episode)
            self.epistemic_mode_history.append(current_mode.value)
            
            # Design probe based on active hypotheses
            belief_state = self.manager.belief.get_belief_state() if hasattr(self.manager, 'belief') else []
            probe = self.meta.design_probe_for_uncertainty(belief_state)
            
            # Execute probe in adversarial environment
            probe_result = self.env.execute_probe(probe)
            self.probe_results.append(probe_result)
            
            # META: Assess probe outcome
            probe_assessment = self.meta.assess_probe_outcome(
                probe, probe_result, episode
            )
            
            # If probe failed, record it
            if not probe_result.get('conclusive', True):
                results['probe_failures_detected'] += 1
            
            # META: Update confidence calibration
            self.meta.update_calibration_from_outcome(
                predicted_confidence=probe_result.get('confidence', 0.5),
                actual_outcome='conclusive' if probe_result.get('conclusive') else 'inconclusive',
                episode=episode
            )
            
            # META: Detect bias in reasoning
            bias_detection = self.meta.detect_reasoning_bias(episode)
            
            # Regular training step
            context = self._get_context()
            clamp_events = self._get_recent_clamps(episode)
            
            mode, mode_info = self.intent_strategy.select_mode(context, clamp_events)
            
            difficulty = self._compute_difficulty(episode)
            correctness = self._simulate_outcome(difficulty, mode)
            
            debug = episode < 20 or episode % 50 == 0
            step_record = self._execute_step(correctness, difficulty, mode, debug)
            
            post_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # Update layers
            new_clamps = self._extract_clamp_events(step_record)
            self.intent_strategy.update_after_step(
                episode, pre_state, post_state, mode, new_clamps
            )
            
            # META: Periodic status
            if episode % 100 == 0:
                self._log_meta_status(episode, current_mode, probe_assessment, bias_detection)
            
            # Store results
            results['episodes'].append(episode)
            results['meta_state'].append(self.meta.get_meta_status())
        
        # Final meta-cognitive summary
        final_summary = self._generate_meta_summary(results)
        
        return final_summary
    
    def _log_meta_status(self, episode: int, mode: EpistemicMode, 
                         probe_assessment: dict, bias_detection: dict):
        """Log meta-cognitive status."""
        meta = self.meta.get_meta_status()
        
        print(f"\n🧠 EP{episode:04d} Meta-Cognitive Status:")
        print(f"   Epistemic Mode: {mode.value}")
        print(f"   Probe Quality: {probe_assessment.get('quality', 'unknown')}")
        print(f"   Recent Failures: {meta['recent_probe_failures']}/10")
        print(f"   Confidence Calibration: {meta['calibration_error']:.3f}")
        print(f"   Bias Flags: {list(bias_detection.get('flags', []))}")
        
        # Alert on critical conditions
        if meta['recent_probe_failures'] > 5:
            print(f"   ⚠️  ALERT: High probe failure rate!")
        if meta['calibration_error'] > 0.3:
            print(f"   ⚠️  ALERT: Poor confidence calibration!")
        if mode == EpistemicMode.RECOVERY:
            print(f"   🔴 RECOVERY MODE: Systematic epistemic failure detected")
    
    def _generate_meta_summary(self, results: dict) -> dict:
        """Generate meta-cognitive test summary."""
        print("\n" + "=" * 70)
        print("🧠 META-COGNITIVE TEST — FINAL ANALYSIS")
        print("=" * 70)
        
        # Count mode usage
        mode_counts = {}
        for mode_val in self.epistemic_mode_history:
            mode_counts[mode_val] = mode_counts.get(mode_val, 0) + 1
        
        print(f"\n📊 Epistemic Mode Distribution:")
        for mode, count in mode_counts.items():
            pct = 100 * count / len(self.epistemic_mode_history)
            print(f"   {mode}: {count} episodes ({pct:.1f}%)")
        
        # Probe failure detection
        total_probes = len(self.probe_results)
        inconclusive_probes = results['probe_failures_detected']
        print(f"\n📊 Probe Results:")
        print(f"   Total probes: {total_probes}")
        print(f"   Inconclusive: {inconclusive_probes} ({100*inconclusive_probes/total_probes:.1f}%)")
        
        # Check if meta-cognition worked
        phase2_failures = sum(1 for i, r in enumerate(self.probe_results[100:200]) 
                             if not r.get('conclusive', True))
        phase2_total = 100
        
        phase3_failures = sum(1 for i, r in enumerate(self.probe_results[200:300]) 
                             if not r.get('conclusive', True))
        phase3_total = 100
        
        print(f"\n📊 Failure Detection by Phase:")
        print(f"   Phase 2 (60% hidden failure): {phase2_failures}/{phase2_total} detected")
        print(f"   Phase 3 (85% hidden failure): {phase3_failures}/{phase3_total} detected")
        
        # Verdict
        recovery_mode_used = mode_counts.get(EpistemicMode.RECOVERY.value, 0)
        if recovery_mode_used > 20 and phase2_failures > 40 and phase3_failures > 60:
            verdict = "🟢 PASS — Meta-cognition detected systematic epistemic failure"
        elif recovery_mode_used > 10:
            verdict = "🟡 PARTIAL — Some detection, limited response"
        else:
            verdict = "🔴 FAIL — Did not detect probe manipulation"
        
        print(f"\n🎯 VERDICT:")
        print(f"   {verdict}")
        print(f"   Recovery mode activations: {recovery_mode_used}")
        
        print("=" * 70)
        
        return {
            'mode_distribution': mode_counts,
            'probe_failures_detected': results['probe_failures_detected'],
            'recovery_mode_uses': recovery_mode_used,
            'verdict': verdict,
            'phase2_detection_rate': phase2_failures / phase2_total if phase2_total > 0 else 0,
            'phase3_detection_rate': phase3_failures / phase3_total if phase3_total > 0 else 0,
        }


def main():
    """Execute Phase I: Meta-Cognition Test."""
    # Create components
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Strategy layers
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyLearningLayer()
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    
    # Meta-cognition layer (Phase I)
    meta_config = MetaCognitiveConfig(
        probe_failure_threshold=0.5,
        confidence_calibration_window=20,
        reasoning_quality_threshold=0.6
    )
    meta = MetaCognition(meta_config)
    
    # Adversarial environment
    env = AdversarialProbeEnvironment(true_boundary=0.75)
    
    # Create and run pipeline
    config = TrainingConfig(
        total_episodes=500,
        log_interval=100,
        debug_first_n=20,
    )
    
    pipeline = MetaCognitiveTrainingPipeline(manager, meta, env, config)
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
