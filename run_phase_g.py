#!/usr/bin/env python3
"""
RAVANA v2 — Phase G: Active Epistemology
Active Discovery Test
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

import random
import numpy as np
from typing import Dict, Any, List

from core import (
    Governor, GovernorConfig,
    ResolutionEngine,
    IdentityEngine,
    StateManager,
    StrategyLayer, StrategyConfig, ExplorationMode, BehavioralContext,
    StrategyLearningLayer, LearningConfig,
    StrategyWithLearning,
    IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective,
    BeliefReasoner, BeliefConfig,
    ActiveEpistemology, VoIConfig, InformationGainMethod
)


class PartialWorldEnvironment:
    """Creates a 'consistent but partial' world."""
    
    def __init__(self, true_boundary: float = 0.75, alternative_boundary: float = 0.95):
        self.true_boundary = true_boundary
        self.alternative_boundary = alternative_boundary
        self.episode_count = 0
        self.ambiguity_zone_low = min(true_boundary, alternative_boundary) - 0.05
        self.ambiguity_zone_high = min(true_boundary, alternative_boundary) + 0.05
        self.distinction_threshold = max(true_boundary, alternative_boundary) + 0.02
        
    def step(self, episode: int, dissonance: float) -> Dict[str, Any]:
        """Generate observation that tests hypothesis disambiguation."""
        self.episode_count = episode
        in_ambiguity_zone = self.ambiguity_zone_low <= dissonance <= self.ambiguity_zone_high
        
        if in_ambiguity_zone:
            difficulty = 0.5 + random.gauss(0, 0.1)
            observation_quality = "ambiguous"
        elif dissonance > self.distinction_threshold:
            difficulty = 0.9
            observation_quality = "distinguishing"
        else:
            difficulty = 0.3
            observation_quality = "consistent"
        
        return {
            'difficulty': np.clip(difficulty, 0.1, 0.95),
            'true_boundary': self.true_boundary,
            'alternative_boundary': self.alternative_boundary,
            'observation_quality': observation_quality,
            'in_ambiguity_zone': in_ambiguity_zone
        }
    
    def get_true_boundary(self) -> float:
        return self.true_boundary


class PhaseGTrainingPipeline:
    """Phase G: Active Epistemology training with discovery testing."""
    
    def __init__(self, state_manager, intent_strategy, belief_reasoner, 
                 active_epistemology, env, config=None):
        self.manager = state_manager
        self.intent_strategy = intent_strategy
        self.belief = belief_reasoner
        self.epistemology = active_epistemology
        self.env = env
        self.config = config or {'total_episodes': 1000, 'log_interval': 100}
        self.probes_when_uncertain = 0
        self.total_uncertain_episodes = 0
        
    def _simulate_outcome(self, difficulty: float) -> bool:
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        return random.random() < success_rate
    
    def train(self) -> Dict[str, Any]:
        """Execute active epistemology training."""
        print("=" * 70)
        print("🧠 RAVANA v2 — Phase G: Active Epistemology")
        print("=" * 70)
        print("Test: Does RAVANA intentionally act to resolve uncertainty?")
        print("=" * 70)
        
        # Manually spawn competing hypotheses
        self.belief.hypotheses[2] = {
            'id': 2,
            'belief': self.env.alternative_boundary,
            'confidence': 0.4,
            'birth_episode': 0,
            'evidence_count': 0
        }
        
        print(f"\n🎭 ACTIVE DISCOVERY TEST")
        print(f"   True boundary: {self.env.true_boundary}")
        print(f"   Alternative hypothesis: {self.env.alternative_boundary}")
        print(f"   Initial: RAVANA maintains BOTH hypotheses")
        print(f"   Test: Will RAVANA probe to disambiguate?\n")
        
        for episode in range(self.config['total_episodes']):
            pre_state = {'dissonance': self.manager.state.dissonance}
            
            # Get world state
            world_state = self.env.step(episode, pre_state['dissonance'])
            
            # Use active epistemology to select action
            action, metadata = self.epistemology.act_and_learn(
                episode, pre_state, ExplorationMode.EXPLORE_SAFE
            )
            
            # Track uncertainty and probing
            belief_state = self.belief.get_belief_state()
            hypotheses = belief_state.get('hypotheses', {})
            
            if len(hypotheses) >= 2:
                sorted_hyps = sorted(hypotheses.items(), key=lambda x: x[1]['confidence'], reverse=True)
                gap = sorted_hyps[0][1]['confidence'] - sorted_hyps[1][1]['confidence']
                is_uncertain = gap < 0.2
                
                if is_uncertain:
                    self.total_uncertain_episodes += 1
                    if metadata.get("reason") == "hypothesis_driven_probe":
                        self.probes_when_uncertain += 1
            
            # Execute step
            difficulty = world_state['difficulty']
            correctness = self._simulate_outcome(difficulty)
            step_record = self.manager.step(correctness=correctness, difficulty=difficulty, debug=False)
            
            post_state = {'dissonance': self.manager.state.dissonance}
            
            # Update belief with evidence
            mode = ExplorationMode.EXPLORE_SAFE if action == "conservative_stabilize" else ExplorationMode.EXPLORE_AGGRESSIVE
            self.belief.observe(episode, pre_state, post_state, mode, self.env.get_true_boundary())
            
            # Logging
            if (episode + 1) % self.config['log_interval'] == 0 or metadata.get("reason") == "hypothesis_driven_probe":
                marker = "🔬" if metadata.get("reason") == "hypothesis_driven_probe" else "  "
                print(f"{marker}EP{episode+1:04d} | D={post_state['dissonance']:.2f} | "
                      f"Belief={self.belief.current_belief:.2f}±{self.belief.current_uncertainty:.2f} | "
                      f"Hyps={len(hypotheses)} | "
                      f"Action={action[:15]:15s} | "
                      f"Reason={metadata.get('reason', 'default')[:20]}")
        
        # Final results
        print("\n" + "=" * 70)
        print("📊 PHASE G RESULTS: Active Epistemology Test")
        print("=" * 70)
        
        active_discovery_rate = self.probes_when_uncertain / max(1, self.total_uncertain_episodes)
        
        print(f"\n🎯 UNCERTAINTY BEHAVIOR:")
        print(f"   Total uncertain episodes: {self.total_uncertain_episodes}")
        print(f"   Intentional probes: {self.probes_when_uncertain}")
        print(f"   Active discovery rate: {active_discovery_rate:.1%}")
        
        print(f"\n🧠 BELIEF EVOLUTION:")
        print(f"   Final belief: {self.belief.current_belief:.3f} (true: {self.env.true_boundary})")
        print(f"   Final uncertainty: {self.belief.current_uncertainty:.3f}")
        print(f"   Remaining hypotheses: {len(hypotheses)}")
        
        print(f"\n🔬 ACTIVE EPISTEMOLOGY:")
        epistemic_summary = self.epistemology.get_epistemic_status()
        print(f"   Total experiments: {epistemic_summary['action_selection']['total_probes']}")
        print(f"   Uncertainties resolved: {epistemic_summary['uncertainties_resolved']}")
        print(f"   Avg info gain: {epistemic_summary['avg_info_gain']:.4f}")
        
        # Verdict
        print("\n" + "=" * 70)
        if active_discovery_rate > 0.3:
            print("🏆 VERDICT: ACTIVE DISCOVERER 🚀")
            print("   RAVANA intentionally acts to resolve uncertainty")
            print("   when two hypotheses are close in confidence.")
        elif active_discovery_rate > 0.1:
            print("⚖️ VERDICT: EMERGING CURIOSITY")
            print("   RAVANA sometimes probes, but not consistently")
        else:
            print("🛑 VERDICT: PASSIVE THINKER")
            print("   RAVANA lives with uncertainty without acting")
        print("=" * 70)
        
        return {
            'total_episodes': self.config['total_episodes'],
            'final_belief': self.belief.current_belief,
            'true_boundary': self.env.true_boundary,
            'active_discovery_rate': active_discovery_rate,
            'total_probes': self.probes_when_uncertain,
            'epistemic_status': epistemic_summary,
            'verdict': 'active_discoverer' if active_discovery_rate > 0.3 else 'passive_thinker'
        }


def main():
    # Components
    governor = Governor(GovernorConfig(max_dissonance=0.95, min_dissonance=0.15, max_identity=0.95, min_identity=0.10))
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyLearningLayer(LearningConfig())
    learning_wrapper = StrategyWithLearning(strategy, learning)
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning_wrapper, intent)
    
    belief = BeliefReasoner(BeliefConfig())
    voi_config = VoIConfig(info_gain_weight=0.4, uncertainty_threshold=0.15, min_probe_interval=30)
    epistemology = ActiveEpistemology(belief, voi_config)
    
    env = PartialWorldEnvironment(true_boundary=0.75, alternative_boundary=0.90)
    
    config = {'total_episodes': 1000, 'log_interval': 100}
    pipeline = PhaseGTrainingPipeline(manager, intent_strategy, belief, epistemology, env, config)
    
    results = pipeline.train()
    return results


if __name__ == "__main__":
    main()
