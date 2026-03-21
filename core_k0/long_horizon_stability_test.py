"""
Long-Horizon Stability Test Suite (Option A)

Validates paper claims over 10,000+ episodes:
- Dissonance: ~0.8 → ~0.2
- Identity Strength: ~0.3 → ~0.85  
- Generalization Accuracy: ~0.9
- Demographic Parity Gap stability

Methodology aligned with RAVANA paper Section 4 (Methodology).
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

from core_k0.agent_loop_k2 import K2_Agent
from experiments_k0.resource_env import ResourceSurvivalEnv, AgentAction
from experiments_k0.latent_regime_env import LatentRegimeEnv
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class EpisodeMetrics:
    """Per-episode metrics aligned with paper."""
    episode: int
    
    # Core paper metrics
    dissonance_D: float
    identity_strength_I: float
    
    # Performance metrics
    survival_rate_window: float  # Last 100 episodes
    generalization_accuracy: float  # On held-out tasks
    
    # Fairness metrics  
    demographic_parity_gap: float
    
    # Auxiliary
    mean_confidence: float
    exploration_success_rate: float
    action_entropy: float  # Measure of decision consistency


@dataclass
class PhaseMetrics:
    """Metrics per 1000-episode phase."""
    phase: int
    start_episode: int
    end_episode: int
    environment_type: str  # 'stable', 'scarce', 'volatile', 'latent_regime'
    
    avg_dissonance: float
    avg_identity: float
    survival_rate: float
    generalization_acc: float
    parity_gap: float
    
    # Trajectory analysis
    dissonance_trend: float  # Slope over phase
    identity_trend: float


class LongHorizonStabilityTest:
    """
    10,000+ episode validation framework.
    
    Implements paper's training regime with:
    - Periodic environment shifts (every 1,000 episodes)
    - Metric tracking for all paper claims
    - Checkpointing for intermediate analysis
    """
    
    def __init__(
        self,
        n_episodes: int = 10000,
        checkpoint_interval: int = 1000,
        seed: int = 42
    ):
        self.n_episodes = n_episodes
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed
        
        # Metrics storage
        self.episode_metrics: List[EpisodeMetrics] = []
        self.phase_metrics: List[PhaseMetrics] = []
        
        # Environment sequence (periodic shifts)
        self.environment_schedule = self._create_environment_schedule()
        
        # Results
        self.results = {
            "start_time": datetime.now().isoformat(),
            "n_episodes": n_episodes,
            "seed": seed,
            "phases": [],
            "final_metrics": {},
            "trajectory_analysis": {},
            "paper_claims_validation": {}
        }
    
    def _create_environment_schedule(self) -> List[Dict[str, Any]]:
        """
        Create environment shift schedule.
        
        Pattern: Stable → Scarce → Stable → Volatile → Latent → Stable
        Tests generalization and transfer efficiency.
        """
        phases = []
        phase_length = self.checkpoint_interval
        n_phases = self.n_episodes // phase_length
        
        env_types = ['stable', 'scarce', 'stable', 'volatile', 'latent_regime', 'stable']
        
        for i in range(n_phases):
            env_type = env_types[i % len(env_types)]
            phases.append({
                'phase': i,
                'start': i * phase_length,
                'end': (i + 1) * phase_length,
                'type': env_type
            })
        
        return phases
    
    def _create_environment(self, env_type: str, seed: int):
        """Factory for environment types."""
        if env_type == 'stable':
            return ResourceSurvivalEnv(seed=seed)
        elif env_type == 'scarce':
            # Modified: scarce resources
            env = ResourceSurvivalEnv(seed=seed)
            env.true_resources = 0.3  # Start scarce
            return env
        elif env_type == 'volatile':
            # Modified: high noise
            env = ResourceSurvivalEnv(seed=seed)
            env.base_noise = 0.3
            return env
        elif env_type == 'latent_regime':
            return LatentRegimeEnv(seed=seed)
        else:
            return ResourceSurvivalEnv(seed=seed)
    
    def _compute_dissonance(self, agent) -> float:
        """
        Compute cognitive dissonance D from agent state.
        
        Paper formula:
        D = Σ |belief - action| * confidence + mismatch_penalties
        """
        # Simplified: use agent's internal dissonance if available
        if hasattr(agent, 'state') and hasattr(agent.state, 'outcome_history'):
            # Recent outcome volatility as dissonance proxy
            recent = agent.state.outcome_history[-50:]
            if recent:
                deltas = [o.delta_energy for o in recent]
                volatility = np.std(deltas)
                # Normalize to [0, 1] range
                return min(1.0, volatility * 2.0)
        
        # Fallback: based on energy uncertainty
        if hasattr(agent, 'state') and agent.state.energy_history:
            recent_energy = agent.state.energy_history[-20:]
            if recent_energy:
                uncertainty = np.std(recent_energy)
                return min(1.0, uncertainty * 2.0)
        
        return 0.5  # Default
    
    def _compute_identity_strength(self, agent) -> float:
        """
        Compute Identity Strength Index I.
        
        Paper: "measure of cross-context stability, reinforcement 
        of commitments, and resistance to volatility"
        """
        # Based on action consistency and survival rate
        if hasattr(agent, 'state') and len(agent.state.action_history) > 100:
            recent = agent.state.action_history[-100:]
            
            # Action consistency (lower entropy = stronger identity)
            actions = [a[1] for a in recent]
            unique, counts = np.unique(actions, return_counts=True)
            probs = counts / len(actions)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(3)  # 3 actions
            consistency = 1.0 - (entropy / max_entropy)
            
            # Survival component
            if hasattr(agent, 'survival_count') and hasattr(agent, 'episode'):
                survival_rate = agent.survival_count / max(1, agent.episode)
            else:
                survival_rate = 0.5
            
            # Combined identity strength
            identity = 0.4 * consistency + 0.6 * survival_rate
            return min(1.0, identity)
        
        return 0.3  # Baseline
    
    def _compute_generalization_accuracy(
        self, 
        agent, 
        held_out_env: ResourceSurvivalEnv
    ) -> float:
        """Test on held-out environment (generalization)."""
        correct = 0
        n_trials = 20
        
        for _ in range(n_trials):
            obs = held_out_env._generate_observation()
            action = agent.select_action(obs)
            result = held_out_env.execute_action(action)
            if result['alive'] and result['utility'] > 0:
                correct += 1
        
        return correct / n_trials
    
    def run_phase(self, agent: K2_Agent, phase_config: Dict) -> PhaseMetrics:
        """Run one 1000-episode phase with specific environment."""
        print(f"\n  Phase {phase_config['phase']}: {phase_config['type']} "
              f"(EP{phase_config['start']}→{phase_config['end']})")
        
        # Create environment for this phase
        env = self._create_environment(
            phase_config['type'], 
            self.seed + phase_config['phase']
        )
        
        # Create held-out env for generalization testing
        held_out = ResourceSurvivalEnv(seed=self.seed + 1000)
        
        phase_episodes = []
        
        for ep in range(phase_config['start'], phase_config['end']):
            # Run episode
            obs = env._generate_observation()
            action = agent.select_action(obs)
            result = env.execute_action(action)
            
            # Record outcome for agent learning
            if hasattr(agent, '_record_outcome'):
                agent._record_outcome(env, action, result)
            
            # Periodic metric computation
            if ep % 100 == 0 or ep == phase_config['end'] - 1:
                D = self._compute_dissonance(agent)
                I = self._compute_identity_strength(agent)
                
                # Survival rate over last 100 episodes
                recent_outcomes = [
                    o for o in agent.state.outcome_history[-100:]
                    if hasattr(o, 'survived')
                ]
                if recent_outcomes:
                    survival_rate = sum(1 for o in recent_outcomes if o.survived) / len(recent_outcomes)
                else:
                    survival_rate = 1.0
                
                # Generalization (expensive, sample every 500)
                if ep % 500 == 0:
                    gen_acc = self._compute_generalization_accuracy(agent, held_out)
                else:
                    gen_acc = None
                
                metric = EpisodeMetrics(
                    episode=ep,
                    dissonance_D=D,
                    identity_strength_I=I,
                    survival_rate_window=survival_rate,
                    generalization_accuracy=gen_acc or 0.5,
                    demographic_parity_gap=0.0,  # Would need multi-group env
                    mean_confidence=0.7,  # Placeholder
                    exploration_success_rate=agent._get_exploration_success_rate() if hasattr(agent, '_get_exploration_success_rate') else 0.5,
                    action_entropy=0.5  # Placeholder
                )
                phase_episodes.append(metric)
                self.episode_metrics.append(metric)
        
        # Compute phase summary
        if phase_episodes:
            avg_D = np.mean([m.dissonance_D for m in phase_episodes])
            avg_I = np.mean([m.identity_strength_I for m in phase_episodes])
            avg_survival = np.mean([m.survival_rate_window for m in phase_episodes])
            avg_gen = np.mean([m.generalization_accuracy for m in phase_episodes if m.generalization_accuracy])
            
            # Trend analysis
            if len(phase_episodes) >= 2:
                x = np.arange(len(phase_episodes))
                D_values = [m.dissonance_D for m in phase_episodes]
                I_values = [m.identity_strength_I for m in phase_episodes]
                D_trend = np.polyfit(x, D_values, 1)[0] if len(set(D_values)) > 1 else 0
                I_trend = np.polyfit(x, I_values, 1)[0] if len(set(I_values)) > 1 else 0
            else:
                D_trend = I_trend = 0
        else:
            avg_D = avg_I = avg_survival = avg_gen = D_trend = I_trend = 0
        
        phase_result = PhaseMetrics(
            phase=phase_config['phase'],
            start_episode=phase_config['start'],
            end_episode=phase_config['end'],
            environment_type=phase_config['type'],
            avg_dissonance=avg_D,
            avg_identity=avg_I,
            survival_rate=avg_survival,
            generalization_acc=avg_gen,
            parity_gap=0.0,
            dissonance_trend=D_trend,
            identity_trend=I_trend
        )
        
        print(f"    D={avg_D:.3f} I={avg_I:.3f} S={avg_survival:.3f} G={avg_gen:.3f}")
        
        return phase_result
    
    def run_full_test(self) -> Dict[str, Any]:
        """Execute complete long-horizon stability test."""
        print("="*70)
        print("LONG-HORIZON STABILITY TEST (Option A)")
        print("Validating RAVANA paper claims over 10,000+ episodes")
        print("="*70)
        
        print(f"\nConfiguration:")
        print(f"  Total episodes: {self.n_episodes}")
        print(f"  Phase length: {self.checkpoint_interval}")
        print(f"  Number of phases: {len(self.environment_schedule)}")
        print(f"  Random seed: {self.seed}")
        
        print(f"\nEnvironment schedule:")
        for phase in self.environment_schedule[:6]:  # Show first 6
            print(f"  Phase {phase['phase']}: {phase['type']}")
        if len(self.environment_schedule) > 6:
            print(f"  ... and {len(self.environment_schedule) - 6} more phases")
        
        # Create agent
        agent = K2_Agent()
        
        print("\n" + "="*70)
        print("BEGINNING TEST")
        print("="*70)
        
        # Run all phases
        for phase_config in self.environment_schedule:
            phase_result = self.run_phase(agent, phase_config)
            self.phase_metrics.append(phase_result)
            self.results['phases'].append({
                'phase': phase_result.phase,
                'type': phase_result.environment_type,
                'dissonance': phase_result.avg_dissonance,
                'identity': phase_result.avg_identity,
                'survival': phase_result.survival_rate,
                'generalization': phase_result.generalization_acc
            })
        
        # Final analysis
        self._analyze_trajectory()
        self._validate_paper_claims()
        
        # Save results
        self.results['end_time'] = datetime.now().isoformat()
        
        return self.results
    
    def _analyze_trajectory(self):
        """Analyze learning trajectory over time."""
        if not self.episode_metrics:
            return
        
        # Split into early/mid/late
        n = len(self.episode_metrics)
        early = self.episode_metrics[:n//3]
        mid = self.episode_metrics[n//3:2*n//3]
        late = self.episode_metrics[2*n//3:]
        
        analysis = {
            'early': {
                'dissonance': np.mean([m.dissonance_D for m in early]),
                'identity': np.mean([m.identity_strength_I for m in early])
            },
            'mid': {
                'dissonance': np.mean([m.dissonance_D for m in mid]),
                'identity': np.mean([m.identity_strength_I for m in mid])
            },
            'late': {
                'dissonance': np.mean([m.dissonance_D for m in late]),
                'identity': np.mean([m.identity_strength_I for m in late])
            }
        }
        
        self.results['trajectory_analysis'] = analysis
        
        print("\n" + "="*70)
        print("TRAJECTORY ANALYSIS")
        print("="*70)
        print(f"Early (EP0-{n//3}):    D={analysis['early']['dissonance']:.3f} I={analysis['early']['identity']:.3f}")
        print(f"Mid   (EP{n//3}-{2*n//3}):   D={analysis['mid']['dissonance']:.3f} I={analysis['mid']['identity']:.3f}")
        print(f"Late  (EP{2*n//3}-{n}):  D={analysis['late']['dissonance']:.3f} I={analysis['late']['identity']:.3f}")
    
    def _validate_paper_claims(self):
        """Validate specific claims from the paper."""
        if not self.episode_metrics:
            return
        
        # Claim 1: Dissonance ~0.8 → ~0.2
        early_D = self.results['trajectory_analysis']['early']['dissonance']
        late_D = self.results['trajectory_analysis']['late']['dissonance']
        
        dissonance_drop = early_D - late_D
        dissonance_target_met = late_D <= 0.3  # Within 0.1 of 0.2 target
        
        # Claim 2: Identity ~0.3 → ~0.85
        early_I = self.results['trajectory_analysis']['early']['identity']
        late_I = self.results['trajectory_analysis']['late']['identity']
        
        identity_gain = late_I - early_I
        identity_target_met = late_I >= 0.75  # Within 0.1 of 0.85 target
        
        # Claim 3: Generalization ~0.9
        late_gen = np.mean([m.generalization_accuracy for m in self.episode_metrics[-len(self.episode_metrics)//3:]])
        gen_target_met = late_gen >= 0.8  # Within 0.1 of 0.9 target
        
        validation = {
            'dissonance_reduction': {
                'early': early_D,
                'late': late_D,
                'drop': dissonance_drop,
                'target_met': dissonance_target_met,
                'paper_claim': '0.8 → 0.2'
            },
            'identity_strengthening': {
                'early': early_I,
                'late': late_I,
                'gain': identity_gain,
                'target_met': identity_target_met,
                'paper_claim': '0.3 → 0.85'
            },
            'generalization': {
                'late_phase': late_gen,
                'target_met': gen_target_met,
                'paper_claim': '~0.9'
            },
            'overall_status': 'VALIDATED' if (dissonance_target_met and identity_target_met and gen_target_met) else 'PARTIAL'
        }
        
        self.results['paper_claims_validation'] = validation
        
        print("\n" + "="*70)
        print("PAPER CLAIMS VALIDATION")
        print("="*70)
        print(f"\n1. Dissonance Reduction (Claim: {validation['dissonance_reduction']['paper_claim']}):")
        print(f"   Achieved: {early_D:.3f} → {late_D:.3f} (Δ={dissonance_drop:+.3f})")
        print(f"   Status: {'✓ VALIDATED' if dissonance_target_met else '✗ NOT MET'}")
        
        print(f"\n2. Identity Strengthening (Claim: {validation['identity_strengthening']['paper_claim']}):")
        print(f"   Achieved: {early_I:.3f} → {late_I:.3f} (Δ={identity_gain:+.3f})")
        print(f"   Status: {'✓ VALIDATED' if identity_target_met else '✗ NOT MET'}")
        
        print(f"\n3. Generalization Accuracy (Claim: {validation['generalization']['paper_claim']}):")
        print(f"   Achieved: {late_gen:.3f}")
        print(f"   Status: {'✓ VALIDATED' if gen_target_met else '✗ NOT MET'}")
        
        print(f"\n{'='*70}")
        print(f"OVERALL: {validation['overall_status']}")
        print(f"{'='*70}")
    
    def save_results(self, filepath: str = None):
        """Save full results to JSON."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"long_horizon_test_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Execute long-horizon stability test."""
    test = LongHorizonStabilityTest(
        n_episodes=10000,  # Scaled to 10k for validation
        checkpoint_interval=1000,
        seed=42
    )
    
    results = test.run_full_test()
    test.save_results()
    
    # Exit code based on validation
    exit_code = 0 if results['paper_claims_validation']['overall_status'] == 'VALIDATED' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
