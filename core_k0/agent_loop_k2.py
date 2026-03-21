"""
RAVANA K2 — Survival-Conditioned Policy Adaptation
"Experience → Strategy"

Core principle: Learn from action outcomes, weighted by survival criticality.
Not deep RL. Simple, interpretable preference learning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import sys
sys.path.insert(0, '/home/workspace/ravana_v2')
from experiments_k0.resource_env import AgentAction


class ExplorationMode(Enum):
    """K2 inherits K1.3's context-aware states."""
    DISABLED = "disabled"
    GUARDED = "guarded"
    ENABLED = "enabled"


@dataclass
class ActionOutcome:
    """Record of what happened when an action was taken."""
    episode: int
    context: Dict[str, float]  # (energy, uncertainty, trend, regime)
    action: AgentAction
    energy_before: float
    energy_after: float
    delta_energy: float
    survived: bool
    exploration_success: bool  # For EXPLORE: did energy increase?


@dataclass
class PolicyWeights:
    """Simple learned preferences for each action."""
    explore: float = 0.5
    exploit: float = 0.5
    conserve: float = 0.5
    
    def normalize(self):
        """Keep weights in [0, 1] range."""
        self.explore = np.clip(self.explore, 0.1, 0.9)
        self.exploit = np.clip(self.exploit, 0.1, 0.9)
        self.conserve = np.clip(self.conserve, 0.1, 0.9)


@dataclass
class AgentState:
    """K2 state with learned policy."""
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    
    # K2: Outcome memory
    outcome_history: List[ActionOutcome] = field(default_factory=list)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        self.energy_estimate = obs.get("energy_obs", self.energy_estimate)
        self.resource_estimate = obs.get("resource_obs", self.resource_estimate)
        noise = obs.get("noise", 0.0)
        self.risk_estimate = 0.2 + noise * 0.5
        self.uncertainty = obs.get("observation_quality", 0.3)
        self.energy_history.append(self.energy_estimate)
        if len(self.energy_history) > 20:
            self.energy_history = self.energy_history[-20:]
    
    def get_energy_trend(self, window: int = 5) -> float:
        """Slope of energy over last N steps."""
        if len(self.energy_history) < window:
            return 0.0
        recent = self.energy_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0] if len(set(recent)) > 1 else 0.0
        return float(slope)
    
    def record_outcome(self, outcome: ActionOutcome):
        """Store what happened for learning."""
        self.outcome_history.append(outcome)
        # Keep last 100 outcomes (sliding window)
        if len(self.outcome_history) > 100:
            self.outcome_history = self.outcome_history[-100:]
    
    def get_context_key(self, energy: float, uncertainty: float, trend: float) -> str:
        """Discretize context for learning."""
        # Bucket into coarse categories
        e_bucket = "low" if energy < 0.25 else "med" if energy < 0.5 else "high"
        u_bucket = "low" if uncertainty < 0.3 else "high"
        t_bucket = "falling" if trend < -0.02 else "rising" if trend > 0.02 else "stable"
        return f"{e_bucket}_{u_bucket}_{t_bucket}"


class K2_Agent:
    """
    K2: Survival-Conditioned Policy Adaptation.
    
    Learns:
    - Which actions work in which contexts
    - Exploration success rates by energy level
    - Survival-critical adjustments (higher learning rate near death)
    """
    
    def __init__(self, learning_rate: float = 0.1, survival_boost: float = 3.0):
        self.state = AgentState()
        self.policy = PolicyWeights()
        
        self.episode: int = 0
        self.survival_count: int = 0
        self.death_count: int = 0
        self.cumulative_reward: float = 0.0
        
        # K1.3 survival parameters
        self.energy_critical: float = 0.15
        self.energy_low: float = 0.35
        self.uncertainty_high: float = 0.4
        self.base_metabolism: float = 0.02
        
        # Tracking
        self.steps_since_explore: int = 0
        self.steps_without_resource_gain: int = 0
        self.last_resource_estimate: float = 0.5
        self.consecutive_exploration_failures: int = 0
        
        # K2 learning parameters
        self.learning_rate = learning_rate
        self.survival_boost = survival_boost  # Multiplier near death
        
        # Context-specific action preferences (learned)
        self.context_weights: Dict[str, Dict[str, float]] = {}
        
        # Exploration success tracking
        self.explore_success_by_context: Dict[str, List[bool]] = {}
    
    def _get_exploration_mode(self) -> ExplorationMode:
        """K2 inherits K1.3's context-aware mode selection."""
        E = self.state.energy_estimate
        trend = self.state.get_energy_trend(window=5)
        survival_buffer = self.base_metabolism * 3
        
        if E < survival_buffer * 2 or trend < -0.05:
            return ExplorationMode.DISABLED
        if E < survival_buffer * 4 or trend < 0:
            return ExplorationMode.GUARDED
        return ExplorationMode.ENABLED
    
    def _is_near_death(self) -> bool:
        """Check if in survival-critical zone."""
        return self.state.energy_estimate < self.energy_critical * 2
    
    def _get_effective_learning_rate(self) -> float:
        """Higher learning rate when survival is at stake."""
        if self._is_near_death():
            return self.learning_rate * self.survival_boost
        return self.learning_rate
    
    def _learn_from_outcome(self, outcome: ActionOutcome):
        """
        Update policy based on what happened.
        
        Simple rule-based learning (no neural nets):
        - Positive energy change → strengthen action weight
        - Near-death failure → strongly penalize
        - Context-specific tracking
        """
        lr = self._get_effective_learning_rate()
        context_key = self.state.get_context_key(
            outcome.context["energy"],
            outcome.context["uncertainty"],
            outcome.context["trend"]
        )
        
        # Initialize context weights if new
        if context_key not in self.context_weights:
            self.context_weights[context_key] = {
                "explore": 0.5, "exploit": 0.5, "conserve": 0.5
            }
        
        action_name = outcome.action.value
        
        # Update based on outcome
        if outcome.delta_energy > 0:
            # Success: strengthen this action in this context
            self.context_weights[context_key][action_name] += lr * 0.5
            
            # Special bonus for exploration that actually gains energy
            if outcome.action == AgentAction.EXPLORE:
                self.policy.explore += lr * 0.3
                self.consecutive_exploration_failures = 0
        else:
            # Failure: weaken this action
            penalty = lr * 0.3
            if not outcome.survived:
                # Death is a strong signal
                penalty *= 2.0
            self.context_weights[context_key][action_name] -= penalty
            
            if outcome.action == AgentAction.EXPLORE:
                self.consecutive_exploration_failures += 1
        
        # Keep weights bounded
        self.policy.normalize()
        for weights in self.context_weights.values():
            for k in weights:
                weights[k] = np.clip(weights[k], 0.1, 0.9)
    
    def _get_exploration_probability(self) -> float:
        """
        Adaptive exploration: probability based on past success + current state.
        """
        mode = self._get_exploration_mode()
        
        if mode == ExplorationMode.DISABLED:
            return 0.0
        
        # Base probability from learned policy
        base_prob = self.policy.explore
        
        # Adjust by recent success rate
        recent_explores = [
            o for o in self.state.outcome_history[-20:]
            if o.action == AgentAction.EXPLORE
        ]
        if recent_explores:
            success_rate = sum(1 for o in recent_explores if o.exploration_success) / len(recent_explores)
            # If exploration usually fails, reduce probability
            base_prob *= (0.5 + 0.5 * success_rate)  # Scale 0.5x to 1.0x
        
        # Penalize consecutive failures
        failure_penalty = min(0.3, self.consecutive_exploration_failures * 0.1)
        base_prob -= failure_penalty
        
        # Guarded mode: reduce further
        if mode == ExplorationMode.GUARDED:
            base_prob *= 0.5
        
        return max(0.0, min(1.0, base_prob))
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        """K2: Context-aware + learned preferences."""
        self.episode += 1
        self.state.update_from_observation(obs, self.episode)
        
        E = self.state.energy_estimate
        U = self.state.uncertainty
        R = self.state.resource_estimate
        trend = self.state.get_energy_trend(5)
        
        # Detect resource gain
        if R > self.last_resource_estimate + 0.05:
            self.steps_without_resource_gain = 0
        else:
            self.steps_without_resource_gain += 1
        self.last_resource_estimate = R
        
        # Update exploration tracking
        self.steps_since_explore += 1
        
        # Get context key for learned preferences
        context_key = self.state.get_context_key(E, U, trend)
        context_prefs = self.context_weights.get(context_key, {
            "explore": 0.5, "exploit": 0.5, "conserve": 0.5
        })
        
        # 🔥 K2: STARVATION TRIGGERS (with learned refinements)
        if E < self.energy_critical:
            # Near death: prefer action with highest learned weight
            best_action = max(context_prefs, key=context_prefs.get)
            return AgentAction(best_action)
        
        if self.steps_without_resource_gain > 15:
            # Starving: check if exploration is feasible
            mode = self._get_exploration_mode()
            if mode != ExplorationMode.DISABLED:
                explore_prob = self._get_exploration_probability()
                if np.random.random() < explore_prob:
                    self.steps_since_explore = 0
                    return AgentAction.EXPLORE
            return AgentAction.CONSERVE
        
        # 🔥 K2: ADAPTIVE EXPLORATION FLOOR
        if self.steps_since_explore > 10:
            explore_prob = self._get_exploration_probability()
            if np.random.random() < explore_prob and self._get_exploration_mode() != ExplorationMode.DISABLED:
                self.steps_since_explore = 0
                return AgentAction.EXPLORE
            # Fall through to safe policy
            self.steps_since_explore = 0
        
        # Normal policy with learned preferences
        if U > self.uncertainty_high and E > self.energy_low:
            # Uncertain but safe: maybe explore
            if np.random.random() < context_prefs["explore"]:
                return AgentAction.EXPLORE
            return AgentAction.EXPLOIT
        elif E < self.energy_low:
            return AgentAction.CONSERVE
        else:
            # Stable: use highest-weighted action
            best_action = max(context_prefs, key=context_prefs.get)
            return AgentAction(best_action)
    
    def step(self, env) -> Dict[str, Any]:
        """Execute one step with learning."""
        # Capture state before action
        energy_before = env.true_energy
        context = {
            "energy": self.state.energy_estimate,
            "uncertainty": self.state.uncertainty,
            "trend": self.state.get_energy_trend(5),
            "regime": getattr(env, 'current_regime', 'unknown')
        }
        
        # Select and execute action
        obs = env._generate_observation()
        action = self.select_action(obs)
        result = env.execute_action(action)
        
        # Track if this was exploration
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
        
        # Record outcome for learning
        energy_after = env.true_energy
        outcome = ActionOutcome(
            episode=self.episode,
            context=context,
            action=action,
            energy_before=energy_before,
            energy_after=energy_after,
            delta_energy=energy_after - energy_before,
            survived=result["alive"],
            exploration_success=(action == AgentAction.EXPLORE and energy_after > energy_before)
        )
        self.state.record_outcome(outcome)
        
        # Learn from this experience
        self._learn_from_outcome(outcome)
        
        self.cumulative_reward += result["utility"]
        self.state.action_history.append((self.episode, action, result["utility"]))
        
        if result["alive"]:
            self.survival_count += 1
        else:
            self.death_count += 1
        
        return {
            "alive": result["alive"],
            "observation": obs,
            "action": action,
            "episode": self.episode,
            "mode": self._get_exploration_mode().value,
            "energy_trend": context["trend"],
            "learned": True
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Report including learned state."""
        return {
            "episode": self.episode,
            "survival_count": self.survival_count,
            "death_count": self.death_count,
            "survival_rate": self.survival_count / max(1, self.episode),
            "cumulative_reward": self.cumulative_reward,
            "policy_weights": {
                "explore": self.policy.explore,
                "exploit": self.policy.exploit,
                "conserve": self.policy.conserve
            },
            "context_weights_count": len(self.context_weights),
            "exploration_success_rate": self._get_recent_exploration_success(),
            "current_state": {
                "energy": self.state.energy_estimate,
                "uncertainty": self.state.uncertainty,
                "energy_trend": self.state.get_energy_trend(5),
                "mode": self._get_exploration_mode().value,
                "consecutive_failures": self.consecutive_exploration_failures
            }
        }
    
    def _get_recent_exploration_success(self) -> float:
        """Calculate recent exploration success rate."""
        recent = [
            o for o in self.state.outcome_history[-20:]
            if o.action == AgentAction.EXPLORE
        ]
        if not recent:
            return 0.5
        return sum(1 for o in recent if o.exploration_success) / len(recent)
