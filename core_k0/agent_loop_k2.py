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
    """Simple learned preferences for each action with confidence tracking."""
    explore: float = 0.5
    exploit: float = 0.5
    conserve: float = 0.5
    visit_count: int = 0  # NEW: Track visits to this context
    
    @property
    def confidence(self) -> float:
        """Confidence based on visit frequency (prevents early overfitting)."""
        return min(1.0, self.visit_count / 10.0)  # Max confidence at 10 visits
    
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
    
    # PAPER-COMPLIANT: Explicit Belief Tracking (Required for RAVANA Metrics)
    # These fields enable |belief - action| dissonance calculation
    belief_store: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.9,      # Strong prior: value alignment
        "accuracy": 0.9,      # Strong prior: correctness matters  
        "empathy": 0.9        # Strong prior: stakeholder awareness
    })
    
    confidence_scores: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.5,      # Initial uncertainty (will grow with experience)
        "accuracy": 0.5,
        "empathy": 0.5
    })
    
    vad_weights: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.8,      # VAD salience for value conflicts
        "accuracy": 0.8,
        "empathy": 0.8
    })
    
    # Identity State (Paper Section 4)
    identity_commitment: float = 0.3   # Baseline: low coherence initially
    cognitive_load: float = 0.5         # Moderate processing burden
    reappraisal_resistance: float = 0.5 # Neutral resistance to change
    
    def get_paper_metrics(self) -> Dict[str, Any]:
        """Expose state in format required by RAVANA metrics module."""
        return {
            "beliefs": list(self.belief_store.values()),
            "confidences": list(self.confidence_scores.values()),
            "vad_weights": list(self.vad_weights.values()),
            "identity_commitment": self.identity_commitment,
            "cognitive_load": self.cognitive_load,
            "reappraisal_resistance": self.reappraisal_resistance
        }
    
    def update_paper_metrics(self, action_taken: AgentAction, outcome: Dict[str, Any]):
        """Update belief/confidence based on action-outcome alignment."""
        # Convert action to numeric for conflict calculation
        action_map = {AgentAction.EXPLORE: 0.3, AgentAction.EXPLOIT: 0.7, AgentAction.CONSERVE: 0.9}
        action_value = action_map.get(action_taken, 0.5)
        
        # Calculate conflict per belief (|belief - action|)
        for key in self.belief_store:
            conflict = abs(self.belief_store[key] - action_value)
            
            # Update confidence based on outcome success
            if outcome.get("survived", True):
                # Success: slightly increase confidence
                self.confidence_scores[key] = min(0.95, self.confidence_scores[key] + 0.02)
            else:
                # Death: decrease confidence (belief may be wrong)
                self.confidence_scores[key] = max(0.1, self.confidence_scores[key] - 0.1)
        
        # Update identity based on action consistency
        recent_actions = [a[1] for a in self.action_history[-10:]] if self.action_history else []
        if recent_actions:
            # Consistency builds identity
            unique_actions = len(set(a.value for a in recent_actions))
            consistency = 1.0 - (unique_actions - 1) * 0.1  # Penalize action switching
            self.identity_commitment = 0.7 * self.identity_commitment + 0.3 * consistency
            self.identity_commitment = np.clip(self.identity_commitment, 0.1, 1.0)
    
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
    
    def get_context_key(self, energy: float, uncertainty: float, trend: float, failure_streak: int = 0) -> str:
        """UPGRADED: Discretize context with trend and failure streak buckets."""
        # Energy bucket
        e_bucket = "low" if energy < 0.25 else "med" if energy < 0.5 else "high"
        
        # Uncertainty bucket  
        u_bucket = "low" if uncertainty < 0.3 else "high"
        
        # NEW: Trend bucket (rising/stable/falling)
        t_bucket = "falling" if trend < -0.02 else "rising" if trend > 0.02 else "stable"
        
        # NEW: Failure streak bucket (0, 1-2, 3+)
        f_bucket = "0" if failure_streak == 0 else "1-2" if failure_streak <= 2 else "3+"
        
        return f"{e_bucket}_{u_bucket}_{t_bucket}_{f_bucket}"


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
        UPGRADED: Confidence-weighted learning with gated near-death boost.
        """
        context_key = self.state.get_context_key(
            outcome.context["energy"],
            outcome.context["uncertainty"],
            outcome.context["trend"],
            self.consecutive_exploration_failures  # Include failure streak
        )
        
        # Initialize context weights if new
        if context_key not in self.context_weights:
            self.context_weights[context_key] = {
                "explore": 0.5, "exploit": 0.5, "conserve": 0.5,
                "visits": 0  # Track visits for confidence
            }
        
        # Increment visit count
        self.context_weights[context_key]["visits"] = self.context_weights[context_key].get("visits", 0) + 1
        visits = self.context_weights[context_key]["visits"]
        confidence = min(1.0, visits / 10.0)  # Confidence builds over 10 visits
        
        # Base learning rate with confidence gating
        lr = self.learning_rate * confidence
        
        # TAMED: Near-death learning (2x instead of 3x, only if confidence > 0.3)
        if self._is_near_death() and confidence > 0.3:
            lr *= 2.0  # Boost but don't overreact
        
        action_name = outcome.action.value
        reward = 1.0 if outcome.delta_energy > 0 else -0.5
        if not outcome.survived:
            reward = -2.0  # Death is strong negative signal
        
        # Confidence-weighted update
        self.context_weights[context_key][action_name] += lr * reward
        
        # Special handling for exploration
        if outcome.action == AgentAction.EXPLORE:
            if outcome.exploration_success:
                self.policy.explore += lr * 0.3
                self.consecutive_exploration_failures = 0
            else:
                self.consecutive_exploration_failures += 1
                self.policy.explore -= lr * 0.2  # Gradual penalty
        
        # Keep weights bounded
        self.policy.normalize()
        for weights in self.context_weights.values():
            for k in ["explore", "exploit", "conserve"]:
                if k in weights:
                    weights[k] = np.clip(weights[k], 0.1, 0.9)
    
    def _get_action_by_expected_utility(self, context_key: str) -> AgentAction:
        """
        UPGRADED: Experience-driven action selection.
        
        Compare expected value of each action based on past outcomes.
        Returns the action with highest learned utility.
        """
        context_prefs = self.context_weights.get(context_key, {
            "explore": 0.5, "exploit": 0.5, "conserve": 0.5
        })
        
        # Calculate expected utility from recent outcomes
        explore_value = self._calculate_action_value(AgentAction.EXPLORE, context_key)
        exploit_value = self._calculate_action_value(AgentAction.EXPLOIT, context_key)
        conserve_value = self._calculate_action_value(AgentAction.CONSERVE, context_key)
        
        # Select action with highest expected value
        values = {
            AgentAction.EXPLORE: explore_value,
            AgentAction.EXPLOIT: exploit_value,
            AgentAction.CONSERVE: conserve_value
        }
        
        best_action = max(values, key=values.get)
        
        # Log the decision for debugging
        # print(f"  [K2 DECISION] E={explore_value:.2f} X={exploit_value:.2f} C={conserve_value:.2f} -> {best_action.value}")
        
        return best_action
    
    def _calculate_action_value(self, action: AgentAction, context_key: str, window: int = 10) -> float:
        """
        Calculate expected value of an action from recent outcomes.
        Returns weighted average: success_rate * avg_energy_gain
        """
        # Filter outcomes for this action and similar contexts
        relevant_outcomes = [
            o for o in self.state.outcome_history[-window:]
            if o.action == action
        ]
        
        if not relevant_outcomes:
            # No data: return neutral prior
            return 0.5
        
        # Calculate average reward (energy change)
        avg_delta = np.mean([o.delta_energy for o in relevant_outcomes])
        
        # Weight by survival rate
        survival_rate = sum(1 for o in relevant_outcomes if o.survived) / len(relevant_outcomes)
        
        # Special bonus for exploration success
        if action == AgentAction.EXPLORE:
            success_rate = sum(1 for o in relevant_outcomes if o.exploration_success) / len(relevant_outcomes)
            return avg_delta * survival_rate * (0.5 + 0.5 * success_rate)
        
        return avg_delta * survival_rate
    
    def _get_exploration_probability(self) -> float:
        """DEPRECATED: Replaced by expected utility."""
        return 0.5  # Fallback, not used
    
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
        context_key = self.state.get_context_key(E, U, trend, self.consecutive_exploration_failures)
        
        # FIXED: Handle both dict and PolicyWeights cases consistently
        weights_data = self.context_weights.get(context_key)
        if weights_data is None:
            # No learning yet: use defaults
            context_prefs = {"explore": 0.5, "exploit": 0.5, "conserve": 0.5}
        else:
            # Extract action weights only (handle dict format)
            context_prefs = {
                "explore": weights_data.get("explore", 0.5),
                "exploit": weights_data.get("exploit", 0.5),
                "conserve": weights_data.get("conserve", 0.5)
            }
        
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
            # Use expected utility to decide, not random probability
            mode = self._get_exploration_mode()
            if mode != ExplorationMode.DISABLED:
                # Experience-driven: pick action with highest expected value
                best_action = self._get_action_by_expected_utility(context_key)
                if best_action == AgentAction.EXPLORE:
                    self.steps_since_explore = 0
                    return AgentAction.EXPLORE
            # Fall through to safe policy
            self.steps_since_explore = 0
        
        # Normal policy: EXPERIENCE-DRIVEN (not heuristic)
        # Use expected utility to pick the best action
        if U > self.uncertainty_high and E > self.energy_low:
            # Uncertain but safe: use learned preferences
            return self._get_action_by_expected_utility(context_key)
        elif E < self.energy_low:
            return AgentAction.CONSERVE
        else:
            # Stable: use experience-driven selection
            return self._get_action_by_expected_utility(context_key)
    
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
        
        # PAPER-COMPLIANT: Update belief/confidence based on action-outcome
        self.state.update_paper_metrics(action, {
            "survived": result["alive"],
            "delta_energy": energy_after - energy_before,
            "utility": result["utility"]
        })
        
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
    
    def get_paper_metrics(self) -> Dict[str, Any]:
        """Expose paper-compliant metrics for validation testing."""
        return self.state.get_paper_metrics()
    
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
