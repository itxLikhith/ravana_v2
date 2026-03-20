"""
RAVANA v2 — ADAPTATION ENGINE (Phase B Minimal)
Lightweight policy tweak layer that learns from clamp events.

PRINCIPLE: Learn to avoid needing correction, not just to stay in bounds.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from collections import deque


@dataclass
class AdaptationConfig:
    """Configuration for adaptation learning."""
    # Learning rates
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Dual objective weights
    exploration_bonus: float = 0.1  # Encourage healthy dissonance
    clamp_penalty: float = 1.0      # Penalize needing correction
    
    # State encoding
    state_window: int = 5  # Episodes of history for pattern
    
    # Conservative learning
    max_tweak: float = 0.05   # Max adjustment per episode
    decay_rate: float = 0.999  # Slow forgetting


@dataclass
class ClampExperience:
    """Single experience from clamp event."""
    episode: int
    state_encoding: np.ndarray  # [D, I, dD/dt, trend, mode]
    variable: str  # 'dissonance' or 'identity'
    correction: float
    layer: str
    
    # Computed reward
    reward: float = 0.0


class PolicyTweakLayer:
    """
    Minimal policy adaptation layer.
    
    This sits BETWEEN upstream computation and governor.
    It nudges raw deltas based on learned patterns.
    
    Design constraints:
    - Lightweight: 50-100 lines core logic
    - Reversible: Can be disabled instantly
    - Measurable: Clear before/after comparison
    """
    
    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()
        
        # Simple linear policy: state -> tweak direction
        # 5D state -> 2D output (dissonance_tweak, identity_tweak)
        self.weights = np.zeros((5, 2))
        self.velocity = np.zeros((5, 2))  # Momentum accumulator
        
        # Experience buffer
        self.experiences: deque = deque(maxlen=1000)
        
        # Metrics
        self.total_tweaks = 0
        self.cumulative_tweak_magnitude = 0.0
        self.learning_steps = 0
        
    def encode_state(
        self,
        current_d: float,
        current_i: float,
        recent_history: List[Dict],
        mode: str
    ) -> np.ndarray:
        """
        Encode current cognitive state for policy input.
        
        5D encoding:
        [dissonance, identity, velocity_d, velocity_i, mode_encoded]
        """
        # Compute velocities from recent history
        if len(recent_history) >= 2:
            vel_d = recent_history[-1]['dissonance'] - recent_history[-2]['dissonance']
            vel_i = recent_history[-1]['identity'] - recent_history[-2]['identity']
        else:
            vel_d, vel_i = 0.0, 0.0
        
        # Encode mode as scalar
        mode_map = {'normal': 0.0, 'exploration': -1.0, 'resolution': 1.0,
                   'recovery': 2.0, 'plateau': -0.5}
        mode_val = mode_map.get(mode, 0.0)
        
        return np.array([current_d, current_i, vel_d, vel_i, mode_val])
    
    def compute_tweak(
        self,
        raw_d_delta: float,
        raw_i_delta: float,
        state_encoding: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute policy tweak for raw deltas.
        
        Returns: (tweaked_d_delta, tweaked_i_delta)
        """
        # Forward pass: state -> tweak
        tweak = self.weights.T @ state_encoding  # [2] vector
        
        # Clip to conservative bounds
        tweak = np.clip(tweak, -self.config.max_tweak, self.config.max_tweak)
        
        # Apply tweaks
        tweaked_d = raw_d_delta + tweak[0]
        tweaked_i = raw_i_delta + tweak[1]
        
        # Track metrics
        self.total_tweaks += 1
        self.cumulative_tweak_magnitude += np.abs(tweak).sum()
        
        return tweaked_d, tweaked_i
    
    def learn_from_clamp(
        self,
        experience: ClampExperience
    ):
        """
        Update policy from a single clamp event.
        
        This is the core learning signal:
        - Bigger correction = bigger negative reward
        - State encoding tells us WHERE we were wrong
        """
        # Compute reward: negative of correction (scaled)
        base_reward = -experience.correction * self.config.clamp_penalty
        
        # Add exploration bonus (healthy dissonance is good)
        dissonance = experience.state_encoding[0]
        if 0.3 < dissonance < 0.7:  # Sweet spot
            base_reward += self.config.exploration_bonus
        
        experience.reward = base_reward
        self.experiences.append(experience)
        
        # Gradient update (simplified policy gradient)
        # If we got clamped, nudge weights AWAY from that state's direction
        gradient = np.outer(experience.state_encoding, 
                           [experience.correction if experience.variable == 'dissonance' else 0,
                            experience.correction if experience.variable == 'identity' else 0])
        
        # Momentum update
        self.velocity = (self.config.momentum * self.velocity + 
                        (1 - self.config.momentum) * gradient)
        
        # Apply gradient with learning rate
        self.weights -= self.config.learning_rate * self.velocity
        
        # Decay (forget slowly)
        self.weights *= self.config.decay_rate
        
        self.learning_steps += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Return adaptation status."""
        recent_exps = list(self.experiences)[-100:] if self.experiences else []
        mean_reward = np.mean([e.reward for e in recent_exps]) if recent_exps else 0.0
        
        return {
            "total_tweaks": self.total_tweaks,
            "mean_tweak_magnitude": (self.cumulative_tweak_magnitude / max(1, self.total_tweaks)),
            "learning_steps": self.learning_steps,
            "experience_buffer_size": len(self.experiences),
            "mean_recent_reward": mean_reward,
            "weight_norm": np.linalg.norm(self.weights),
        }
    
    def get_learning_report(self) -> str:
        """Generate human-readable learning report."""
        s = self.get_status()
        
        lines = [
            "=" * 50,
            "🧠 ADAPTATION ENGINE REPORT",
            "=" * 50,
            f"Learning steps: {s['learning_steps']:,}",
            f"Experiences: {s['experience_buffer_size']:,}",
            f"Mean tweak magnitude: {s['mean_tweak_magnitude']:.4f}",
            f"Mean recent reward: {s['mean_recent_reward']:.4f}",
            f"Weight norm: {s['weight_norm']:.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class AdaptiveGovernorBridge:
    """
    Bridge that wires adaptation layer into governor flow.
    
    This is the integration point:
    1. Intercepts raw deltas from upstream
    2. Applies policy tweak
    3. Passes to governor (which may still clamp)
    4. Feeds back clamp events to adaptation layer
    """
    
    def __init__(self, governor, adaptation: PolicyTweakLayer):
        self.governor = governor
        self.adaptation = adaptation
        
        # State history for encoding
        self.state_history: deque = deque(maxlen=10)
        
    def step(
        self,
        current_d: float,
        current_i: float,
        raw_signals,
        episode: int = 0
    ):
        """
        Execute one adaptive step.
        
        Flow:
        raw_signals -> tweak -> governor -> clamp_check -> learn
        """
        # Encode current state
        recent_mode = self.governor.mode_history[-1].value if self.governor.mode_history else 'normal'
        state_enc = self.adaptation.encode_state(
            current_d, current_i,
            list(self.state_history), recent_mode
        )
        
        # Apply policy tweak BEFORE governor
        tweaked_d, tweaked_i = self.adaptation.compute_tweak(
            raw_signals.dissonance_delta,
            raw_signals.identity_delta,
            state_enc
        )
        
        # Create tweaked signals
        from .governor import CognitiveSignals
        tweaked_signals = CognitiveSignals(
            dissonance_delta=tweaked_d,
            identity_delta=tweaked_i,
            exploration_drive=raw_signals.exploration_drive,
            resolution_potential=raw_signals.resolution_potential,
            trend=raw_signals.trend,
            horizon=raw_signals.horizon,
            source=f"{raw_signals.source}+tweaked"
        )
        
        # Governor regulation (may still clamp)
        regulated = self.governor.regulate(
            current_d, current_i, tweaked_signals, episode
        )
        
        # Check for clamp events and feed to adaptation
        self._feed_clamps_to_adaptation(episode, state_enc)
        
        # Update state history
        self.state_history.append({
            'dissonance': current_d + regulated.dissonance_delta,
            'identity': current_i + regulated.identity_delta,
            'mode': regulated.mode.value
        })
        
        return regulated
    
    def _feed_clamps_to_adaptation(self, episode: int, state_enc: np.ndarray):
        """Extract clamp events and feed to adaptation layer."""
        diagnostics = self.governor.clamp_diagnostics
        
        # Get recent events we haven't processed yet
        # (Simplified: just check last few events)
        for event in list(diagnostics.events)[-5:]:
            if event.episode == episode:  # Event from current step
                exp = ClampExperience(
                    episode=episode,
                    state_encoding=state_enc,
                    variable=event.variable,
                    correction=event.correction,
                    layer=event.layer
                )
                self.adaptation.learn_from_clamp(exp)
