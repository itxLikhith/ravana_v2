"""
RAVANA v2 — IDENTITY ENGINE
Manages identity strength with momentum and recovery bias.

Core principle: Identity grows from resolution, decays from stagnation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class IdentityState:
    """Current identity state."""
    strength: float = 0.5
    momentum: float = 0.0  # Directional inertia
    stability: float = 0.5  # Resistance to change
    
    # History for trend analysis
    history: List[float] = field(default_factory=lambda: [0.5])
    
    def update(self, new_strength: float):
        """Update identity with history tracking."""
        self.history.append(new_strength)
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self.strength = new_strength


class IdentityEngine:
    """
    Identity dynamics with momentum and recovery bias.
    
    Key features:
    - Momentum: Changes have inertia (trend continuation)
    - Recovery bias: Low identity gets bonus growth
    - Stability: High identity resists change
    """
    
    def __init__(
        self,
        initial_strength: float = 0.5,
        momentum_factor: float = 0.3,
        recovery_bias: float = 0.05,
        stability_threshold: float = 0.7
    ):
        self.state = IdentityState(strength=initial_strength)
        self.momentum_factor = momentum_factor
        self.recovery_bias = recovery_bias
        self.stability_threshold = stability_threshold
        
        # Update tracking
        self.last_delta: float = 0.0
        
    def compute_update(
        self,
        resolution_delta: float,
        resolution_success: bool,
        regulated_identity_delta: float,  # From governor
        current_dissonance: float
    ) -> float:
        """
        Compute identity update with all dynamics.
        
        Returns: new identity strength (not delta - absolute value)
        """
        # Start with governor-regulated delta
        delta = regulated_identity_delta
        
        # Momentum: Continue previous trend
        if abs(self.last_delta) > 0.001:
            momentum = np.sign(self.last_delta) * self.momentum_factor * abs(self.last_delta)
            delta += momentum
        
        # Resolution bonus: Successful resolution strengthens identity
        if resolution_success and resolution_delta > 0.05:
            delta += 0.02  # Small bonus for successful resolution
        
        # Recovery bias: Low identity gets growth boost
        if self.state.strength < 0.3:
            recovery_boost = self.recovery_bias * (0.3 - self.state.strength)
            delta += recovery_boost
        
        # Stability: High identity resists change
        if self.state.strength > self.stability_threshold:
            stability_damping = (self.state.strength - self.stability_threshold) * 0.3
            delta *= (1.0 - stability_damping)
        
        # Dissonance coupling: High dissonance weakens identity growth
        if current_dissonance > 0.7:
            stress_penalty = (current_dissonance - 0.7) * 0.1
            delta -= stress_penalty
        
        # Compute new strength
        new_strength = self.state.strength + delta
        
        # Track for momentum
        self.last_delta = delta
        
        return new_strength
    
    def apply_update(self, new_strength: float):
        """Apply computed update to state."""
        self.state.update(new_strength)
    
    def get_trend(self, window: int = 20) -> float:
        """Compute recent trend (positive = growing, negative = shrinking)."""
        if len(self.state.history) < window:
            return 0.0
        
        recent = self.state.history[-window:]
        early_mean = np.mean(recent[:window//2])
        late_mean = np.mean(recent[window//2:])
        
        return late_mean - early_mean
    
    def get_status(self) -> Dict[str, Any]:
        """Return identity status for monitoring."""
        return {
            "strength": self.state.strength,
            "momentum": self.last_delta,
            "stability": self.state.stability,
            "trend": self.get_trend(),
            "history_length": len(self.state.history),
        }
