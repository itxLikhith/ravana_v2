"""
RAVANA v2 — STRATEGY LAYER v0 (Phase B.5 Minimal)
Deliberate mode selection: choosing how to explore.

PRINCIPLE: Don't just react to clamps. Choose exploration mode based on context.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum, auto
from collections import deque


class ExplorationMode(Enum):
    """Deliberate exploration modes."""
    EXPLORE_AGGRESSIVE = "explore_aggressive"  # High uncertainty, high potential
    EXPLORE_SAFE = "explore_safe"              # Near boundary, need caution
    STABILIZE = "stabilize"                     # Lock in gains
    RECOVER = "recover"                         # Crisis response


@dataclass
class ModeSelection:
    """Result of mode selection decision."""
    mode: ExplorationMode
    confidence: float  # 0-1, how clear the decision is
    reason: str
    context: Dict[str, float]


@dataclass
class StrategyConfig:
    """Configuration for strategy layer."""
    # Mode thresholds
    crisis_clamp_rate: float = 0.15      # >15% clamps = crisis
    high_exploration_threshold: float = 0.25  # D < 0.25 = room to explore
    boundary_proximity: float = 0.75   # D > 0.75 = near boundary
    stability_threshold: float = 0.02    # Var < 0.02 = stable
    
    # Mode → policy bias
    aggressive_delta_scale: float = 1.3
    aggressive_noise: float = 0.03
    aggressive_dampening: float = 0.3
    
    safe_delta_scale: float = 0.7
    safe_noise: float = 0.01
    safe_dampening: float = 0.7
    
    stabilize_delta_scale: float = 0.4
    stabilize_noise: float = 0.005
    stabilize_dampening: float = 0.9
    
    recover_delta_scale: float = 0.3
    recover_noise: float = 0.0
    recover_dampening: float = 1.0


@dataclass
class BehavioralContext:
    """Snapshot of current system state for mode selection."""
    clamp_rate: float = 0.0
    dissonance: float = 0.5
    identity: float = 0.5
    dissonance_trend: float = 0.0  # Positive = rising
    identity_drift: float = 0.0    # Negative = falling
    stability: float = 0.5         # Low variance = stable
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for policy input."""
        return np.array([
            self.clamp_rate,
            self.dissonance,
            self.identity,
            self.dissonance_trend,
            self.identity_drift,
            self.stability
        ])


class StrategyLayer:
    """
    Minimal strategy layer: deliberate mode selection.
    
    Replaces: "react to clamps"
    With: "choose exploration mode based on context"
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.current_mode: ExplorationMode = ExplorationMode.EXPLORE_SAFE
        self.mode_history: deque = deque(maxlen=100)
        self.context_history: deque = deque(maxlen=20)
        
        # Mode statistics
        self.mode_durations: Dict[ExplorationMode, List[int]] = {
            mode: [] for mode in ExplorationMode
        }
        self.mode_switches: int = 0
        self.current_mode_start: int = 0
        
    def select_mode(self, context: BehavioralContext, episode: int) -> ModeSelection:
        """
        Select exploration mode based on current context.
        
        This is where intent emerges: choosing HOW to behave.
        """
        self.context_history.append(context)
        
        # Priority 1: Crisis override
        if context.clamp_rate > self.config.crisis_clamp_rate:
            return self._select(ExplorationMode.RECOVER, 0.9, 
                              f"CRISIS: clamp_rate={context.clamp_rate:.2%}", context)
        
        # Priority 2: High stability + good identity = lock in gains
        if (context.stability < self.config.stability_threshold and 
            context.identity > 0.7 and 
            context.dissonance < 0.5):
            return self._select(ExplorationMode.STABILIZE, 0.8,
                              f"STABLE: var={context.stability:.3f}, I={context.identity:.2f}", context)
        
        # Priority 3: Low dissonance = room for aggressive exploration
        if context.dissonance < self.config.high_exploration_threshold:
            return self._select(ExplorationMode.EXPLORE_AGGRESSIVE, 0.7,
                              f"ROOM: D={context.dissonance:.2f}", context)
        
        # Priority 4: Near boundary = cautious exploration
        if context.dissonance > self.config.boundary_proximity:
            return self._select(ExplorationMode.EXPLORE_SAFE, 0.75,
                              f"BOUNDARY: D={context.dissonance:.2f}", context)
        
        # Default: safe exploration
        return self._select(ExplorationMode.EXPLORE_SAFE, 0.5,
                          f"DEFAULT: ambiguous context", context)
    
    def _select(self, mode: ExplorationMode, confidence: float, 
                reason: str, context: BehavioralContext) -> ModeSelection:
        """Record mode selection and return result."""
        # Track mode switches
        if mode != self.current_mode:
            duration = len(self.mode_history) - self.current_mode_start
            self.mode_durations[self.current_mode].append(duration)
            self.mode_switches += 1
            self.current_mode = mode
            self.current_mode_start = len(self.mode_history)
        
        self.mode_history.append(mode)
        
        return ModeSelection(
            mode=mode,
            confidence=confidence,
            reason=reason,
            context={
                'clamp_rate': context.clamp_rate,
                'dissonance': context.dissonance,
                'identity': context.identity,
                'trend': context.dissonance_trend
            }
        )
    
    def apply_policy_bias(self, raw_deltas: Tuple[float, float], 
                         mode: ExplorationMode) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply mode-specific policy bias to raw deltas.
        
        Returns: (modified_d_delta, modified_i_delta, bias_info)
        """
        dd, di = raw_deltas
        
        if mode == ExplorationMode.EXPLORE_AGGRESSIVE:
            dd *= self.config.aggressive_delta_scale
            noise_dd = np.random.normal(0, self.config.aggressive_noise)
            noise_di = np.random.normal(0, self.config.aggressive_noise * 0.5)
            dd += noise_dd
            di += noise_di
            dampening = self.config.aggressive_dampening
            
        elif mode == ExplorationMode.EXPLORE_SAFE:
            dd *= self.config.safe_delta_scale
            noise_dd = np.random.normal(0, self.config.safe_noise)
            dd += noise_dd
            dampening = self.config.safe_dampening
            
        elif mode == ExplorationMode.STABILIZE:
            dd *= self.config.stabilize_delta_scale
            # Reduce identity noise to preserve gains
            noise_di = np.random.normal(0, self.config.stabilize_noise)
            di += noise_di
            dampening = self.config.stabilize_dampening
            
        elif mode == ExplorationMode.RECOVER:
            dd *= self.config.recover_delta_scale
            # Force dissonance reduction
            if dd > 0:
                dd = -abs(dd) * 0.5  # Reverse and dampen
            dampening = self.config.recover_dampening
            
        else:
            dampening = 0.5  # Default
        
        return dd, di, {
            'mode': mode.value,
            'delta_scale': getattr(self.config, f'{mode.value}_delta_scale', 1.0),
            'noise_injected': noise_dd if 'noise_dd' in dir() else 0.0,
            'dampening': dampening
        }
    
    def compute_context(self, governor, state_manager, window: int = 20) -> BehavioralContext:
        """
        Compute behavioral context from recent history.
        
        Extracts trends and patterns for mode selection.
        """
        history = state_manager.history[-window:] if len(state_manager.history) >= window else state_manager.history
        
        if len(history) < 5:
            return BehavioralContext()  # Not enough data
        
        # Recent clamp rate
        recent_caps = sum(1 for h in history if h.get('constraint_activated', False))
        clamp_rate = recent_caps / len(history)
        
        # Current state
        current = state_manager.state
        
        # Trend computation (simple linear trend)
        if len(history) >= 10:
            recent_d = [h['post_dissonance'] for h in history[-10:]]
            early_mean = np.mean(recent_d[:5])
            late_mean = np.mean(recent_d[5:])
            dissonance_trend = late_mean - early_mean
        else:
            dissonance_trend = 0.0
        
        # Identity drift
        if len(history) >= 10:
            recent_i = [h['post_identity'] for h in history[-10:]]
            early_i = np.mean(recent_i[:5])
            late_i = np.mean(recent_i[5:])
            identity_drift = late_i - early_i
        else:
            identity_drift = 0.0
        
        # Stability (variance)
        if len(history) >= 5:
            recent_states = [h['post_dissonance'] for h in history[-5:]]
            stability = np.var(recent_states)
        else:
            stability = 0.5
        
        return BehavioralContext(
            clamp_rate=clamp_rate,
            dissonance=current.dissonance,
            identity=current.identity,
            dissonance_trend=dissonance_trend,
            identity_drift=identity_drift,
            stability=stability
        )
    
    def get_mode_analytics(self) -> Dict[str, Any]:
        """
        Return analytics about mode usage and effectiveness.
        """
        if not self.mode_history:
            return {"error": "No mode history"}
        
        # Mode distribution
        mode_counts = {}
        for mode in self.mode_history:
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1
        
        # Average durations
        avg_durations = {}
        for mode, durations in self.mode_durations.items():
            avg_durations[mode.value] = np.mean(durations) if durations else 0
        
        # Switch frequency
        total_steps = len(self.mode_history)
        switch_rate = self.mode_switches / total_steps if total_steps > 0 else 0
        
        return {
            "mode_distribution": mode_counts,
            "avg_mode_duration": avg_durations,
            "mode_switches": self.mode_switches,
            "switch_rate": switch_rate,
            "current_mode": self.current_mode.value,
            "current_mode_duration": len(self.mode_history) - self.current_mode_start
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Quick status summary."""
        return {
            "current_mode": self.current_mode.value,
            "mode_history_length": len(self.mode_history),
            "mode_switches": self.mode_switches,
            "recent_modes": [m.value for m in list(self.mode_history)[-5:]]
        }
