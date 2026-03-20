"""
RAVANA v2 — GOVERNOR (First-Class Citizen)
Central control system for cognitive state regulation.

PRINCIPLE: No state modification without governor passage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class RegulationMode(Enum):
    """Governor regulation modes."""
    NORMAL = "normal"          # Standard operation
    EXPLORATION = "exploration"  # High uncertainty, seek novelty
    RESOLUTION = "resolution"    # Active conflict resolution
    RECOVERY = "recovery"        # Crisis recovery mode
    PLATEAU = "plateau"          # Stagnation detected


@dataclass
class GovernorConfig:
    """Immutable governor configuration."""
    # Hard constraints (non-negotiable)
    max_dissonance: float = 0.95
    min_dissonance: float = 0.15
    target_dissonance: float = 0.60  # Sweet spot for learning
    max_identity: float = 0.95
    soft_limit: float = 0.70  # Start pressure here
    boundary_k: float = 10.0  # Pressure curve steepness
    min_pressure: float = 0.2  # Minimum allowed pressure
    min_identity: float = 0.10
    
    # Regulation parameters
    dissonance_target: float = 0.45
    identity_target: float = 0.65
    exploration_threshold: float = 0.30
    resolution_threshold: float = 0.70
    
    # Recovery parameters
    recovery_boost: float = 0.15
    crisis_threshold: float = 0.90
    
    # Plateau detection
    plateau_window: int = 50
    plateau_tolerance: float = 0.02


@dataclass
class CognitiveSignals:
    """Raw signals from cognitive modules (pre-governor)."""
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    exploration_drive: float = 0.0
    resolution_potential: float = 0.0
    
    # 🔮 Grace Layer: Predictive fields
    trend: float = 0.0  # dD/dt (emotional salience proxy)
    predicted_dissonance: float = 0.0  # Look-ahead prediction
    horizon: int = 3  # Prediction steps ahead
    
    # Metadata
    source: str = "unknown"
    confidence: float = 0.5


@dataclass
class RegulatedOutput:
    """Governor-regulated output (post-governor)."""
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    mode: RegulationMode = RegulationMode.NORMAL
    
    # Governor decisions
    dampened: bool = False
    boosted: bool = False
    capped: bool = False
    
    # Debug info
    reason: str = ""
    raw_input: Optional[CognitiveSignals] = None


class Governor:
    """
    Central governor for RAVANA v2.
    
    NON-NEGOTIABLE PRINCIPLE:
    All state changes flow through here. No exceptions.
    """
    
    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self.history: List[RegulatedOutput] = []
        self.mode_history: List[RegulationMode] = []
        
        # Plateau tracking
        self.recent_dissonance: List[float] = []
        self.recent_identity: List[float] = []
        
        # 🔮 Grace Layer tracking
        self.predictions_made: int = 0
        self.predictions_correct: int = 0
        self.boundary_pressure_history: List[float] = []
        self.center_force_history: List[float] = []
        self.dampening_activations: int = 0
        self.overshoot_events: int = 0

        # 🔴 CLAMP DIAGNOSTICS: Track controller vs clamp alignment
        self.clamp_corrections_total = 0.0
        self.clamp_activations = 0
        self.upstream_suggestions = 0
        self.clamp_correction_history = []
        
    def regulate(
        self,
        current_dissonance: float,
        current_identity: float,
        signals: CognitiveSignals,
        episode: int = 0
    ) -> RegulatedOutput:
        """
        Central regulation point. ALL state changes pass through here.
        
        Returns regulated deltas that keep system in healthy bounds.
        """
        # Detect current mode
        mode = self._detect_mode(
            current_dissonance,
            current_identity,
            signals
        )
        
        # Apply hard constraints FIRST
        dissonance_delta, identity_delta, constraints = self._apply_hard_constraints(
            current_dissonance,
            current_identity,
            signals
        )
        

        # 🔮 PHASE B.0: Grace Layer — Predictive & Soft Regulation
        # 1. Look ahead: dampen based on predicted future state
        dissonance_delta = self._predictive_dampening(current_dissonance, dissonance_delta, signals)
        
        # 2. Feel the boundary: air resistance near limits
        dissonance_delta = self._boundary_pressure(current_dissonance, dissonance_delta)
        
        # 3. Return to center: homeostatic pull
        dissonance_delta = self._center_seeking_force(current_dissonance, dissonance_delta)
        
        # Apply mode-specific regulation
        dissonance_delta, identity_delta = self._apply_mode_regulation(
            mode,
            dissonance_delta,
            identity_delta,
            current_dissonance,
            current_identity
        )
        
        # Build output
        output = RegulatedOutput(
            dissonance_delta=dissonance_delta,
            identity_delta=identity_delta,
            mode=mode,
            dampened=constraints.get('dampened', False),
            boosted=constraints.get('boosted', False),
            capped=constraints.get('capped', False),
            reason=constraints.get('reason', ''),
            raw_input=signals
        )
        
        # Track history
        self.history.append(output)
        self.mode_history.append(mode)
        self._update_tracking(current_dissonance, current_identity)
        
        return output
    

    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Return health metrics for the Grace Layer.
        """
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        
        return {
            'predictions_made': self.predictions_made,
            'boundary_pressure_avg': sum(self.boundary_pressure_history[-10:]) / len(self.boundary_pressure_history[-10:]) if self.boundary_pressure_history else 0,
            'center_force_avg': sum(self.center_force_history[-10:]) / len(self.center_force_history[-10:]) if self.center_force_history else 0,
            'overshoot_count': sum(1 for r in recent if r.capped),
            'mean_approach_velocity': np.mean([abs(r.dissonance_delta) for r in recent]) if recent else 0.0,
            'total_regulation_events': len(self.history),
            'predictions_made': self.predictions_made,
            'prediction_accuracy': self.predictions_correct / max(1, self.predictions_made),
        }

    def _detect_mode(
        self,
        dissonance: float,
        identity: float,
        signals: CognitiveSignals
    ) -> RegulationMode:
        """Detect which regulation mode to use."""
        
        # CRISIS: Near catastrophic bounds
        if dissonance > self.config.crisis_threshold or identity < self.config.min_identity:
            return RegulationMode.RECOVERY
        
        # PLATEAU: Stagnation detected
        if self._is_plateau():
            return RegulationMode.PLATEAU
        
        # RESOLUTION: High dissonance, ready to resolve
        if dissonance > self.config.resolution_threshold:
            return RegulationMode.RESOLUTION
        
        # EXPLORATION: Low dissonance, seek novelty
        if dissonance < self.config.exploration_threshold:
            return RegulationMode.EXPLORATION
        
        # Normal operation
        return RegulationMode.NORMAL
    
    def _apply_hard_constraints(
        self,
        current_dissonance: float,
        current_identity: float,
        signals: CognitiveSignals
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply hard constraints. These are ABSOLUTE and cannot be overridden.
        """
        constraints = {'dampened': False, 'boosted': False, 'capped': False, 'reason': ''}
        
        dd = signals.dissonance_delta
        id_val = signals.identity_delta
        
        # CEILING: Prevent dissonance > max_dissonance
        projected_d = current_dissonance + dd
        if projected_d > self.config.max_dissonance:
            dd = self.config.max_dissonance - current_dissonance - 0.01  # Stay just below
            constraints['capped'] = True
            constraints['reason'] = f"dissonance_ceiling (proj={projected_d:.3f})"
        
        # FLOOR: Prevent dissonance < min_dissonance
        if projected_d < self.config.min_dissonance:
            dd = self.config.min_dissonance - current_dissonance + 0.01  # Stay just above
            constraints['capped'] = True
            constraints['reason'] += f" dissonance_floor (proj={projected_d:.3f})"
        
        # IDENTITY FLOOR: Prevent identity collapse
        projected_i = current_identity + id_val
        if projected_i < self.config.min_identity:
            id_val = self.config.min_identity - current_identity + 0.01
            constraints['boosted'] = True
            constraints['reason'] += f" identity_floor (proj={projected_i:.3f})"
        
        # IDENTITY CEILING: Prevent identity inflation
        if projected_i > self.config.max_identity:
            id_val = self.config.max_identity - current_identity - 0.01
            constraints['capped'] = True
            constraints['reason'] += f" identity_ceiling (proj={projected_i:.3f})"
        
        return dd, id_val, constraints
    


    def _predictive_dampening(
        self,
        current_d: float,
        dd: float,
        signals: CognitiveSignals
    ) -> float:
        """
        🔮 Look-ahead regulation: dampen based on FUTURE state, not current.
        
        Principle: "Slow down before you see the wall"
        """
        # Predict where we'll be after horizon steps
        predicted_d = current_d + dd * signals.horizon
        signals.predicted_dissonance = predicted_d
        
        # If prediction exceeds threshold, apply early dampening
        threshold = self.config.max_dissonance * 0.85  # Start early
        if predicted_d > threshold:
            # Progressive reduction: stronger as we approach limit
            overshoot = predicted_d - threshold
            reduction = 1.0 / (1.0 + overshoot * 2.0)  # Smooth decay
            dd *= reduction
            self.predictions_made += 1
            if hasattr(self, '_last_log') and self._last_log:
                print(f"  [PREDICTIVE] D={current_d:.3f} → predicted={predicted_d:.3f}, reducing dd by {reduction:.2f}x")
        
        return dd

    def _boundary_pressure(
        self,
        current_d: float,
        dd: float
    ) -> float:
        """
        🌊 Soft boundary pressure: air resistance, not brick wall.
        
        Starts subtle, becomes dominant near boundary.
        Returns dampened dd.
        """
        if current_d < self.config.soft_limit:
            # Track zero pressure for metrics
            if hasattr(self, 'boundary_pressure_history'):
                self.boundary_pressure_history.append(0.0)
                if len(self.boundary_pressure_history) > 100:
                    self.boundary_pressure_history.pop(0)
            return dd  # No pressure below soft limit
        
        # Sigmoid pressure curve
        excess = current_d - self.config.soft_limit
        k = getattr(self.config, 'boundary_k', 10.0)
        
        # Sigmoid: smooth transition from 0 to 1
        import math
        pressure = 1.0 / (1.0 + math.exp(-k * (excess - 0.05)))
        
        # Apply pressure: more resistance as we approach limit
        dampened_dd = dd * (1.0 - pressure * 0.8)  # Max 80% reduction
        
        # Track for metrics
        if hasattr(self, 'boundary_pressure_history'):
            self.boundary_pressure_history.append(pressure)
            if len(self.boundary_pressure_history) > 100:
                self.boundary_pressure_history.pop(0)
        
        return dampened_dd

    def _apply_mode_regulation(
        self,
        mode: RegulationMode,
        dd: float,
        id_val: float,
        current_d: float,
        current_i: float
    ) -> Tuple[float, float]:
        """Apply mode-specific regulation."""
        
        if mode == RegulationMode.RECOVERY:
            # Recovery: aggressive stabilization
            if current_d > self.config.crisis_threshold:
                dd = -0.05  # Force reduction
            if current_i < self.config.min_identity + 0.1:
                id_val = 0.03  # Force boost
                
        elif mode == RegulationMode.PLATEAU:
            # Plateau: controlled perturbation
            dd += np.random.normal(0, 0.03)
            id_val += np.random.normal(0, 0.01)
            
        elif mode == RegulationMode.RESOLUTION:
            # Resolution: adaptive amplification based on safety
            # Safe (low D): amplify freely
            # Near boundary: dampen to avoid overshoot
            safety_factor = 1.0 - current_d  # 1.0 at D=0, 0.0 at D=1.0
            amplification = 1.0 + (0.2 * safety_factor)  # 1.2 at D=0, 1.0 at D=1.0
            dd *= amplification
            
            if dd < 0:  # Reducing dissonance
                # Identity reward scales with safety
                id_val += 0.01 * (0.5 + 0.5 * safety_factor)
                
        elif mode == RegulationMode.EXPLORATION:
            # Exploration: maintain curiosity
            if abs(dd) < 0.01:  # Too stable
                dd = np.random.choice([-0.02, 0.02])  # Induce small variation
        
        # Normal mode: no additional modification
        
        # FINAL HARD CLAMP: Absolute enforcement after all processing
        # Clamp dissonance delta
        max_allowed_d = self.config.max_dissonance - current_d
        min_allowed_d = self.config.min_dissonance - current_d
        dd = np.clip(dd, min_allowed_d - 0.01, max_allowed_d - 0.01)
        
        # 🔴 CRITICAL FIX: Clamp identity delta too (was missing!)
        max_allowed_i = self.config.max_identity - current_i
        min_allowed_i = self.config.min_identity - current_i
        
        # 📊 CLAMP DIAGNOSTIC: Track controller/clamp alignment
        id_val_before = id_val
        id_val = np.clip(id_val, min_allowed_i + 0.01, max_allowed_i - 0.01)
        correction = abs(id_val_before - id_val)
        
        if correction > 0.001:  # Clamp actually changed the value
            self.clamp_activations += 1
            self.clamp_corrections_total += correction
            self.clamp_correction_history.append(correction)
            if len(self.clamp_correction_history) > 100:
                self.clamp_correction_history.pop(0)
            
            # Log if clamp is fighting upstream significantly
            if correction > 0.05:  # More than 5% correction
                print(f"  [CLAMP ALERT] Upstream suggested {id_val_before:+.3f}, clamped to {id_val:+.3f} (Δ{correction:.3f})")
        
        self.upstream_suggestions += 1
        
        return dd, id_val
    
    def _is_plateau(self) -> bool:
        """Detect if system is in plateau (stagnation)."""
        if len(self.recent_dissonance) < self.config.plateau_window:
            return False
        
        recent = self.recent_dissonance[-self.config.plateau_window:]
        variance = np.var(recent)
        
        return variance < self.config.plateau_tolerance
    

    def _center_seeking_force(
        self,
        current_d: float,
        dd: float
    ) -> float:
        """
        🎯 Anti-overshoot term: pull toward center when far from it.
        
        Prevents drift accumulation, creates homeostasis.
        """
        target = self.config.target_dissonance
        distance_from_center = current_d - target
        
        # Only apply when far from center
        if abs(distance_from_center) > 0.15:
            # Gentle pull toward center
            k_center = 0.1
            center_force = -distance_from_center * k_center
            dd += center_force
            
            if hasattr(self, '_last_log') and self._last_log:
                direction = "→" if center_force > 0 else "←"
                print(f"  [CENTER] D={current_d:.3f} {direction} target={target:.3f} (force={center_force:.4f})")
        
        return dd

    def _update_tracking(self, dissonance: float, identity: float):
        """Update tracking buffers."""
        self.recent_dissonance.append(dissonance)
        self.recent_identity.append(identity)
        
        # Keep only necessary history
        max_len = self.config.plateau_window + 10
        if len(self.recent_dissonance) > max_len:
            self.recent_dissonance = self.recent_dissonance[-max_len:]
            self.recent_identity = self.recent_identity[-max_len:]
    
    def get_status(self) -> Dict[str, Any]:
        """Return governor status for monitoring."""
        if not self.history:
            return {"mode": "unknown", "cycles": 0}
        
        recent_modes = self.mode_history[-20:]
        mode_counts = {}
        for m in recent_modes:
            mode_counts[m.value] = mode_counts.get(m.value, 0) + 1
        
        return {
            "current_mode": self.mode_history[-1].value if self.mode_history else "unknown",
            "mode_distribution": mode_counts,
            "total_cycles": len(self.history),
            "constraint_activations": sum(1 for h in self.history if h.capped or h.dampened),
            "recent_dissonance_variance": np.var(self.recent_dissonance) if self.recent_dissonance else 0,
        }
