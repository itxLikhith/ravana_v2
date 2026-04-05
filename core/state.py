"""
RAVANA v2 — COGNITIVE STATE
Unified state container with governor-gated updates.

PRINCIPLE: State is immutable except through official channels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class CognitivePhase(Enum):
    """Cognitive processing phases."""
    PERCEPTION = "perception"
    RESOLUTION = "resolution"
    INTEGRATION = "integration"


@dataclass
class CognitiveState:
    """
    Immutable cognitive state container.
    
    All modifications go through StateManager.update() which applies governor.
    """
    # Core metrics
    dissonance: float = 0.5
    identity: float = 0.5
    dissonance_ema: float = 0.5  # Smoothed dissonance for regulation
    
    # Episode tracking
    episode: int = 0
    cycle: int = 0
    
    # Learning accumulators
    accumulated_wisdom: float = 0.0
    resolution_streak: int = 0
    
    # Debug
    last_update_reason: str = "initial"
    constraint_activated: bool = False
    
    def snapshot(self) -> Dict[str, float]:
        """Return serializable snapshot."""
        return {
            "dissonance": self.dissonance,
            "dissonance_ema": self.dissonance_ema,
            "identity": self.identity,
            "episode": self.episode,
            "cycle": self.cycle,
            "wisdom": self.accumulated_wisdom,
        }


class StateManager:
    """
    Central state manager with governor integration.
    
    ALL state modifications flow through here.
    """
    
    def __init__(self, governor, resolution_engine, identity_engine, smoothing_alpha: float = 0.2):
        from .memory import RavanaMemorySystem
        self.state = CognitiveState()
        self.governor = governor
        self.resolution = resolution_engine
        self.identity = identity_engine
        self.memory = RavanaMemorySystem()
        self.smoothing_alpha = smoothing_alpha
        
        # History for analysis
        self.history: list = []
        
    def step(
        self,
        correctness: bool,
        difficulty: float = 0.5,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute one cognitive step with full governor regulation.
        
        This is the ONLY way state changes. No exceptions.
        """
        # Capture pre-update state
        pre_d = self.state.dissonance
        pre_i = self.state.identity
        
        # 1. ESTIMATE: Predict desired delta from outcome
        # If correct: dissonance should drop. If wrong: it should rise.
        # This is what the system WANTS to do.
        desired_d_delta = -0.1 if correctness else 0.15
        
        # 2. IDENTITY: Compute desired update based on ESTIMATED resolution
        # We use a placeholder for success here, governor will regulate final
        est_res_success = (desired_d_delta < 0 and correctness)
        desired_identity = self.identity.compute_update(
            resolution_delta=abs(desired_d_delta),
            resolution_success=est_res_success,
            regulated_identity_delta=0.0,
            current_dissonance=pre_d
        )
        identity_delta = desired_identity - pre_i
        
        # 3. GOVERNOR: Regulate all changes
        from .governor import CognitiveSignals
        
        signals = CognitiveSignals(
            dissonance_delta=desired_d_delta,
            identity_delta=identity_delta,
            exploration_drive=0.0,
            resolution_potential=0.1 if correctness else 0.0,
            source="state_step"
        )
        
        # Use EMA dissonance if governor prefers
        reg_d = self.state.dissonance_ema if getattr(self.governor.config, 'use_smoothed_dissonance', False) else pre_d
        
        regulated = self.governor.regulate(
            current_dissonance=reg_d,
            current_identity=pre_i,
            signals=signals,
            episode=self.state.episode
        )
        
        # 4. APPLY: Governor-approved changes
        new_dissonance = np.clip(
            pre_d + regulated.dissonance_delta,
            self.governor.config.min_dissonance,
            self.governor.config.max_dissonance
        )
        
        # Update EMA
        new_ema = (1 - self.smoothing_alpha) * self.state.dissonance_ema + self.smoothing_alpha * new_dissonance
        
        # Identity update: use governor-regulated delta
        regulated_identity = self.identity.compute_update(
            resolution_delta=abs(regulated.dissonance_delta),
            resolution_success=(regulated.dissonance_delta < 0 and correctness),
            regulated_identity_delta=regulated.identity_delta,
            current_dissonance=pre_d
        )
        new_identity = np.clip(
            regulated_identity,
            self.governor.config.min_identity,
            self.governor.config.max_identity
        )
        
        # 5. RESOLUTION: Compute what ACTUALLY happened
        # Now we pass actual pre vs post dissonance
        resolution_result = self.resolution.compute(
            episode=self.state.episode,
            prev_dissonance=pre_d,
            current_dissonance=new_dissonance,
            correctness=correctness,
            difficulty=difficulty,
            source="episode_step"
        )
        
        # 6. WISDOM: Check for generation
        wisdom_generated = resolution_result["wisdom_generated"]
        
        # 7. UPDATE STATE
        self.state = CognitiveState(
            dissonance=new_dissonance,
            identity=new_identity,
            dissonance_ema=new_ema,
            episode=self.state.episode + 1,
            cycle=self.state.cycle + 1,
            accumulated_wisdom=self.state.accumulated_wisdom + wisdom_generated,
            resolution_streak=resolution_result["streak"],
            last_update_reason=regulated.reason,
            constraint_activated=regulated.capped or regulated.dampened
        )
        
        # Track history
        step_record = {
            "episode": self.state.episode,
            "pre_dissonance": pre_d,
            "post_dissonance": new_dissonance,
            "pre_identity": pre_i,
            "post_identity": new_identity,
            "resolution": resolution_result,
            "mode": regulated.mode.value,
            "wisdom": wisdom_generated,
            "reason": regulated.reason,
        }
        self.history.append(step_record)
        
        # 8. MEMORY: Integrate new data
        self.memory.process_step(
            episode_data=step_record,
            state_snapshot=self.state.snapshot()
        )
        
        if debug:
            self._log_step(step_record)
        
        return step_record
    
    def _log_step(self, record: Dict[str, Any]):
        """Debug logging."""
        print(f"  [EP{record['episode']:04d}] "
              f"D:{record['pre_dissonance']:.3f}→{record['post_dissonance']:.3f} "
              f"I:{record['pre_identity']:.3f}→{record['post_identity']:.3f} "
              f"Mode:{record['mode'][:4]} "
              f"Res:{'✓' if record['resolution']['full_resolution'] else '·'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Full system status."""
        return {
            "state": self.state.snapshot(),
            "governor": self.governor.get_status(),
            "identity": self.identity.get_status(),
            "resolution": self.resolution.get_memory_status(),
            "total_steps": len(self.history),
        }
