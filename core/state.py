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
    
    def __init__(self, governor, resolution_engine, identity_engine):
        from .memory import RavanaMemorySystem
        self.state = CognitiveState()
        self.governor = governor
        self.resolution = resolution_engine
        self.identity = identity_engine
        self.memory = RavanaMemorySystem()
        
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
        
        # 1. RESOLUTION: Compute what happened
        resolution_result = self.resolution.compute(
            episode=self.state.episode,
            prev_dissonance=pre_d,
            current_dissonance=pre_d,  # Will be updated after governor
            correctness=correctness,
            difficulty=difficulty,
            source="episode_step"
        )
        
        # 2. IDENTITY: Compute desired update
        desired_identity = self.identity.compute_update(
            resolution_delta=resolution_result["delta"],
            resolution_success=resolution_result["full_resolution"],
            regulated_identity_delta=0.0,  # Start with 0, governor will modify
            current_dissonance=pre_d
        )
        identity_delta = desired_identity - pre_i
        
        # 3. GOVERNOR: Regulate all changes
        from .governor import CognitiveSignals
        
        signals = CognitiveSignals(
            dissonance_delta=resolution_result["delta"],
            identity_delta=identity_delta,
            exploration_drive=0.0,  # Simplified for Phase A
            resolution_potential=resolution_result["partial_credit"],
            source="state_step"
        )
        
        regulated = self.governor.regulate(
            current_dissonance=pre_d,
            current_identity=pre_i,
            signals=signals,
            episode=self.state.episode
        )
        
        # 4. APPLY: Governor-approved changes only
        new_dissonance = np.clip(
            pre_d + regulated.dissonance_delta,
            self.governor.config.min_dissonance,
            self.governor.config.max_dissonance
        )
        
        # Identity update: use governor-regulated delta
        regulated_identity = self.identity.compute_update(
            resolution_delta=resolution_result["delta"],
            resolution_success=resolution_result["full_resolution"],
            regulated_identity_delta=regulated.identity_delta,
            current_dissonance=pre_d
        )
        new_identity = np.clip(
            regulated_identity,
            self.governor.config.min_identity,
            self.governor.config.max_identity
        )
        
        # 5. WISDOM: Check for generation
        wisdom_generated = resolution_result["wisdom_generated"]
        
        # 6. UPDATE STATE
        self.state = CognitiveState(
            dissonance=new_dissonance,
            identity=new_identity,
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
        
        # 7. MEMORY: Integrate new data
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
