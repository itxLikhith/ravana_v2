"""
RAVANA v2 Core Modules

Phase A: Governor + Resolution + Identity
Phase B: Adaptation (learning from clamp events)
"""

from .governor import Governor, GovernorConfig, RegulationMode, ClampDiagnostics
from .resolution import ResolutionEngine, ResolutionMemory
from .identity import IdentityEngine, IdentityState
from .state import StateManager, CognitiveState
from .adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig

__all__ = [
    # Phase A
    "Governor",
    "GovernorConfig",
    "RegulationMode",
    "ClampDiagnostics",
    "ResolutionEngine",
    "ResolutionMemory",
    "IdentityEngine",
    "IdentityState",
    "StateManager",
    "CognitiveState",
    # Phase B
    "PolicyTweakLayer",
    "AdaptiveGovernorBridge",
    "AdaptationConfig",
]
