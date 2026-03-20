"""
RAVANA v2 Core Modules

Phase A: Governor + Resolution + Identity
Phase B: Adaptation (learning from clamp events)
Phase B.5: Strategy (deliberate mode selection)
Phase C: Strategy Learning (rules → learned preferences)
Phase D: Intent Engine (dynamic objectives → "wants")
"""

from .governor import Governor, GovernorConfig, RegulationMode, ClampDiagnostics
from .resolution import ResolutionEngine, ResolutionMemory
from .identity import IdentityEngine, IdentityState
from .state import StateManager, CognitiveState
from .adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from .strategy import (
    StrategyLayer, 
    StrategyConfig, 
    ExplorationMode,
    ModeSelection,
    BehavioralContext
)
from .strategy_learning import (
    StrategyLearningLayer,
    ModeOutcome,
    LearningConfig,
    StrategyWithLearning
)
from .intent import (
    IntentEngine,
    IntentConfig,
    SystemObjective,
    IntentAwareStrategy
)

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
    # Phase B.5
    "StrategyLayer",
    "StrategyConfig",
    "ExplorationMode",
    "ModeSelection",
    "BehavioralContext",
    # Phase C
    "StrategyLearningLayer",
    "ModeOutcome",
    "LearningConfig",
    "StrategyWithLearning",
    # Phase D
    "IntentEngine",
    "IntentConfig",
    "SystemObjective",
    "IntentAwareStrategy",
]
