"""
RAVANA v2 Core Modules

Phase A: Governor + Resolution + Identity
Phase B: Adaptation (learning from clamp events)
Phase B.5: Strategy (deliberate mode selection)
Phase C: Strategy Learning (rules → learned preferences)
Phase D: Intent (dynamic objective formation)
Phase D.5: Planning (future-simulation for anticipatory goals)
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
    IntentAwareStrategy,
    SystemObjective
)
from .planning import (
    MicroPlanner,
    PlanningConfig,
    SimulatedFuture
)
from .environment import (
    NonStationaryEnvironment,
    EnvironmentConfig,
    HiddenDynamics,
    WorldState
)
from .predictive_world import (
    LearnedWorldModel,
    WorldModelConfig,
    Prediction,
    AnomalyEvent,
    FalseWorldTester,
)
from .belief_reasoner import (
    BeliefReasoner,
    BeliefConfig,
    Hypothesis,
    EvidenceEvent,
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
    "IntentAwareStrategy",
    "SystemObjective",
    # Phase D.5
    "MicroPlanner",
    "PlanningConfig",
    "SimulatedFuture",
    # Phase E
    "NonStationaryEnvironment",
    "EnvironmentConfig",
    "HiddenDynamics",
    "WorldState",
    # Phase F
    "LearnedWorldModel",
    "WorldModelConfig",
    "Prediction",
    "AnomalyEvent",
    "FalseWorldTester",
    # Phase F.5
    "BeliefReasoner",
    "BeliefConfig",
    "Hypothesis",
    "EvidenceEvent",
]
