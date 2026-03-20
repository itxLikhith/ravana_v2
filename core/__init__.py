"""
RAVANA v2 Core Modules
"""

from .governor import Governor, GovernorConfig, RegulationMode, ClampDiagnostics
from .resolution import ResolutionEngine, ResolutionMemory
from .identity import IdentityEngine, IdentityState
from .state import StateManager, CognitiveState
from .adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from .strategy import StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext
from .strategy_learning import StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning
from .intent import IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective
from .planning import MicroPlanner, PlanningConfig, SimulatedFuture
from .environment import NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState
from .predictive_world import LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester
from .belief_reasoner import BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent
from .active_epistemology import ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector

__all__ = [
    "Governor", "GovernorConfig", "RegulationMode", "ClampDiagnostics",
    "ResolutionEngine", "ResolutionMemory",
    "IdentityEngine", "IdentityState",
    "StateManager", "CognitiveState",
    "PolicyTweakLayer", "AdaptiveGovernorBridge", "AdaptationConfig",
    "StrategyLayer", "StrategyConfig", "ExplorationMode", "ModeSelection", "BehavioralContext",
    "StrategyLearningLayer", "ModeOutcome", "LearningConfig", "StrategyWithLearning",
    "IntentEngine", "IntentConfig", "IntentAwareStrategy", "SystemObjective",
    "MicroPlanner", "PlanningConfig", "SimulatedFuture",
    "NonStationaryEnvironment", "EnvironmentConfig", "HiddenDynamics", "WorldState",
    "LearnedWorldModel", "WorldModelConfig", "PredictedState", "AnomalyEvent", "FalseWorldTester",
    "BeliefReasoner", "BeliefConfig", "Hypothesis", "EvidenceEvent",
    "ActiveEpistemology", "VoIConfig", "InformationGainMethod", "HypothesisDrivenActionSelector",
    "OccamLayer",
    "OccamConfig", 
    "HypothesisScore",
    "DisciplinedBeliefSystem",
]

# Phase G.5
from .surgical_probes import (
    SurgicalProbeSelector,
    SurgicalProbeConfig,
    ProbeType,
    ProbeExperiment,
    SurgicalProbing
)

__all__.extend([
    "SurgicalProbeSelector",
    "SurgicalProbeConfig",
    "ProbeType",
    "ProbeExperiment",
    "SurgicalProbing",
    # Phase J
    "HypothesisGenerator",
    "GenerationConfig",
    "HypothesisType",
    "GeneratedHypothesis"
    "OccamLayer",
    "OccamConfig", 
    "HypothesisScore",
    "DisciplinedBeliefSystem",
])

# Phase J
from .hypothesis_generation import (
from .occam_layer import OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem
    HypothesisGenerator,
    GenerationConfig,
    HypothesisType,
    GeneratedHypothesis
)
