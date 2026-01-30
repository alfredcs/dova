"""DOVA Agents Module."""

from dova.agents.base import BaseAgent, AgentResult, AgentTask
from dova.agents.orchestrator import DOVAOrchestrator
from dova.agents.research import ResearchAgent
from dova.agents.profiling import ProfilingAgent
from dova.agents.validation import ValidationAgent
from dova.agents.synthesis import SynthesisAgent
from dova.agents.debate import DebateAgent, BullAgent, BearAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentTask",
    "DOVAOrchestrator",
    "ResearchAgent",
    "ProfilingAgent",
    "ValidationAgent",
    "SynthesisAgent",
    "DebateAgent",
    "BullAgent",
    "BearAgent",
]
