"""
DOVA - Deep Orchestrated Versatile Agent Platform

A production-grade multi-agent research automation system built on
AWS Strands Agents SDK and Amazon Bedrock AgentCore.
"""

__version__ = "0.1.0"
__author__ = "DOVA Team"

from dova.agents.orchestrator import DOVAOrchestrator
from dova.agents.research import ResearchAgent
from dova.agents.profiling import ProfilingAgent
from dova.agents.validation import ValidationAgent
from dova.config.settings import Settings

__all__ = [
    "DOVAOrchestrator",
    "ResearchAgent",
    "ProfilingAgent",
    "ValidationAgent",
    "Settings",
    "__version__",
]
