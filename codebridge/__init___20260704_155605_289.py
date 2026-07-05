"""CodeBridge — AI Code Governance Platform.

CodeBridge is a full-lifecycle AI code governance platform that bridges the gap
between AI code generation and production deployment. It addresses the #1 pain
point in modern software development: AI writes code faster than humans can review,
creating a dangerous gap between generation velocity and quality assurance.

Six integrated subsystems:
    BridgeGraph  — Code Knowledge Graph for structural understanding
    BridgeReview — Multi-agent AI code review with risk-based routing
    BridgeSecure — AI-powered security vulnerability detection
    BridgeFlow   — Context compression for LLM token optimization
    BridgeGate   — Risk-based CI/CD quality gating
    BridgeMetrics — Real-time delivery analytics and governance

Quick Start:
    pip install codebridge
    codebridge init
    codebridge scan .
    codebridge review --base main --head feature/new-api
    codebridge gate --env production
"""

__version__ = "0.1.0"
__author__ = "lanekingkong"
__license__ = "Apache-2.0"

from codebridge.core.graph_engine import BridgeGraph, GraphStats
from codebridge.core.review_engine import BridgeReview, ReviewResult, ReviewFinding
from codebridge.core.security_engine import BridgeSecure, SecurityReport, Vulnerability
from codebridge.core.context_engine import BridgeFlow, CompressionStats, CompressionLevel
from codebridge.core.gate_engine import BridgeGate, GateResult, GateDecision, Environment
from codebridge.core.metrics_engine import BridgeMetrics, PipelineMetrics

__all__ = [
    # Graph
    "BridgeGraph",
    "GraphStats",
    # Review
    "BridgeReview",
    "ReviewResult",
    "ReviewFinding",
    # Security
    "BridgeSecure",
    "SecurityReport",
    "Vulnerability",
    # Context
    "BridgeFlow",
    "CompressionStats",
    "CompressionLevel",
    # Gate
    "BridgeGate",
    "GateResult",
    "GateDecision",
    "Environment",
    # Metrics
    "BridgeMetrics",
    "PipelineMetrics",
]
