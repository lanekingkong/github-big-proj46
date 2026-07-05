"""BridgeGate: Risk-Based CI/CD Quality Gating System.

BridgeGate implements an intelligent quality gate that makes automated go/no-go
decisions for code promotion through the CI/CD pipeline. Unlike traditional
binary pass/fail gates, BridgeGate uses risk-based routing to reduce review
bottlenecks while maintaining safety standards.

Key Features:
- Multi-dimensional risk scoring (security, complexity, test coverage, etc.)
- Configurable gate policies per environment (dev/staging/production)
- Smart routing: auto-approve low-risk, require review for medium, block for high
- Integration with GitHub Actions, GitLab CI, Jenkins
- Graduated deployment: canary → staging → production
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class GateDecision(Enum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"


@dataclass
class GateCheck:
    name: str
    passed: bool
    score: float
    threshold: float
    details: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class GateResult:
    environment: Environment
    decision: GateDecision
    overall_score: float
    checks: list[GateCheck] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BridgeGate:
    """Risk-based CI/CD quality gating engine.

    Usage:
        gate = BridgeGate()
        result = gate.evaluate(
            env=Environment.PRODUCTION,
            security_score=8.5,
            review_score=7.0,
            test_coverage=82.0,
            complexity_delta=5,
        )
        if result.decision == GateDecision.PASS:
            print("Ready for deployment!")
        else:
            print(f"Blocked: {result.blocked_by}")
    """

    DEFAULT_POLICIES = {
        "dev": {
            "security_score": {"threshold": 3.0, "weight": 0.15},
            "review_score": {"threshold": 5.0, "weight": 0.20},
            "test_coverage": {"threshold": 50.0, "weight": 0.15},
            "complexity_delta": {"threshold": 20, "weight": 0.10},
            "lint_score": {"threshold": 5.0, "weight": 0.10},
            "dependency_health": {"threshold": 5.0, "weight": 0.10},
            "breaking_changes": {"threshold": 5, "weight": 0.20},
        },
        "staging": {
            "security_score": {"threshold": 6.0, "weight": 0.20},
            "review_score": {"threshold": 7.0, "weight": 0.20},
            "test_coverage": {"threshold": 70.0, "weight": 0.20},
            "complexity_delta": {"threshold": 10, "weight": 0.10},
            "lint_score": {"threshold": 7.0, "weight": 0.10},
            "dependency_health": {"threshold": 7.0, "weight": 0.10},
            "breaking_changes": {"threshold": 2, "weight": 0.10},
        },
        "canary": {
            "security_score": {"threshold": 7.0, "weight": 0.25},
            "review_score": {"threshold": 7.5, "weight": 0.20},
            "test_coverage": {"threshold": 75.0, "weight": 0.20},
            "complexity_delta": {"threshold": 8, "weight": 0.10},
            "lint_score": {"threshold": 7.5, "weight": 0.10},
            "dependency_health": {"threshold": 7.5, "weight": 0.10},
            "breaking_changes": {"threshold": 1, "weight": 0.05},
        },
        "production": {
            "security_score": {"threshold": 8.0, "weight": 0.30},
            "review_score": {"threshold": 8.0, "weight": 0.25},
            "test_coverage": {"threshold": 80.0, "weight": 0.20},
            "complexity_delta": {"threshold": 5, "weight": 0.10},
            "lint_score": {"threshold": 8.0, "weight": 0.05},
            "dependency_health": {"threshold": 8.0, "weight": 0.05},
            "breaking_changes": {"threshold": 0, "weight": 0.05},
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        self.policies = self._load_policies(config_path)
        self._decision_history: list[GateResult] = []

    def _load_policies(self, config_path: Optional[str] = None) -> dict:
        policies = self.DEFAULT_POLICIES.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f)
                if custom:
                    self._deep_merge_policies(policies, custom)
        return policies

    def _deep_merge_policies(self, base: dict, override: dict):
        for env, checks in override.items():
            if env in base:
                base[env].update(checks)
            else:
                base[env] = checks

    def evaluate(self, env: Environment,
                 security_score: float = 10.0,
                 review_score: float = 10.0,
                 test_coverage: float = 100.0,
                 complexity_delta: int = 0,
                 lint_score: float = 10.0,
                 dependency_health: float = 10.0,
                 breaking_changes: int = 0,
                 custom_metrics: Optional[dict] = None) -> GateResult:
        """Evaluate code readiness against environment-specific policies.

        Args:
            env: Target deployment environment
            security_score: Security audit score (0-10)
            review_score: Code review score (0-10)
            test_coverage: Test coverage percentage (0-100)
            complexity_delta: Change in cyclomatic complexity
            lint_score: Code style/lint score (0-10)
            dependency_health: Dependency security score (0-10)
            breaking_changes: Number of breaking API changes
            custom_metrics: Additional custom metrics

        Returns:
            GateResult with decision and detailed check results
        """
        policy = self.policies.get(env.value, self.policies["production"])

        metrics = {
            "security_score": security_score,
            "review_score": review_score,
            "test_coverage": test_coverage,
            "complexity_delta": complexity_delta,
            "lint_score": lint_score,
            "dependency_health": dependency_health,
            "breaking_changes": breaking_changes,
        }
        if custom_metrics:
            metrics.update(custom_metrics)

        checks = []
        weighted_sum = 0.0
        total_weight = 0.0
        blocked_by = []
        warnings = []

        for check_name, check_policy in policy.items():
            metric_value = metrics.get(check_name)
            if metric_value is None:
                continue

            threshold = check_policy.get("threshold", 5.0)
            weight = check_policy.get("weight", 0.0)

            # Some metrics are "higher is better", others "lower is better"
            if check_name in ("complexity_delta", "breaking_changes"):
                passed = metric_value <= threshold
                # Normalize score: 0 is best, threshold is worst
                normalized = max(0.0, 10.0 * (1 - metric_value / max(threshold, 1)))
            else:
                passed = metric_value >= threshold
                # Normalize to 0-10
                normalized = min(10.0, metric_value)

            checks.append(GateCheck(
                name=check_name,
                passed=passed,
                score=round(normalized, 1),
                threshold=threshold,
                details=f"{metric_value} vs threshold {threshold}",
            ))

            weighted_sum += normalized * weight
            total_weight += weight

            if not passed:
                if weight >= 0.2:  # High-weight failures block
                    blocked_by.append(check_name)
                else:
                    warnings.append(f"{check_name}: {metric_value} (threshold: {threshold})")

        overall_score = weighted_sum / max(total_weight, 0.01)

        # Decision logic
        if blocked_by:
            decision = GateDecision.BLOCKED
        elif overall_score >= 8.0:
            decision = GateDecision.PASS
        elif overall_score >= 6.0:
            decision = GateDecision.PASS_WITH_WARNINGS
        else:
            decision = GateDecision.NEEDS_REVIEW

        # Generate recommendations
        recommendations = []
        for check in checks:
            if not check.passed:
                if check.name == "security_score":
                    recommendations.append("Run BridgeSecure full scan and fix all HIGH/CRITICAL findings")
                elif check.name == "review_score":
                    recommendations.append("Address review findings and request re-review")
                elif check.name == "test_coverage":
                    recommendations.append(f"Add tests to increase coverage above {check.threshold}%")
                elif check.name == "complexity_delta":
                    recommendations.append("Refactor complex code into smaller, focused functions")
                elif check.name == "breaking_changes":
                    recommendations.append("Document breaking changes and ensure backward compatibility or version bump")

        result = GateResult(
            environment=env,
            decision=decision,
            overall_score=round(overall_score, 1),
            checks=checks,
            blocked_by=blocked_by,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "policies_used": list(policy.keys()),
                "custom_metrics_provided": list(custom_metrics.keys()) if custom_metrics else [],
            },
        )

        self._decision_history.append(result)
        return result

    def evaluate_from_review(self, review_result, env: Environment = Environment.STAGING,
                             security_result=None, test_coverage: float = 80.0) -> GateResult:
        """Convenience method to evaluate from BridgeReview and BridgeSecure results."""
        review_score = review_result.score if hasattr(review_result, "score") else 7.0
        security_score = 10.0
        if security_result:
            security_score = max(0, 10.0 - security_result.risk_score / 10)

        return self.evaluate(
            env=env,
            security_score=security_score,
            review_score=review_score,
            test_coverage=test_coverage,
        )

    def get_decision_history(self) -> list[GateResult]:
        return self._decision_history

    def export_gate_status(self, result: GateResult) -> dict:
        """Export gate result as JSON for CI/CD pipeline integration."""
        return {
            "decision": result.decision.value,
            "environment": result.environment.value,
            "overall_score": result.overall_score,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": c.score,
                    "threshold": c.threshold,
                }
                for c in result.checks
            ],
            "blocked_by": result.blocked_by,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
        }

    def generate_badge(self, result: GateResult) -> str:
        """Generate a shield.io badge URL for the gate status."""
        color_map = {
            GateDecision.PASS: "brightgreen",
            GateDecision.PASS_WITH_WARNINGS: "yellow",
            GateDecision.NEEDS_REVIEW: "orange",
            GateDecision.BLOCKED: "red",
        }
        color = color_map.get(result.decision, "lightgrey")
        score_text = str(result.overall_score)
        return f"https://img.shields.io/badge/codebridge-{score_text}%2F10-{color}"
