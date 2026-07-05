"""BridgeReview: Multi-Agent AI Code Review System.

Inspired by SpecVector (context-aware review), Sakura-AI-Reviewer (structured
review reports), and GitHub Copilot CCR (4-agent parallel review architecture),
BridgeReview orchestrates multiple specialized review agents that analyze code
changes from different dimensions: lint, security, logic, architecture, and
performance. Results are aggregated and prioritized by a master orchestrator.

Key features:
- 5 specialized review agents running in parallel
- Risk-based file classification (SKIP/FAST_PASS/DEEP_DIVE)
- Structured review reports with severity levels
- Incremental review (only changed files on push)
- Knowledge graph integration for architectural context
- Configurable review depth and scope
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from codebridge.core.graph_engine import BridgeGraph


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewStatus(Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENT = "comment"


class FileRisk(Enum):
    SKIP = "skip"
    FAST_PASS = "fast_pass"
    DEEP_DIVE = "deep_dive"


@dataclass
class ReviewFinding:
    id: str
    file_path: str
    line_start: int
    line_end: int
    severity: Severity
    category: str
    title: str
    description: str
    suggestion: str = ""
    rule_id: str = ""
    agent_name: str = ""


@dataclass
class ReviewResult:
    pr_id: str = ""
    status: ReviewStatus = ReviewStatus.COMMENT
    score: float = 0.0
    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    metrics: dict = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class FileClassification:
    file_path: str
    risk: FileRisk
    reason: str
    language: str = ""


class BridgeReview:
    """Multi-agent AI code review orchestrator.

    Usage:
        reviewer = BridgeReview(graph=BridgeGraph())
        reviewer.index_project("/path/to/project")
        result = reviewer.review_pr(
            base_branch="main",
            head_branch="feature/new-api",
            pr_description="Adds new REST API endpoints"
        )
    """

    DEFAULT_CONFIG = {
        "agents": {
            "lint": {"enabled": True, "weight": 0.15},
            "security": {"enabled": True, "weight": 0.30},
            "logic": {"enabled": True, "weight": 0.25},
            "architecture": {"enabled": True, "weight": 0.20},
            "performance": {"enabled": True, "weight": 0.10},
        },
        "risk_rules": {
            "skip_patterns": ["*.md", "*.txt", "*.csv", "*.json", "*.yaml", "*.yml", "LICENSE", "*.lock"],
            "deep_dive_patterns": ["**/auth/**", "**/security/**", "**/payment/**", "**/*.sql", "**/config/**"],
            "deep_dive_extensions": [".sql", ".tf", ".hcl"],
        },
        "thresholds": {
            "approve_score": 8.0,
            "changes_requested_score": 5.0,
            "max_findings_per_file": 15,
            "critical_blocking": True,
        },
    }

    def __init__(self, graph: Optional[BridgeGraph] = None, config_path: Optional[str] = None):
        self.graph = graph or BridgeGraph()
        self.config = self._load_config(config_path)
        self._project_root = ""
        self._review_history: list[ReviewResult] = []

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._deep_merge(config, user_config)
        return config

    def _deep_merge(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def index_project(self, root_path: str):
        """Index the project with the knowledge graph for architectural context."""
        self._project_root = root_path
        self.graph.index_project(root_path)

    def review_pr(self, base_branch: str, head_branch: str,
                  pr_description: str = "", changed_files: Optional[list] = None,
                  diff_content: Optional[str] = None) -> ReviewResult:
        """Run a full PR review with all enabled agents.

        Args:
            base_branch: Target branch name
            head_branch: Source branch name
            pr_description: PR description text
            changed_files: List of changed file paths
            diff_content: Raw git diff content

        Returns:
            ReviewResult with findings, score, and status
        """
        import time
        start = time.time()

        if diff_content:
            changed_files = self._parse_diff_files(diff_content)

        if not changed_files:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                score=10.0,
                summary="No files to review.",
            )

        classifications = self._classify_files(changed_files)
        findings: list[ReviewFinding] = []

        # Run all enabled agents in parallel (simulated)
        agent_results = {}
        agents_config = self.config.get("agents", {})
        for agent_name, agent_cfg in agents_config.items():
            if agent_cfg.get("enabled", True):
                agent_results[agent_name] = self._run_agent(
                    agent_name, classifications, diff_content or "", pr_description
                )

        # Aggregate findings from all agents
        for agent_name, agent_findings in agent_results.items():
            for f in agent_findings:
                f.agent_name = agent_name
                findings.append(f)

        # Calculate score
        score = self._calculate_score(findings)

        # Determine status
        thresholds = self.config.get("thresholds", {})
        has_critical = any(f.severity == Severity.CRITICAL for f in findings)
        if has_critical and thresholds.get("critical_blocking", True):
            status = ReviewStatus.CHANGES_REQUESTED
        elif score >= thresholds.get("approve_score", 8.0):
            status = ReviewStatus.APPROVED
        elif score >= thresholds.get("changes_requested_score", 5.0):
            status = ReviewStatus.COMMENT
        else:
            status = ReviewStatus.CHANGES_REQUESTED

        summary = self._generate_summary(findings, score, status, classifications)

        result = ReviewResult(
            pr_id=f"{base_branch}..{head_branch}",
            status=status,
            score=round(score, 1),
            findings=findings,
            summary=summary,
            metrics={
                "files_reviewed": len(changed_files),
                "findings_count": len(findings),
                "agents_used": list(agent_results.keys()),
                "deep_dive_count": sum(1 for c in classifications if c.risk == FileRisk.DEEP_DIVE),
            },
            duration_ms=round((time.time() - start) * 1000, 1),
        )

        self._review_history.append(result)
        return result

    def _parse_diff_files(self, diff_content: str) -> list[str]:
        """Extract changed file paths from git diff output."""
        files = set()
        for line in diff_content.split("\n"):
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line[6:] if line.startswith("+++ b/") else line[6:]
                if path != "/dev/null":
                    files.add(path)
        return list(files)

    def _classify_files(self, file_paths: list[str]) -> list[FileClassification]:
        """Classify files by risk level for review depth."""
        import fnmatch
        classifications = []
        rules = self.config.get("risk_rules", {})

        for fp in file_paths:
            # Check skip patterns
            should_skip = False
            for pattern in rules.get("skip_patterns", []):
                if fnmatch.fnmatch(os.path.basename(fp), pattern):
                    classifications.append(FileClassification(fp, FileRisk.SKIP, f"Matches skip pattern: {pattern}"))
                    should_skip = True
                    break
            if should_skip:
                continue

            # Check deep dive patterns
            is_deep = False
            for pattern in rules.get("deep_dive_patterns", []):
                if fnmatch.fnmatch(fp, pattern):
                    classifications.append(FileClassification(fp, FileRisk.DEEP_DIVE, f"Matches deep dive pattern: {pattern}"))
                    is_deep = True
                    break
            if is_deep:
                continue

            ext = os.path.splitext(fp)[1]
            if ext in rules.get("deep_dive_extensions", []):
                classifications.append(FileClassification(fp, FileRisk.DEEP_DIVE, f"Deep dive extension: {ext}"))
                continue

            classifications.append(FileClassification(fp, FileRisk.FAST_PASS, "Standard review"))

        return classifications

    def _run_agent(self, agent_name: str, classifications: list[FileClassification],
                   diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Run a specific review agent on the changed files."""
        agent_methods = {
            "lint": self._lint_agent,
            "security": self._security_agent,
            "logic": self._logic_agent,
            "architecture": self._architecture_agent,
            "performance": self._performance_agent,
        }
        agent_fn = agent_methods.get(agent_name, self._generic_agent)
        return agent_fn(classifications, diff_content, pr_description)

    # ---- Specialized Agents ----

    def _lint_agent(self, classifications: list[FileClassification],
                    diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Lint agent: checks code style, formatting, and basic patterns."""
        findings = []
        files_to_check = [c for c in classifications if c.risk != FileRisk.SKIP]

        for fc in files_to_check:
            ext = os.path.splitext(fc.file_path)[1]

            # Python style checks
            if ext == ".py":
                # Check for missing type hints
                if "def " in diff_content and "->" not in diff_content:
                    findings.append(ReviewFinding(
                        id=f"lint-type-hint-{fc.file_path}",
                        file_path=fc.file_path,
                        line_start=0, line_end=0,
                        severity=Severity.LOW,
                        category="style",
                        title="Consider adding type hints",
                        description="Functions without return type annotations reduce code readability and IDE support.",
                        suggestion="Add return type annotations to new/modified functions.",
                        rule_id="PY-TYPEHINT-001",
                    ))

                # Check for print statements
                if re.search(r'\bprint\s*\(', diff_content):
                    findings.append(ReviewFinding(
                        id=f"lint-print-{fc.file_path}",
                        file_path=fc.file_path,
                        line_start=0, line_end=0,
                        severity=Severity.INFO,
                        category="style",
                        title="Print statement detected",
                        description="Production code should use logging instead of print().",
                        suggestion="Replace print() with appropriate logging calls.",
                        rule_id="PY-PRINT-001",
                    ))

                # Check for bare except
                if re.search(r'except\s*:', diff_content):
                    findings.append(ReviewFinding(
                        id=f"lint-bare-except-{fc.file_path}",
                        file_path=fc.file_path,
                        line_start=0, line_end=0,
                        severity=Severity.MEDIUM,
                        category="style",
                        title="Bare except clause",
                        description="Bare except: catches all exceptions including KeyboardInterrupt.",
                        suggestion="Specify the exception type explicitly.",
                        rule_id="PY-BARE-EXCEPT-001",
                    ))

            # JavaScript/TypeScript checks
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                if "console.log" in diff_content:
                    findings.append(ReviewFinding(
                        id=f"lint-console-{fc.file_path}",
                        file_path=fc.file_path,
                        line_start=0, line_end=0,
                        severity=Severity.INFO,
                        category="style",
                        title="console.log in production code",
                        description="Production code should not contain console.log statements.",
                        suggestion="Remove or use a proper logging library.",
                        rule_id="JS-CONSOLE-001",
                    ))

        return findings

    def _security_agent(self, classifications: list[FileClassification],
                        diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Security agent: checks for vulnerabilities and security anti-patterns."""
        findings = []

        # SQL Injection detection
        if re.search(r'(?:execute|cursor\.execute)\s*\(\s*(?:f["\']|["\'].*%.*\b(?:select|insert|update|delete)\b)', diff_content, re.IGNORECASE):
            findings.append(ReviewFinding(
                id="sec-sqli-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.CRITICAL,
                category="security",
                title="Potential SQL Injection vulnerability",
                description="String formatting in SQL queries may allow SQL injection attacks.",
                suggestion="Use parameterized queries with placeholders instead of string formatting.",
                rule_id="SEC-SQLI-001",
            ))

        # Hardcoded secrets
        secret_patterns = [
            (r'(?:api[_-]?key|apikey|secret|password|token)\s*[:=]\s*["\'][^"\']{8,}["\']', "Hardcoded secret/credential"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
            (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API Key"),
            (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
            (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "Private Key"),
        ]
        for pattern, desc in secret_patterns:
            if re.search(pattern, diff_content):
                findings.append(ReviewFinding(
                    id=f"sec-secret-{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
                    file_path="",
                    line_start=0, line_end=0,
                    severity=Severity.CRITICAL,
                    category="security",
                    title=f"Hardcoded secret detected: {desc}",
                    description="Secrets committed to the repository are exposed to all repository accessors.",
                    suggestion="Use environment variables or a secrets manager.",
                    rule_id="SEC-SECRET-001",
                ))

        # XSS vulnerability in web code
        if re.search(r'(?:innerHTML|dangerouslySetInnerHTML|document\.write)\s*[=(]', diff_content):
            findings.append(ReviewFinding(
                id="sec-xss-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.HIGH,
                category="security",
                title="Potential XSS vulnerability",
                description="Direct DOM manipulation with innerHTML/dangerouslySetInnerHTML may enable XSS attacks.",
                suggestion="Use safe DOM APIs like textContent or proper sanitization libraries.",
                rule_id="SEC-XSS-001",
            ))

        # Path traversal
        if re.search(r'os\.path\.join\s*\(\s*.*request\.', diff_content):
            findings.append(ReviewFinding(
                id="sec-path-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.HIGH,
                category="security",
                title="Potential path traversal vulnerability",
                description="User input used in file path construction without sanitization.",
                suggestion="Validate and sanitize user input; use os.path.realpath() to resolve paths.",
                rule_id="SEC-PATH-001",
            ))

        # Command injection
        if re.search(r'(?:os\.system|subprocess\.call|subprocess\.Popen|exec|eval)\s*\(', diff_content):
            findings.append(ReviewFinding(
                id="sec-cmdinj-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.CRITICAL,
                category="security",
                title="Potentially dangerous function usage",
                description="os.system/subprocess/eval with unsanitized input can lead to command injection.",
                suggestion="Use subprocess.run with args as a list, or avoid eval entirely.",
                rule_id="SEC-CMDINJ-001",
            ))

        return findings

    def _logic_agent(self, classifications: list[FileClassification],
                     diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Logic agent: checks for logical errors and edge cases."""
        findings = []

        # Check for potential None/undefined access
        if re.search(r'\.\w+\s*\(', diff_content):
            findings.append(ReviewFinding(
                id="logic-null-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.MEDIUM,
                category="logic",
                title="Potential null reference risk",
                description="Method calls on potentially null/undefined objects.",
                suggestion="Add null checks before method invocations.",
                rule_id="LOGIC-NULL-001",
            ))

        # Check for division by zero risk
        if re.search(r'/\s*\w+', diff_content) and not re.search(r'(?:if|assert).*\b(?:!=|==)\s*0', diff_content):
            findings.append(ReviewFinding(
                id="logic-divzero-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.MEDIUM,
                category="logic",
                title="Potential division by zero",
                description="Division operation without zero check.",
                suggestion="Add a guard clause to check for zero before division.",
                rule_id="LOGIC-DIVZERO-001",
            ))

        # Check for unchecked async operations
        async_patterns = [
            (r'(?:async\s+def|await\s)', "Async function without timeout"),
            (r'asyncio\.create_task\s*\(', "Fire-and-forget task without error handling"),
        ]
        for pattern, desc in async_patterns:
            if re.search(pattern, diff_content):
                findings.append(ReviewFinding(
                    id=f"logic-async-{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
                    file_path="",
                    line_start=0, line_end=0,
                    severity=Severity.LOW,
                    category="logic",
                    title=desc,
                    description="Async operations should have proper error handling and timeout mechanisms.",
                    suggestion="Add try/except blocks and asyncio.wait_for with appropriate timeouts.",
                    rule_id="LOGIC-ASYNC-001",
                ))

        return findings

    def _architecture_agent(self, classifications: list[FileClassification],
                            diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Architecture agent: checks structural and design patterns."""
        findings = []

        deep_dive_files = [c.file_path for c in classifications if c.risk == FileRisk.DEEP_DIVE]

        # Check for circular dependencies using graph
        if self.graph and self._project_root:
            for fp in deep_dive_files:
                impact = self.graph.impact_analysis(os.path.basename(fp))
                if isinstance(impact, dict) and impact.get("risk_level") == "HIGH":
                    findings.append(ReviewFinding(
                        id=f"arch-impact-{fp}",
                        file_path=fp,
                        line_start=0, line_end=0,
                        severity=Severity.HIGH,
                        category="architecture",
                        title=f"High-impact change: {impact.get('direct_dependents', 0)} dependents",
                        description=f"This file has {impact.get('transitive_dependents', 0)} transitive dependents. Changes may cause cascading failures.",
                        suggestion="Consider adding integration tests and performing gradual rollout.",
                        rule_id="ARCH-IMPACT-001",
                    ))

        # Check for large files (potential god objects)
        for fc in classifications:
            if fc.risk != FileRisk.SKIP and os.path.exists(os.path.join(self._project_root, fc.file_path)):
                full_path = os.path.join(self._project_root, fc.file_path)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        line_count = len(f.readlines())
                    if line_count > 500:
                        findings.append(ReviewFinding(
                            id=f"arch-size-{fc.file_path}",
                            file_path=fc.file_path,
                            line_start=0, line_end=0,
                            severity=Severity.MEDIUM,
                            category="architecture",
                            title=f"Large file ({line_count} lines)",
                            description=f"File exceeds 500 lines. Consider splitting into smaller modules.",
                            suggestion="Refactor into multiple focused modules with single responsibilities.",
                            rule_id="ARCH-SIZE-001",
                        ))
                except Exception:
                    pass

        # Check for missing tests
        test_files = [c for c in classifications if "test" in c.file_path.lower() or c.file_path.endswith("_test.py")]
        source_files = [c for c in classifications if c not in test_files and c.risk != FileRisk.SKIP]
        if source_files and not test_files:
            findings.append(ReviewFinding(
                id="arch-test-missing",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.MEDIUM,
                category="architecture",
                title="No test files in this PR",
                description="New/modified source files without corresponding test changes.",
                suggestion="Consider adding unit tests for the changed functionality.",
                rule_id="ARCH-TEST-001",
            ))

        return findings

    def _performance_agent(self, classifications: list[FileClassification],
                           diff_content: str, pr_description: str) -> list[ReviewFinding]:
        """Performance agent: checks for performance anti-patterns."""
        findings = []

        # N+1 query pattern
        if re.search(r'for\s+\w+\s+in\s+.+:\s*\n\s*.+\.(?:execute|query|find|get)\s*\(', diff_content):
            findings.append(ReviewFinding(
                id="perf-nplus1-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.HIGH,
                category="performance",
                title="Potential N+1 query pattern",
                description="Database query inside a loop can cause severe performance degradation.",
                suggestion="Use batch queries, eager loading, or caching to reduce database round trips.",
                rule_id="PERF-NPLUS1-001",
            ))

        # Large list in memory
        if re.search(r'\.read\s*\(\s*\)', diff_content) and not re.search(r'chunk', diff_content, re.IGNORECASE):
            findings.append(ReviewFinding(
                id="perf-memory-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.MEDIUM,
                category="performance",
                title="Potential memory issue: reading entire file",
                description="Reading large files entirely into memory can cause OOM errors.",
                suggestion="Use streaming/chunked reading for large files.",
                rule_id="PERF-MEM-001",
            ))

        # Blocking I/O in async context
        if re.search(r'await\s', diff_content) and re.search(r'(?:time\.sleep|open\s*\(|json\.load\s*\()', diff_content):
            findings.append(ReviewFinding(
                id="perf-async-001",
                file_path="",
                line_start=0, line_end=0,
                severity=Severity.MEDIUM,
                category="performance",
                title="Blocking operation in async context",
                description="Synchronous I/O operations in async functions block the event loop.",
                suggestion="Use async equivalents (aiofiles, asyncio.to_thread) for I/O in async functions.",
                rule_id="PERF-ASYNC-001",
            ))

        return findings

    def _generic_agent(self, classifications: list[FileClassification],
                       diff_content: str, pr_description: str) -> list[ReviewFinding]:
        return []

    def _calculate_score(self, findings: list[ReviewFinding]) -> float:
        """Calculate overall review score based on findings severity and agent weights."""
        if not findings:
            return 10.0

        weights = self.config.get("agents", {})
        severity_penalty = {
            Severity.CRITICAL: 3.0,
            Severity.HIGH: 1.5,
            Severity.MEDIUM: 0.5,
            Severity.LOW: 0.2,
            Severity.INFO: 0.05,
        }

        total_penalty = 0.0
        agent_counts: dict[str, int] = defaultdict(int)
        for f in findings:
            penalty = severity_penalty.get(f.severity, 0.1)
            agent_counts[f.agent_name] += 1
            agent_weight = weights.get(f.agent_name, {}).get("weight", 0.2)
            total_penalty += penalty * agent_weight

        score = max(0.0, 10.0 - total_penalty)
        return score

    def _generate_summary(self, findings: list[ReviewFinding], score: float,
                          status: ReviewStatus, classifications: list[FileClassification]) -> str:
        """Generate a human-readable review summary."""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        for f in findings:
            severity_counts[f.severity.value] += 1
            category_counts[f.category] += 1

        lines = [
            f"# CodeBridge Review Report",
            f"",
            f"**Score**: {score}/10 | **Status**: {status.value}",
            f"**Files Reviewed**: {len(classifications)}",
            f"**Findings**: {len(findings)}",
            f"",
            f"## Severity Breakdown",
        ]
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            count = severity_counts.get(sev.value, 0)
            if count > 0:
                lines.append(f"- 🔴 {sev.value}: {count}" if sev == Severity.CRITICAL else
                            f"- 🟠 {sev.value}: {count}" if sev == Severity.HIGH else
                            f"- 🟡 {sev.value}: {count}" if sev == Severity.MEDIUM else
                            f"- 🟢 {sev.value}: {count}")

        if category_counts:
            lines.append(f"\n## Category Breakdown")
            for cat, count in sorted(category_counts.items()):
                lines.append(f"- {cat}: {count}")

        if findings:
            lines.append(f"\n## Top Findings")
            for f in sorted(findings, key=lambda x: severity_order(x.severity))[:5]:
                sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}.get(f.severity.value, "")
                lines.append(f"- {sev_icon} [{f.severity.value}] **{f.title}** ({f.category})")
                if f.file_path:
                    lines.append(f"  📁 `{f.file_path}`")
                lines.append(f"  {f.suggestion}")

        return "\n".join(lines)


def severity_order(sev: Severity) -> int:
    order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
    return order.get(sev, 5)


import hashlib
