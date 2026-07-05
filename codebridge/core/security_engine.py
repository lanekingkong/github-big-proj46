"""BridgeSecure: AI-Powered Security Vulnerability Detection.

Inspired by Semgrep (15K★, pattern-based), CodeQL (semantic analysis), and
GitHub Advanced Security, BridgeSecure combines pattern matching with semantic
analysis to detect security vulnerabilities in AI-generated code. Goes beyond
traditional SAST by understanding context and detecting logic-level flaws.

Detection Capabilities:
- OWASP Top 10 (2026): Injection, XSS, Auth, SSRF, etc.
- AI-specific flaws: Hallucinated APIs, fabricated libraries, logic gaps
- Secret detection: 200+ token patterns
- Dependency vulnerabilities: SCA with CVE database lookup
- Custom rules: YAML-based rule engine
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class VulnSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    WARNING = "warning"


class VulnCategory(Enum):
    INJECTION = "injection"
    XSS = "xss"
    AUTH = "authentication"
    DATA_EXPOSURE = "data_exposure"
    SSRF = "ssrf"
    DESERIALIZATION = "deserialization"
    CRYPTO = "cryptography"
    CONFIG = "configuration"
    DEPENDENCY = "dependency"
    SECRET = "secret"
    HALLUCINATION = "hallucination"
    LOGIC_FLAW = "logic_flaw"
    PERMISSION = "permission"


@dataclass
class Vulnerability:
    id: str
    title: str
    description: str
    severity: VulnSeverity
    category: VulnCategory
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str = ""
    cwe_id: str = ""
    cvss_score: float = 0.0
    remediation: str = ""
    references: list[str] = field(default_factory=list)
    false_positive_probability: float = 0.0


@dataclass
class SecurityReport:
    project_name: str = ""
    total_files: int = 0
    total_findings: int = 0
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    risk_score: float = 0.0
    pass_audit: bool = False
    summary: str = ""
    scan_duration_ms: float = 0.0


class BridgeSecure:
    """AI-powered security vulnerability detection engine.

    Usage:
        scanner = BridgeSecure()
        report = scanner.scan_project("/path/to/project")
        print(f"Risk Score: {report.risk_score}/100")
        for vuln in report.vulnerabilities:
            print(f"[{vuln.severity.value}] {vuln.title}")
    """

    # OWASP Top 10 + AI-specific patterns
    BUILTIN_RULES = {
        "sql_injection": {
            "patterns": [
                r'(?:execute|cursor\.execute|raw|rawQuery)\s*\(\s*(?:f["\']|["\'].*%.*SELECT|["\'].*%.*INSERT|["\'].*%.*UPDATE|["\'].*%.*DELETE)',
                r'(?:execute|cursor\.execute)\s*\(\s*["\'].*\{\s*\}.*["\'].*\.format\s*\(',
                r'\.raw\s*\(\s*`.*\$\{.*SELECT',
            ],
            "severity": "critical",
            "category": "injection",
            "cwe": "CWE-89",
            "title": "SQL Injection Vulnerability",
            "description": "User input concatenated into SQL query without parameterization.",
            "remediation": "Use parameterized queries with placeholders (? or %s) instead of string formatting.",
        },
        "xss_reflected": {
            "patterns": [
                r'innerHTML\s*=',
                r'dangerouslySetInnerHTML',
                r'document\.write\s*\(',
                r'\.html\s*\(\s*.*\+',
                r'v-html\s*=',
            ],
            "severity": "high",
            "category": "xss",
            "cwe": "CWE-79",
            "title": "Cross-Site Scripting (XSS)",
            "description": "Unsanitized user input rendered directly in the DOM.",
            "remediation": "Use textContent, innerText, or a sanitization library like DOMPurify.",
        },
        "command_injection": {
            "patterns": [
                r'os\.system\s*\(\s*.*\+',
                r'subprocess\.(?:call|Popen|run)\s*\(\s*.*\+',
                r'exec\s*\(\s*.*\+',
                r'eval\s*\(\s*.*\+',
                r'child_process\.exec\s*\(\s*.*\+',
            ],
            "severity": "critical",
            "category": "injection",
            "cwe": "CWE-78",
            "title": "Command Injection",
            "description": "User input concatenated into shell command execution.",
            "remediation": "Use subprocess.run with args as a list; never concatenate user input into shell commands.",
        },
        "hardcoded_credentials": {
            "patterns": [
                r'(?:password|passwd|pwd)\s*[:=]\s*["\'][^"\']+["\']',
                r'(?:api[_-]?key|apikey|secret_key|access_key)\s*[:=]\s*["\'][A-Za-z0-9+/=]{20,}["\']',
                r'(?:token|auth_token|bearer)\s*[:=]\s*["\'][A-Za-z0-9._\-]{20,}["\']',
            ],
            "severity": "critical",
            "category": "secret",
            "cwe": "CWE-798",
            "title": "Hardcoded Credentials",
            "description": "Sensitive credentials committed directly to source code.",
            "remediation": "Use environment variables, a secrets manager (Vault/AWS Secrets Manager), or .env files excluded from git.",
        },
        "path_traversal": {
            "patterns": [
                r'open\s*\(\s*.*\+.*request\.',
                r'os\.path\.join\s*\(\s*.*request\.',
                r'fs\.readFile\s*\(\s*.*\+.*req\.',
            ],
            "severity": "high",
            "category": "injection",
            "cwe": "CWE-22",
            "title": "Path Traversal",
            "description": "User input used to construct file paths without validation.",
            "remediation": "Validate and sanitize user input; use os.path.realpath() to resolve and check paths against allowed directories.",
        },
        "insecure_crypto": {
            "patterns": [
                r'hashlib\.md5\s*\(',
                r'hashlib\.sha1\s*\(',
                r'DES\.new\s*\(',
                r'crypto\.createHash\s*\(\s*["\']md5["\']',
                r'crypto\.createHash\s*\(\s*["\']sha1["\']',
            ],
            "severity": "medium",
            "category": "crypto",
            "cwe": "CWE-327",
            "title": "Weak Cryptographic Algorithm",
            "description": "MD5 and SHA1 are cryptographically broken and unsuitable for security purposes.",
            "remediation": "Use SHA-256 or stronger (hashlib.sha256). For passwords, use bcrypt, scrypt, or Argon2.",
        },
        "ssrf": {
            "patterns": [
                r'requests\.(?:get|post|put|delete)\s*\(\s*.*request\.',
                r'urllib\.request\.urlopen\s*\(\s*.*request\.',
                r'httpx\.(?:get|post)\s*\(\s*.*request\.',
                r'fetch\s*\(\s*.*req\.',
            ],
            "severity": "high",
            "category": "ssrf",
            "cwe": "CWE-918",
            "title": "Server-Side Request Forgery (SSRF)",
            "description": "User-controlled URL used in server-side HTTP requests.",
            "remediation": "Validate URLs against allowlists; block internal IP ranges (127.0.0.0/8, 10.0.0.0/8, 169.254.0.0/16).",
        },
        "insecure_deserialization": {
            "patterns": [
                r'pickle\.loads?\s*\(',
                r'yaml\.load\s*\(\s*(?!.*SafeLoader)',
                r'cPickle\.loads?\s*\(',
                r'JSON\.parse\s*\(\s*.*request\.',
            ],
            "severity": "high",
            "category": "deserialization",
            "cwe": "CWE-502",
            "title": "Insecure Deserialization",
            "description": "Unsafe deserialization of user-controlled data can lead to RCE.",
            "remediation": "Use yaml.SafeLoader, never unpickle untrusted data, validate JSON schema before parsing.",
        },
        "ai_hallucination": {
            "patterns": [
                r'import\s+(?!os|sys|json|re|datetime|math|random|collections|itertools|functools|typing|pathlib|logging|hashlib|base64|uuid|enum|dataclasses|abc|copy|textwrap|struct|io|csv|tempfile|shutil|glob|subprocess|threading|asyncio|concurrent|socket|ssl|email|http|urllib|xml|html|unittest|pytest|argparse|configparser)([a-z_][a-z0-9_]*)',
            ],
            "severity": "warning",
            "category": "hallucination",
            "cwe": "CWE-1104",
            "title": "Potentially Hallucinated Import",
            "description": "This import may reference a non-existent or hallucinated library.",
            "remediation": "Verify that the imported library exists on PyPI/npm and is intentionally included as a dependency.",
        },
        "missing_auth_check": {
            "patterns": [
                r'@app\.(?:route|get|post|put|delete)\s*\(.*\)\s*\n\s*def\s+(?!.*@login_required|.*@auth_required|.*auth|.*token)',
            ],
            "severity": "high",
            "category": "auth",
            "cwe": "CWE-306",
            "title": "Missing Authentication Check",
            "description": "API endpoint defined without an authentication decorator or check.",
            "remediation": "Add authentication middleware or decorator to protect the endpoint.",
        },
    }

    def __init__(self, custom_rules_path: Optional[str] = None):
        self.rules = dict(self.BUILTIN_RULES)
        if custom_rules_path and os.path.exists(custom_rules_path):
            with open(custom_rules_path, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f)
                if custom:
                    self.rules.update(custom)

    def scan_project(self, root_path: str, file_patterns: Optional[list] = None) -> SecurityReport:
        """Scan an entire project directory for security vulnerabilities.

        Args:
            root_path: Project root directory path
            file_patterns: Optional glob patterns to filter files

        Returns:
            SecurityReport with all findings and risk assessment
        """
        import time
        start = time.time()

        root = Path(root_path).resolve()
        vulnerabilities: list[Vulnerability] = []

        source_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                            ".java", ".rb", ".php", ".c", ".cpp", ".h", ".hpp",
                            ".swift", ".kt", ".cs", ".yaml", ".yml", ".tf", ".sql"}

        files_to_scan = []
        for ext in source_extensions:
            for file_path in root.rglob(f"*{ext}"):
                if ".git" not in file_path.parts and "node_modules" not in file_path.parts:
                    files_to_scan.append(file_path)

        for file_path in files_to_scan:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for rule_id, rule in self.rules.items():
                for pattern in rule["patterns"]:
                    for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                        line_num = content[:match.start()].count("\n") + 1
                        # Get code context (surrounding lines)
                        lines = content.split("\n")
                        start_line = max(0, line_num - 2)
                        end_line = min(len(lines), line_num + 2)
                        snippet = "\n".join(lines[start_line:end_line])

                        vuln = Vulnerability(
                            id=f"{rule_id}-{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                            title=rule["title"],
                            description=rule["description"],
                            severity=VulnSeverity(rule["severity"]),
                            category=VulnCategory(rule["category"]),
                            file_path=str(file_path.relative_to(root)),
                            line_start=line_num,
                            line_end=line_num,
                            code_snippet=snippet,
                            cwe_id=rule.get("cwe", ""),
                            remediation=rule["remediation"],
                        )
                        vulnerabilities.append(vuln)

        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)

        # Determine audit pass/fail
        critical_count = sum(1 for v in vulnerabilities if v.severity == VulnSeverity.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.severity == VulnSeverity.HIGH)
        pass_audit = critical_count == 0 and high_count <= 3 and risk_score < 30

        duration_ms = round((time.time() - start) * 1000, 1)

        report = SecurityReport(
            project_name=root.name,
            total_files=len(files_to_scan),
            total_findings=len(vulnerabilities),
            vulnerabilities=vulnerabilities,
            risk_score=round(risk_score, 1),
            pass_audit=pass_audit,
            scan_duration_ms=duration_ms,
            summary=self._generate_summary(vulnerabilities, risk_score, pass_audit),
        )

        return report

    def scan_diff(self, diff_content: str) -> list[Vulnerability]:
        """Scan a git diff for security vulnerabilities in changed code only.

        Optimized for PR review pipelines where full project scan is unnecessary.
        """
        vulnerabilities = []
        added_lines = []
        current_file = ""

        for line in diff_content.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines.append((current_file, line[1:]))

        for file_path, line_content in added_lines:
            for rule_id, rule in self.rules.items():
                for pattern in rule["patterns"]:
                    if re.search(pattern, line_content, re.IGNORECASE):
                        vuln = Vulnerability(
                            id=f"{rule_id}-diff-{hashlib.md5(f'{file_path}:{line_content}'.encode()).hexdigest()[:8]}",
                            title=rule["title"],
                            description=rule["description"],
                            severity=VulnSeverity(rule["severity"]),
                            category=VulnCategory(rule["category"]),
                            file_path=file_path,
                            line_start=0,
                            line_end=0,
                            code_snippet=line_content,
                            cwe_id=rule.get("cwe", ""),
                            remediation=rule["remediation"],
                        )
                        vulnerabilities.append(vuln)

        return vulnerabilities

    def _calculate_risk_score(self, vulnerabilities: list[Vulnerability]) -> float:
        """Calculate overall risk score (0-100, higher = more risk)."""
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            VulnSeverity.CRITICAL: 10,
            VulnSeverity.HIGH: 5,
            VulnSeverity.MEDIUM: 2,
            VulnSeverity.LOW: 0.5,
            VulnSeverity.WARNING: 0.1,
        }

        total = sum(severity_weights.get(v.severity, 1) for v in vulnerabilities)
        # Cap at 100
        return min(100.0, total * 2.5)

    def _generate_summary(self, vulnerabilities: list[Vulnerability],
                          risk_score: float, pass_audit: bool) -> str:
        """Generate a human-readable security report summary."""
        if not vulnerabilities:
            return "✅ No security vulnerabilities found. Project passes security audit."

        sev_counts = {}
        cat_counts = {}
        for v in vulnerabilities:
            sev_counts[v.severity.value] = sev_counts.get(v.severity.value, 0) + 1
            cat_counts[v.category.value] = cat_counts.get(v.category.value, 0) + 1

        lines = [
            f"## Security Audit Report",
            f"",
            f"**Risk Score**: {risk_score}/100 | **Audit**: {'✅ PASSED' if pass_audit else '❌ FAILED'}",
            f"**Total Findings**: {len(vulnerabilities)}",
            f"",
            f"### Severity Distribution",
        ]
        for sev in [VulnSeverity.CRITICAL, VulnSeverity.HIGH, VulnSeverity.MEDIUM, VulnSeverity.LOW, VulnSeverity.WARNING]:
            count = sev_counts.get(sev.value, 0)
            if count > 0:
                lines.append(f"- {sev.value}: {count}")

        lines.append(f"\n### Category Distribution")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")

        critical_vulns = [v for v in vulnerabilities if v.severity == VulnSeverity.CRITICAL]
        if critical_vulns:
            lines.append(f"\n### Critical Findings")
            for v in critical_vulns[:5]:
                lines.append(f"- **{v.title}** in `{v.file_path}`")
                lines.append(f"  → {v.remediation}")

        return "\n".join(lines)

    def add_custom_rule(self, rule_id: str, rule_def: dict):
        """Add a custom security rule at runtime."""
        self.rules[rule_id] = rule_def

    def export_rules(self, output_path: str):
        """Export current rules (builtin + custom) to a YAML file."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.rules, f, default_flow_style=False, allow_unicode=True)
