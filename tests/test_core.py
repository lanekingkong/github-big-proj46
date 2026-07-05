"""Unit tests for CodeBridge Core Engines.

Run with: pytest tests/ -v
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_graph_engine_basic():
    """Test BridgeGraph basic indexing."""
    from codebridge.core.graph_engine import BridgeGraph

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Python file
        py_file = Path(tmpdir) / "sample.py"
        py_file.write_text("""
def calculate_sum(a, b):
    return a + b

def process_data(items):
    result = calculate_sum(items[0], items[1])
    return result

class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, input_data):
        return calculate_sum(input_data[0], input_data[1])
""")

        graph = BridgeGraph(db_path=os.path.join(tmpdir, "test.db"))
        stats = graph.index_project(tmpdir)

        assert stats.total_nodes > 0
        assert stats.total_files == 1
        assert stats.total_functions >= 2
        assert stats.total_classes >= 1

        # Test search
        results = graph.search("calculate_sum")
        assert len(results) >= 1
        assert any(r.name == "calculate_sum" for r in results)

        # Test callers
        callers = graph.find_callers("calculate_sum")
        assert len(callers) >= 1

        # Test dead code detection
        dead = graph.detect_dead_code()
        assert isinstance(dead, list)


def test_graph_engine_impact():
    """Test BridgeGraph impact analysis."""
    from codebridge.core.graph_engine import BridgeGraph

    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "module.py"
        py_file.write_text("""
def helper():
    return 42

def public_api(x):
    return helper() + x

def another_caller():
    return public_api(1)
""")

        graph = BridgeGraph()
        graph.index_project(tmpdir)

        impact = graph.impact_analysis("helper")
        assert "error" not in impact
        assert impact["direct_dependents"] >= 1


def test_security_engine_scan():
    """Test BridgeSecure project scan."""
    from codebridge.core.security_engine import BridgeSecure

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with vulnerabilities
        vuln_file = Path(tmpdir) / "vuln.py"
        vuln_file.write_text("""
import hashlib

password = "admin123"

def login(user_input):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    cursor.execute(query)
    return hashlib.md5(user_input.encode()).hexdigest()
""")

        scanner = BridgeSecure()
        report = scanner.scan_project(tmpdir)

        assert report.total_files >= 1
        assert report.total_findings >= 2  # SQL injection + hardcoded password
        assert report.risk_score > 0

        # Check specific findings
        categories = [v.category.value for v in report.vulnerabilities]
        assert "injection" in categories


def test_security_scan_clean():
    """Test BridgeSecure on clean code."""
    from codebridge.core.security_engine import BridgeSecure

    with tempfile.TemporaryDirectory() as tmpdir:
        clean_file = Path(tmpdir) / "clean.py"
        clean_file.write_text("""
import hashlib
import os

def get_config():
    password = os.environ.get("DB_PASSWORD")
    return {"password": password}
""")

        scanner = BridgeSecure()
        report = scanner.scan_project(tmpdir)

        # Should find no critical findings
        critical = [v for v in report.vulnerabilities if v.severity.value == "critical"]
        assert len(critical) == 0


def test_context_compression():
    """Test BridgeFlow context compression."""
    from codebridge.core.context_engine import BridgeFlow, CompressionLevel

    flow = BridgeFlow()
    text = "def hello():\n    # This is a comment\n    print('hello')\n    return True\n" * 20

    compressed, stats = flow.compress(text, level=CompressionLevel.STANDARD)

    assert stats.original_tokens > stats.compressed_tokens
    assert stats.compression_ratio > 0
    assert len(compressed) > 0


def test_gate_evaluation():
    """Test BridgeGate quality gate evaluation."""
    from codebridge.core.gate_engine import BridgeGate, Environment, GateDecision

    gate = BridgeGate()

    # Should pass with high scores
    result = gate.evaluate(
        env=Environment.PRODUCTION,
        security_score=9.0,
        review_score=9.0,
        test_coverage=90.0,
    )
    assert result.decision == GateDecision.PASS
    assert result.overall_score >= 8.0

    # Should block with low security score
    result = gate.evaluate(
        env=Environment.PRODUCTION,
        security_score=3.0,
        review_score=9.0,
        test_coverage=90.0,
    )
    assert result.decision == GateDecision.BLOCKED

    # Dev should be more lenient
    result = gate.evaluate(
        env=Environment.DEV,
        security_score=5.0,
        review_score=6.0,
        test_coverage=60.0,
    )
    assert result.decision != GateDecision.BLOCKED


def test_metrics_snapshot():
    """Test BridgeMetrics pipeline snapshot."""
    from codebridge.core.metrics_engine import BridgeMetrics

    metrics = BridgeMetrics()

    metrics.record_pr_merge(
        merge_time_hours=2.5,
        ai_code_ratio=0.6,
        lines_changed=200,
        ai_lines=120,
    )
    metrics.record_pr_merge(
        merge_time_hours=4.0,
        ai_code_ratio=0.8,
        lines_changed=500,
        ai_lines=400,
    )
    metrics.record_defect_escape(severity="high")
    metrics.record_vulnerability_scan(vulnerabilities_found=3, total_loc=5000)

    snapshot = metrics.get_pipeline_snapshot()

    assert snapshot.prs_merged == 2
    assert snapshot.mean_merge_time > 0
    assert snapshot.ai_code_ratio > 0.5
    assert snapshot.vulnerability_density > 0
    assert snapshot.escaped_defect_rate > 0


def test_review_engine_classification():
    """Test BridgeReview file classification."""
    from codebridge.core.review_engine import BridgeReview, FileRisk

    reviewer = BridgeReview()

    # Test skip classification
    classifications = reviewer._classify_files(["README.md", "config.json"])
    assert all(c.risk == FileRisk.SKIP for c in classifications)

    # Test deep dive
    classifications = reviewer._classify_files(["auth/login.py", "payment/checkout.py"])
    assert any(c.risk == FileRisk.DEEP_DIVE for c in classifications)

    # Test fast pass
    classifications = reviewer._classify_files(["utils/helpers.py"])
    assert any(c.risk == FileRisk.FAST_PASS for c in classifications)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
