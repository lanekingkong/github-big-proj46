"""CodeBridge CLI: Command-Line Interface for AI Code Governance.

Usage:
    codebridge scan /path/to/project          # Security + quality scan
    codebridge review --base main --head feat  # AI code review
    codebridge gate --env production           # Quality gate check
    codebridge metrics                         # Pipeline metrics dashboard
    codebridge graph --project /path           # Knowledge graph analysis
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

app = typer.Typer(
    name="codebridge",
    help="AI Code Governance Platform — From Generation to Production, Safely.",
    add_completion=False,
)

console = Console()


def _get_version():
    try:
        from importlib.metadata import version
        return version("codebridge")
    except Exception:
        return "0.1.0"


@app.command()
def version():
    """Show CodeBridge version."""
    rprint(f"[bold cyan]CodeBridge[/bold cyan] v{_get_version()}")
    rprint("AI Code Governance Platform")


@app.command()
def scan(
    project_path: str = typer.Argument(..., help="Path to project directory"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    rules: Optional[str] = typer.Option(None, "--rules", "-r", help="Custom security rules YAML"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, sarif"),
):
    """Run security vulnerability scan on a project."""
    from codebridge.core.security_engine import BridgeSecure

    if not os.path.exists(project_path):
        console.print(f"[red]Error:[/red] Path not found: {project_path}")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Scanning for vulnerabilities...", total=None)
        scanner = BridgeSecure(custom_rules_path=rules)
        report = scanner.scan_project(project_path)
        progress.remove_task(task)

    if format == "json":
        result = {
            "project": report.project_name,
            "risk_score": report.risk_score,
            "pass_audit": report.pass_audit,
            "total_files": report.total_files,
            "total_findings": report.total_findings,
            "vulnerabilities": [
                {
                    "id": v.id,
                    "title": v.title,
                    "severity": v.severity.value,
                    "category": v.category.value,
                    "file": v.file_path,
                    "line": v.line_start,
                    "cwe": v.cwe_id,
                    "remediation": v.remediation,
                }
                for v in report.vulnerabilities
            ],
        }
        rprint(json.dumps(result, indent=2, ensure_ascii=False))
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        status_color = "green" if report.pass_audit else "red"
        status_text = "PASSED" if report.pass_audit else "FAILED"

        rprint(Panel.fit(
            f"[bold]Security Audit: [{status_color}]{status_text}[/{status_color}][/bold]\n"
            f"Risk Score: {report.risk_score}/100  |  Files Scanned: {report.total_files}  |  Findings: {report.total_findings}\n"
            f"Duration: {report.scan_duration_ms}ms",
            title="BridgeSecure Scan"
        ))

        if report.vulnerabilities:
            table = Table(title="Vulnerabilities Found", show_header=True, header_style="bold")
            table.add_column("Severity", style="bold", width=10)
            table.add_column("Category", width=14)
            table.add_column("Title", width=40)
            table.add_column("File", width=30)

            severity_styles = {
                "critical": "bold red",
                "high": "red",
                "medium": "yellow",
                "low": "dim",
                "warning": "dim",
            }

            for v in report.vulnerabilities[:20]:
                table.add_row(
                    f"[{severity_styles.get(v.severity.value, '')}]{v.severity.value}[/]",
                    v.category.value,
                    v.title,
                    f"{v.file_path}:{v.line_start}",
                )
            console.print(table)

            if len(report.vulnerabilities) > 20:
                console.print(f"[dim]... and {len(report.vulnerabilities) - 20} more findings[/dim]")

        if output:
            scanner.export_rules(output)


@app.command()
def review(
    base: str = typer.Option("main", "--base", "-b", help="Base branch"),
    head: str = typer.Option("HEAD", "--head", "-h", help="Head branch/commit"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project root path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Run AI-powered code review on changes."""
    from codebridge.core.review_engine import BridgeReview
    from codebridge.core.graph_engine import BridgeGraph

    rprint(f"[bold cyan]CodeBridge Review[/bold cyan]")
    rprint(f"Comparing [bold]{base}[/bold] → [bold]{head}[/bold]")

    graph = BridgeGraph()
    if project and os.path.exists(project):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Building knowledge graph...", total=None)
            graph.index_project(project)
            progress.remove_task(task)
        stats = graph.get_stats()
        rprint(f"Graph: {stats.total_nodes} nodes, {stats.total_edges} edges, {stats.total_files} files")

    reviewer = BridgeReview(graph=graph)
    if project:
        reviewer.index_project(project)

    result = reviewer.review_pr(
        base_branch=base,
        head_branch=head,
        pr_description="CLI review request",
    )

    status_color = {"approved": "green", "changes_requested": "red", "comment": "yellow"}.get(result.status.value, "white")
    rprint(Panel.fit(result.summary, title=f"Review Result: [{status_color}]{result.status.value.upper()}[/{status_color}]"))

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump({
                "status": result.status.value,
                "score": result.score,
                "findings": [
                    {"title": f.title, "severity": f.severity.value, "category": f.category, "file": f.file_path}
                    for f in result.findings
                ],
                "summary": result.summary,
            }, f, indent=2, ensure_ascii=False)


@app.command()
def gate(
    env: str = typer.Option("staging", "--env", "-e", help="Target environment: dev, staging, canary, production"),
    security_score: float = typer.Option(10.0, "--security-score", "-s", help="Security score (0-10)"),
    review_score: float = typer.Option(10.0, "--review-score", "-r", help="Review score (0-10)"),
    test_coverage: float = typer.Option(80.0, "--coverage", "-c", help="Test coverage percentage"),
    complexity_delta: int = typer.Option(0, "--complexity", "-x", help="Complexity delta"),
    breaking_changes: int = typer.Option(0, "--breaking", "-b", help="Number of breaking changes"),
):
    """Evaluate code against environment-specific quality gates."""
    from codebridge.core.gate_engine import BridgeGate, Environment, GateDecision

    try:
        environment = Environment(env)
    except ValueError:
        console.print(f"[red]Invalid environment:[/red] {env}. Use: dev, staging, canary, production")
        raise typer.Exit(1)

    gate_obj = BridgeGate()
    result = gate_obj.evaluate(
        env=environment,
        security_score=security_score,
        review_score=review_score,
        test_coverage=test_coverage,
        complexity_delta=complexity_delta,
        breaking_changes=breaking_changes,
    )

    decision_colors = {
        GateDecision.PASS: "green",
        GateDecision.PASS_WITH_WARNINGS: "yellow",
        GateDecision.NEEDS_REVIEW: "orange1",
        GateDecision.BLOCKED: "red",
    }
    color = decision_colors.get(result.decision, "white")

    rprint(Panel.fit(
        f"[bold {color}]Decision: {result.decision.value.upper()}[/bold {color}]\n"
        f"Overall Score: {result.overall_score}/10  |  Environment: {result.environment.value}",
        title="Quality Gate Result"
    ))

    table = Table(title="Checks", show_header=True)
    table.add_column("Check", style="bold")
    table.add_column("Score")
    table.add_column("Threshold")
    table.add_column("Status")

    for check in result.checks:
        status_icon = "✅" if check.passed else "❌"
        table.add_row(check.name, str(check.score), str(check.threshold), status_icon)
    console.print(table)

    if result.recommendations:
        rprint("\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            rprint(f"  • {rec}")

    badge = gate_obj.generate_badge(result)
    rprint(f"\n[dim]Badge URL: {badge}[/dim]")


@app.command()
def metrics(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file path"),
):
    """Display pipeline metrics dashboard."""
    from codebridge.core.metrics_engine import BridgeMetrics

    m = BridgeMetrics()
    dashboard = m.export_dashboard(output)

    metrics = dashboard["pipeline_metrics"]

    rprint(Panel.fit(
        f"[bold]Trend: {metrics['trend_direction'].upper()}[/bold]",
        title="Pipeline Health"
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    rows = [
        ("Mean Merge Time", f"{metrics['mean_merge_time_hours']}h"),
        ("P95 Merge Time", f"{metrics['p95_merge_time_hours']}h"),
        ("Escaped Defect Rate", str(metrics['escaped_defect_rate'])),
        ("AI Code Ratio", f"{metrics['ai_code_ratio'] * 100:.0f}%"),
        ("Vuln Density (/1K LOC)", str(metrics['vulnerability_density_per_1k_loc'])),
        ("Review Burden", f"{metrics['review_burden_hours_per_week']}h/wk"),
        ("PRs Merged", str(metrics['prs_merged'])),
        ("Review Backlog", str(metrics['review_backlog'])),
        ("Cost per Ticket", f"${metrics['cost_per_solved_ticket_usd']:.2f}"),
        ("Est. Token Cost", f"${metrics['estimated_token_cost_usd']:.2f}"),
    ]

    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)

    if dashboard.get("trends"):
        trends_table = Table(title="Trends", show_header=True)
        trends_table.add_column("Metric", style="bold")
        trends_table.add_column("Current")
        trends_table.add_column("Change")
        trends_table.add_column("Trend")

        for t in dashboard["trends"]:
            emoji = "🟢" if t["trend"] == "improving" else "🔴" if t["trend"] == "declining" else "🟡"
            trends_table.add_row(
                t["metric"],
                str(t["current"]),
                f"{t['change_pct']:+.1f}%",
                f"{emoji} {t['trend']}",
            )
        console.print(trends_table)


@app.command()
def graph(
    project_path: str = typer.Argument(..., help="Path to project directory"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search for entities by name"),
    impact: Optional[str] = typer.Option(None, "--impact", "-i", help="Analyze impact of changing an entity"),
    dead_code: bool = typer.Option(False, "--dead", "-d", help="Detect potentially dead code"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file path"),
):
    """Build and query knowledge graph of a codebase."""
    from codebridge.core.graph_engine import BridgeGraph

    if not os.path.exists(project_path):
        console.print(f"[red]Error:[/red] Path not found: {project_path}")
        raise typer.Exit(1)

    graph_obj = BridgeGraph()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Building knowledge graph...", total=None)
        stats = graph_obj.index_project(project_path)
        progress.remove_task(task)

    rprint(Panel.fit(
        f"Nodes: {stats.total_nodes}  |  Edges: {stats.total_edges}  |  "
        f"Files: {stats.total_files}  |  Functions: {stats.total_functions}  |  "
        f"Classes: {stats.total_classes}\n"
        f"Orphans: {stats.orphan_nodes}  |  Avg Degree: {stats.avg_degree:.1f}  |  Max Depth: {stats.max_depth}",
        title="Knowledge Graph Stats"
    ))

    if query:
        results = graph_obj.search(query)
        if results:
            table = Table(title=f"Search: '{query}'")
            table.add_column("Name", style="bold")
            table.add_column("Type")
            table.add_column("File")
            table.add_column("Complexity")
            for node in results[:30]:
                table.add_row(node.name, node.node_type.value, node.file_path, str(node.complexity))
            console.print(table)
        else:
            console.print(f"[yellow]No results for '{query}'[/yellow]")

    if impact:
        result = graph_obj.impact_analysis(impact)
        if "error" not in result:
            rprint(f"\n[bold]Impact Analysis: {impact}[/bold]")
            rprint(f"Risk Level: [{ 'red' if result['risk_level'] == 'HIGH' else 'yellow' if result['risk_level'] == 'MEDIUM' else 'green' }]{result['risk_level']}[/]")
            rprint(f"Direct Dependents: {result['direct_dependents']}")
            rprint(f"Transitive Dependents: {result['transitive_dependents']}")
            if result.get("direct_list"):
                for dep in result["direct_list"][:10]:
                    rprint(f"  • {dep.name} ({dep.node_type.value}) - {dep.file_path}")
        else:
            console.print(f"[yellow]Entity '{impact}' not found[/yellow]")

    if dead_code:
        dead = graph_obj.detect_dead_code()
        if dead:
            rprint(f"\n[bold]Potentially Dead Code: {len(dead)} entities[/bold]")
            for node in dead[:20]:
                rprint(f"  • [dim]{node.name}[/dim] in {node.file_path}:{node.line_start}")
        else:
            rprint("\n[green]No dead code detected![/green]")

    if output:
        dep_graph = graph_obj.get_dependency_graph()
        with open(output, "w", encoding="utf-8") as f:
            json.dump(dep_graph, f, indent=2)


@app.command()
def init():
    """Initialize CodeBridge in the current project."""
    config_dir = Path(".codebridge")
    if config_dir.exists():
        console.print("[yellow]CodeBridge already initialized in this directory.[/yellow]")
        return

    config_dir.mkdir(exist_ok=True)

    # Create default config
    config = {
        "version": _get_version(),
        "agents": {
            "lint": {"enabled": True, "weight": 0.15},
            "security": {"enabled": True, "weight": 0.30},
            "logic": {"enabled": True, "weight": 0.25},
            "architecture": {"enabled": True, "weight": 0.20},
            "performance": {"enabled": True, "weight": 0.10},
        },
        "gate": {
            "production": {
                "security_score": {"threshold": 8.0, "weight": 0.30},
                "review_score": {"threshold": 8.0, "weight": 0.25},
                "test_coverage": {"threshold": 80.0, "weight": 0.25},
                "breaking_changes": {"threshold": 0, "weight": 0.20},
            }
        },
        "scan": {
            "exclude": [".git", "node_modules", "__pycache__", "dist", "build"],
        },
    }
    with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Create .gitignore entry
    gitignore_path = Path(".gitignore")
    entry = "\n# CodeBridge\n.codebridge_cache/\n"
    if gitignore_path.exists():
        with open(gitignore_path, "a") as f:
            f.write(entry)
    else:
        with open(gitignore_path, "w") as f:
            f.write(entry)

    console.print("[green]✅ CodeBridge initialized![/green]")
    console.print(f"   Config: [dim].codebridge/config.yaml[/dim]")
    console.print(f"   Cache:  [dim].codebridge_cache/[/dim] (gitignored)")


def main():
    app()


if __name__ == "__main__":
    main()
