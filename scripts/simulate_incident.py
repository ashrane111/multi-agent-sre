#!/usr/bin/env python3
"""
Simulate an incident for testing the multi-agent workflow.

This script demonstrates the complete incident response flow:
1. Creates an incident with sample metrics
2. Runs through all agents
3. Displays the results

Usage:
    python scripts/simulate_incident.py
    python scripts/simulate_incident.py --severity P2 --cluster production
"""

import asyncio
import argparse
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


async def main(
    severity: str = "P3",
    cluster: str = "production-cluster",
    namespace: str = "api",
    incident_type: str = "high_cpu",
) -> None:
    """Run the incident simulation."""
    
    # Import here to avoid loading everything at CLI parse time
    from src.logging_config import setup_logging
    from src.workflows.incident_workflow import IncidentWorkflow
    from src.workflows.states import IncidentSeverity

    setup_logging()

    # Map incident type to title/description
    incidents = {
        "high_cpu": {
            "title": "High CPU Usage Alert",
            "description": "CPU utilization exceeded 85% threshold for 5 minutes",
            "alert_source": "prometheus-alertmanager",
        },
        "pod_crash": {
            "title": "Pod CrashLoopBackOff Detected",
            "description": "Multiple pods in CrashLoopBackOff state with OOMKilled",
            "alert_source": "kubernetes-events",
        },
        "memory_leak": {
            "title": "Memory Leak Suspected",
            "description": "Gradual memory increase over 24 hours, approaching limits",
            "alert_source": "grafana-alert",
        },
        "latency_spike": {
            "title": "P99 Latency Spike",
            "description": "API response time exceeded 500ms threshold",
            "alert_source": "datadog",
        },
    }

    incident_config = incidents.get(incident_type, incidents["high_cpu"])

    # Display start banner
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]🚨 Simulating {severity} Incident[/]\n"
            f"[dim]{incident_config['title']}[/]",
            border_style="blue",
        )
    )
    console.print()

    # Show configuration
    config_table = Table(title="Incident Configuration", show_header=False)
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Severity", severity)
    config_table.add_row("Cluster", cluster)
    config_table.add_row("Namespace", namespace)
    config_table.add_row("Alert Source", incident_config["alert_source"])
    config_table.add_row("Title", incident_config["title"])
    console.print(config_table)
    console.print()

    # Create and run workflow
    console.print("[bold yellow]Starting workflow...[/]")
    console.print()

    workflow = IncidentWorkflow(enable_checkpointing=True)
    
    start_time = datetime.utcnow()
    
    try:
        final_state = await workflow.run(
            title=incident_config["title"],
            description=incident_config["description"],
            cluster=cluster,
            namespace=namespace,
            alert_source=incident_config["alert_source"],
            severity=IncidentSeverity(severity),
        )

        duration = (datetime.utcnow() - start_time).total_seconds()

        # Display results
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]✅ Workflow Complete[/]\n"
                f"Duration: {duration:.2f}s",
                border_style="green",
            )
        )
        console.print()

        # Results table
        results_table = Table(title="Incident Summary")
        results_table.add_column("Field", style="cyan")
        results_table.add_column("Value", style="white")

        results_table.add_row("Incident ID", final_state.incident_id)
        results_table.add_row("Status", f"[green]{final_state.status}[/]" if final_state.status == "resolved" else f"[red]{final_state.status}[/]")
        results_table.add_row("Severity", final_state.severity)
        results_table.add_row("Root Cause", final_state.root_cause[:80] + "..." if final_state.root_cause and len(final_state.root_cause) > 80 else final_state.root_cause or "N/A")
        results_table.add_row("Confidence", f"{final_state.confidence:.0%}" if final_state.confidence else "N/A")
        results_table.add_row("Policy Decision", str(final_state.policy_decision))
        results_table.add_row("Human Approved", str(final_state.human_approved))
        results_table.add_row("Resolution Time", f"{final_state.time_to_resolve_minutes:.1f} min" if final_state.time_to_resolve_minutes else "N/A")
        results_table.add_row("LLM Cost", f"${final_state.total_llm_cost_usd:.4f}")
        results_table.add_row("Agents Involved", ", ".join(final_state.agents_involved))

        console.print(results_table)
        console.print()

        # Anomalies
        if final_state.anomalies:
            console.print("[bold]Detected Anomalies:[/]")
            for anomaly in final_state.anomalies:
                console.print(f"  • {anomaly}")
            console.print()

        # Actions
        if final_state.executed_actions:
            actions_table = Table(title="Executed Actions")
            actions_table.add_column("Action", style="cyan")
            actions_table.add_column("Status", style="white")
            actions_table.add_column("Result", style="dim")

            for action in final_state.executed_actions:
                status_color = "green" if action.status == "success" else "red" if action.status == "failed" else "yellow"
                actions_table.add_row(
                    action.action.description[:50],
                    f"[{status_color}]{action.status}[/]",
                    (action.result or action.error or "")[:40],
                )

            console.print(actions_table)
            console.print()

        # Policy violations
        if final_state.policy_violations:
            console.print("[bold yellow]Policy Violations:[/]")
            for violation in final_state.policy_violations:
                console.print(f"  ⚠️  [{violation.severity}] {violation.rule_name}: {violation.message}")
            console.print()

        # Similar past incidents
        if final_state.similar_past_incidents:
            console.print("[bold]Similar Past Incidents (from memory):[/]")
            for inc in final_state.similar_past_incidents:
                console.print(f"  📝 {inc.incident_id} ({inc.similarity_score:.0%} similar)")
                console.print(f"     Root cause: {inc.root_cause[:60]}...")
            console.print()

        # Resolution summary
        if final_state.resolution_summary:
            console.print(
                Panel(
                    final_state.resolution_summary,
                    title="Resolution Summary",
                    border_style="green",
                )
            )

    except Exception as e:
        console.print(f"[bold red]❌ Workflow failed:[/] {e}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate an incident for testing the multi-agent SRE workflow"
    )
    parser.add_argument(
        "--severity",
        choices=["P1", "P2", "P3", "P4"],
        default="P3",
        help="Incident severity (default: P3)",
    )
    parser.add_argument(
        "--cluster",
        default="production-cluster",
        help="Cluster name (default: production-cluster)",
    )
    parser.add_argument(
        "--namespace",
        default="api",
        help="Namespace (default: api)",
    )
    parser.add_argument(
        "--type",
        choices=["high_cpu", "pod_crash", "memory_leak", "latency_spike"],
        default="high_cpu",
        dest="incident_type",
        help="Incident type (default: high_cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            severity=args.severity,
            cluster=args.cluster,
            namespace=args.namespace,
            incident_type=args.incident_type,
        )
    )
