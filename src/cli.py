"""
CLI for Multi-Agent SRE Platform.

Provides commands for:
- Running the API server
- Testing LLM connectivity
- Simulating incidents
- Viewing cost reports
"""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src import __version__

app = typer.Typer(
    name="sre-platform",
    help="Multi-Agent SRE Platform CLI",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold blue]Multi-Agent SRE Platform[/] v{__version__}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of workers"),
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"[bold green]Starting server on {host}:{port}[/]")
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def check_llm(
    provider: str = typer.Option("all", help="Provider to check: all, ollama, groq, gemini"),
) -> None:
    """Test LLM provider connectivity."""
    from src.config import settings
    from src.logging_config import setup_logging

    setup_logging()

    async def _check() -> None:
        import httpx

        checks = {}

        if provider in ("all", "ollama"):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
                    if resp.status_code == 200:
                        models = resp.json().get("models", [])
                        checks["ollama"] = {
                            "status": "✅ Connected",
                            "models": [m["name"] for m in models[:5]],
                        }
                    else:
                        checks["ollama"] = {"status": "❌ Error", "detail": resp.status_code}
            except Exception as e:
                checks["ollama"] = {"status": "❌ Error", "detail": str(e)}

        if provider in ("all", "groq"):
            if settings.llm.groq_api_key:
                try:
                    from langchain_groq import ChatGroq

                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        api_key=settings.llm.groq_api_key.get_secret_value(),
                    )
                    result = await llm.ainvoke("Say 'connected' in one word.")
                    checks["groq"] = {"status": "✅ Connected", "response": result.content[:50]}
                except Exception as e:
                    checks["groq"] = {"status": "❌ Error", "detail": str(e)}
            else:
                checks["groq"] = {"status": "⚠️ No API key configured"}

        if provider in ("all", "gemini"):
            if settings.llm.google_api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-flash-latest",
                        google_api_key=settings.llm.google_api_key.get_secret_value(),
                    )
                    result = await llm.ainvoke("Say 'connected' in one word.")
                    checks["gemini"] = {"status": "✅ Connected", "response": result.content[:50]}
                except Exception as e:
                    checks["gemini"] = {"status": "❌ Error", "detail": str(e)}
            else:
                checks["gemini"] = {"status": "⚠️ No API key configured"}

        # Display results
        table = Table(title="LLM Provider Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        for prov, info in checks.items():
            details = ""
            if "models" in info:
                details = f"Models: {', '.join(info['models'][:3])}..."
            elif "response" in info:
                details = info["response"]
            elif "detail" in info:
                details = str(info["detail"])[:50]

            table.add_row(prov, info["status"], details)

        console.print(table)

    asyncio.run(_check())


@app.command()
def cost_report() -> None:
    """Show LLM usage cost report."""
    from src.models.router import get_model_router

    router = get_model_router()
    report = router.get_cost_report()

    # Summary table
    summary_table = Table(title="Cost Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    for key, value in report["summary"].items():
        summary_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(summary_table)

    # Local vs Cloud table
    lvc_table = Table(title="Local vs Cloud")
    lvc_table.add_column("Metric", style="cyan")
    lvc_table.add_column("Value", style="green")

    for key, value in report["local_vs_cloud"].items():
        lvc_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(lvc_table)


@app.command()
def simulate_incident(
    severity: str = typer.Option("P3", help="Incident severity: P1, P2, P3, P4"),
    cluster: str = typer.Option("cluster-a", help="Cluster name"),
    dry_run: bool = typer.Option(False, help="Don't actually execute remediation"),
) -> None:
    """Simulate an incident for testing."""
    console.print(f"[bold yellow]Simulating {severity} incident in {cluster}[/]")
    console.print("[dim]To be implemented in Phase 2[/]")


@app.command()
def init_db() -> None:
    """Initialize databases (ChromaDB, Redis)."""
    console.print("[bold]Initializing databases...[/]")

    # Create data directories
    import os

    os.makedirs("data/chroma_db", exist_ok=True)
    os.makedirs("data/chroma_runbooks", exist_ok=True)
    os.makedirs("data/runbooks", exist_ok=True)
    os.makedirs("data/policies", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    console.print("[green]✅ Data directories created[/]")
    console.print("[dim]Run 'docker-compose up -d' to start Redis[/]")


@app.command()
def pull_models() -> None:
    """Pull required Ollama models."""
    import subprocess

    models = ["llama3.1:8b", "mistral:7b", "nomic-embed-text"]

    for model in models:
        console.print(f"[bold]Pulling {model}...[/]")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            console.print(f"[green]✅ {model} pulled successfully[/]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to pull {model}: {e}[/]")
        except FileNotFoundError:
            console.print("[red]❌ Ollama not found. Install from https://ollama.com[/]")
            break


@app.command()
def index_runbooks(
    directory: str = typer.Option("data/runbooks", help="Runbooks directory"),
    clear: bool = typer.Option(False, help="Clear existing index first"),
) -> None:
    """Index runbook documents for RAG retrieval."""
    from pathlib import Path

    from src.logging_config import setup_logging

    setup_logging()

    console.print(f"[bold]Indexing runbooks from {directory}...[/]")

    try:
        from src.rag import get_rag_pipeline

        pipeline = get_rag_pipeline(auto_index=False)

        if clear:
            console.print("[yellow]Clearing existing index...[/]")
            pipeline.clear_index()

        documents = pipeline.index_directory(Path(directory))

        # Display results
        table = Table(title="Indexed Documents")
        table.add_column("Document", style="cyan")
        table.add_column("Chunks", style="green")
        table.add_column("Title", style="dim")

        for doc in documents:
            table.add_row(
                doc.metadata.get("source", "unknown"),
                str(len(doc.chunks)),
                doc.metadata.get("title", "")[:40],
            )

        console.print(table)
        console.print(f"\n[green]✅ Indexed {len(documents)} documents[/]")

        # Show stats
        stats = pipeline.stats()
        console.print(f"[dim]Total chunks: {stats['document_count']}[/]")

    except Exception as e:
        console.print(f"[red]❌ Error indexing runbooks: {e}[/]")
        raise typer.Exit(1)


@app.command()
def rag_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(3, help="Number of results"),
) -> None:
    """Search runbook documentation using RAG."""
    from src.logging_config import setup_logging

    setup_logging()

    console.print(f"[bold]Searching for: {query}[/]")

    try:
        from src.rag import get_rag_pipeline

        pipeline = get_rag_pipeline()
        results = pipeline.search_simple(query, top_k=top_k)

        if not results:
            console.print("[yellow]No results found.[/]")
            return

        for i, result in enumerate(results, 1):
            console.print(f"\n[bold cyan]Result {i}[/] (score: {result['score']:.3f})")
            console.print(f"[dim]Source: {result['source']} - {result['section']}[/]")
            console.print(result["content"][:500])
            if len(result["content"]) > 500:
                console.print("[dim]... (truncated)[/]")

    except Exception as e:
        console.print(f"[red]❌ Error searching: {e}[/]")
        raise typer.Exit(1)


@app.command()
def rag_stats() -> None:
    """Show RAG pipeline statistics."""
    from src.logging_config import setup_logging

    setup_logging()

    try:
        from src.rag import get_rag_pipeline

        pipeline = get_rag_pipeline(auto_index=False)
        stats = pipeline.stats()

        # Main stats
        table = Table(title="RAG Pipeline Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collection", stats["collection_name"])
        table.add_row("Document Count", str(stats["document_count"]))
        table.add_row("Runbooks Dir", stats["runbooks_dir"])
        table.add_row("Sources", ", ".join(stats["indexed_sources"]) or "None")

        console.print(table)

        # Retriever stats
        retriever_stats = stats.get("hybrid_retriever", {})
        if retriever_stats:
            console.print("\n[bold]Hybrid Retriever:[/]")
            console.print(f"  Dense weight: {retriever_stats.get('dense_weight', 'N/A')}")
            console.print(f"  Sparse weight: {retriever_stats.get('sparse_weight', 'N/A')}")
            console.print(f"  BM25 index size: {retriever_stats.get('bm25_index_size', 0)}")

    except Exception as e:
        console.print(f"[red]❌ Error getting stats: {e}[/]")
        raise typer.Exit(1)


@app.command()
def memory_stats() -> None:
    """Show episodic memory statistics."""
    from src.logging_config import setup_logging

    setup_logging()

    try:
        from src.memory import get_incident_memory

        memory = get_incident_memory()
        stats = memory.stats()

        # Memory stats
        table = Table(title="Episodic Memory Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collection", stats.get("collection_name", "N/A"))
        table.add_row("Mem0 Available", "✅" if stats.get("mem0_available") else "❌")
        table.add_row("Mem0 Memories", str(stats.get("mem0_memory_count", 0)))
        table.add_row("Local Memories", str(stats.get("local_memory_count", 0)))

        console.print(table)

        # Resolution stats
        res_stats = stats.get("resolution_stats", {})
        if res_stats.get("total_incidents", 0) > 0:
            console.print("\n[bold]Resolution Statistics:[/]")
            console.print(f"  Total incidents: {res_stats.get('total_incidents', 0)}")
            console.print(f"  Success rate: {res_stats.get('success_rate', 0):.1%}")
            console.print(f"  Avg resolution time: {res_stats.get('average_resolution_time_minutes', 0):.1f} min")

            by_severity = res_stats.get("by_severity", {})
            if by_severity:
                console.print(f"  By severity: {by_severity}")

    except Exception as e:
        console.print(f"[red]❌ Error getting memory stats: {e}[/]")
        raise typer.Exit(1)


@app.command()
def memory_clear(
    confirm: bool = typer.Option(False, "--confirm", help="Confirm clearing all memories"),
) -> None:
    """Clear all episodic memories."""
    if not confirm:
        console.print("[yellow]Use --confirm to actually clear memories[/]")
        return

    from src.logging_config import setup_logging

    setup_logging()

    try:
        from src.memory import get_incident_memory

        memory = get_incident_memory()
        count = memory.clear_history()

        console.print(f"[green]✅ Cleared {count} memories[/]")

    except Exception as e:
        console.print(f"[red]❌ Error clearing memories: {e}[/]")
        raise typer.Exit(1)


# ==================== Phase 4: HITL & Governance Commands ====================


@app.command()
def approvals_pending() -> None:
    """List pending approval requests."""
    from src.hitl import get_approval_manager
    from src.logging_config import setup_logging

    setup_logging()

    manager = get_approval_manager()
    pending = manager.get_pending()

    if not pending:
        console.print("[dim]No pending approvals[/]")
        return

    table = Table(title="Pending Approvals")
    table.add_column("Request ID", style="cyan")
    table.add_column("Incident", style="yellow")
    table.add_column("Severity", style="red")
    table.add_column("Actions", style="green")
    table.add_column("Violations", style="magenta")
    table.add_column("Expires", style="dim")

    for req in pending:
        expires_in = ""
        if req.expires_at:
            from datetime import datetime
            delta = (req.expires_at - datetime.utcnow()).total_seconds() / 60
            expires_in = f"{delta:.0f} min"

        table.add_row(
            req.request_id,
            req.incident_id,
            req.severity,
            str(len(req.actions)),
            str(len(req.policy_violations)),
            expires_in,
        )

    console.print(table)


@app.command()
def approve(
    request_id: str = typer.Argument(..., help="Approval request ID"),
    approver: str = typer.Option("cli-user", help="Approver identifier"),
    notes: str = typer.Option("", help="Approval notes"),
) -> None:
    """Approve a pending request."""
    import asyncio

    from src.hitl import get_approval_manager
    from src.logging_config import setup_logging

    setup_logging()

    async def do_approve():
        manager = get_approval_manager()
        try:
            result = await manager.approve(request_id, approver, notes)
            console.print(f"[green]✅ Approved {result.request_id}[/]")
        except ValueError as e:
            console.print(f"[red]❌ {e}[/]")
            raise typer.Exit(1)

    asyncio.run(do_approve())


@app.command()
def reject(
    request_id: str = typer.Argument(..., help="Approval request ID"),
    approver: str = typer.Option("cli-user", help="Rejector identifier"),
    notes: str = typer.Option("", help="Rejection reason"),
) -> None:
    """Reject a pending request."""
    import asyncio

    from src.hitl import get_approval_manager
    from src.logging_config import setup_logging

    setup_logging()

    async def do_reject():
        manager = get_approval_manager()
        try:
            result = await manager.reject(request_id, approver, notes)
            console.print(f"[yellow]❌ Rejected {result.request_id}[/]")
        except ValueError as e:
            console.print(f"[red]❌ {e}[/]")
            raise typer.Exit(1)

    asyncio.run(do_reject())


@app.command()
def audit_log(
    limit: int = typer.Option(20, help="Number of entries to show"),
    incident_id: str = typer.Option(None, help="Filter by incident ID"),
    action: str = typer.Option(None, help="Filter by action type"),
) -> None:
    """View audit trail entries."""
    from src.governance import AuditAction, get_audit_trail
    from src.logging_config import setup_logging

    setup_logging()

    audit = get_audit_trail()

    if incident_id:
        entries = audit.get_by_incident(incident_id)[:limit]
    elif action:
        try:
            action_enum = AuditAction(action)
            entries = audit.get_by_action(action_enum)[:limit]
        except ValueError:
            console.print(f"[red]Invalid action: {action}[/]")
            console.print(f"Valid actions: {[a.value for a in AuditAction]}")
            raise typer.Exit(1)
    else:
        entries = audit.get_recent(limit=limit)

    if not entries:
        console.print("[dim]No audit entries found[/]")
        return

    table = Table(title=f"Audit Trail (last {len(entries)})")
    table.add_column("Time", style="dim")
    table.add_column("Action", style="cyan")
    table.add_column("Incident", style="yellow")
    table.add_column("Actor", style="green")
    table.add_column("Description")
    table.add_column("Risk", style="red")

    for entry in entries:
        risk_style = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }.get(entry.risk_level, "white")

        table.add_row(
            entry.timestamp.strftime("%H:%M:%S"),
            entry.action.value,
            entry.incident_id or "-",
            entry.actor_id or entry.actor_type,
            entry.description[:50],
            f"[{risk_style}]{entry.risk_level}[/]",
        )

    console.print(table)


@app.command()
def audit_stats() -> None:
    """Show audit trail statistics."""
    from src.governance import get_audit_trail
    from src.logging_config import setup_logging

    setup_logging()

    audit = get_audit_trail()
    stats = audit.stats()

    table = Table(title="Audit Trail Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entries", str(stats["total_entries"]))
    table.add_row("Incidents Tracked", str(stats["incidents_tracked"]))
    table.add_row("Failures", str(stats["failure_count"]))

    console.print(table)

    # Risk distribution
    console.print("\n[bold]Risk Distribution:[/]")
    for level, count in stats["risk_distribution"].items():
        bar = "█" * min(count, 20)
        console.print(f"  {level:10} {bar} {count}")

    # Top actions
    if stats["action_counts"]:
        console.print("\n[bold]Top Actions:[/]")
        sorted_actions = sorted(stats["action_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
        for action, count in sorted_actions:
            console.print(f"  {action}: {count}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"[bold]Starting SRE Platform API on {host}:{port}[/]")
    console.print("[dim]Docs available at /docs[/]")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
