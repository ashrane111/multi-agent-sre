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


if __name__ == "__main__":
    app()
