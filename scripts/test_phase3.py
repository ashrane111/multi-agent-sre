#!/usr/bin/env python3
"""
Test script for Phase 3 components.

Tests:
- RAG pipeline initialization and indexing
- Hybrid search (BM25 + dense)
- Cross-encoder reranking
- Episodic memory storage and retrieval
- MCP server stubs

Usage:
    python scripts/test_phase3.py
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_embeddings() -> bool:
    """Test embedding manager."""
    console.print("\n[bold cyan]Testing Embedding Manager...[/]")
    
    try:
        from src.rag.embeddings import get_embedding_manager
        
        manager = get_embedding_manager()
        
        # Test single embedding
        text = "High CPU usage in kubernetes pod"
        embedding = manager.embed_text(text)
        
        console.print(f"  ✅ Embedding generated: dim={len(embedding)}")
        
        # Test batch embeddings
        texts = [
            "Memory leak detected",
            "Pod restart loop",
            "Database connection timeout",
        ]
        embeddings = manager.embed_texts(texts)
        
        console.print(f"  ✅ Batch embeddings: {len(embeddings)} vectors")
        
        # Test caching
        _ = manager.embed_text(text)  # Should hit cache
        stats = manager.cache_stats()
        console.print(f"  ✅ Cache stats: {stats}")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_document_processor() -> bool:
    """Test document processor."""
    console.print("\n[bold cyan]Testing Document Processor...[/]")
    
    try:
        from src.rag.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Process runbooks directory
        runbooks_dir = Path("data/runbooks")
        if not runbooks_dir.exists():
            console.print("  ⚠️ Runbooks directory not found")
            return False
        
        documents = processor.process_directory(runbooks_dir)
        
        console.print(f"  ✅ Processed {len(documents)} documents")
        
        for doc in documents:
            console.print(f"    - {doc.metadata.get('source')}: {len(doc.chunks)} chunks")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_vector_store() -> bool:
    """Test ChromaDB vector store."""
    console.print("\n[bold cyan]Testing Vector Store...[/]")
    
    try:
        from src.rag.document_processor import DocumentChunk
        from src.rag.vector_store import VectorStore
        
        # Use test collection
        store = VectorStore(collection_name="test_runbooks")
        
        # Clear existing data
        store.clear()
        
        # Add test chunks
        chunks = [
            DocumentChunk(
                chunk_id="test-001",
                content="High CPU usage troubleshooting guide",
                metadata={"source": "test.md", "section": "CPU"},
            ),
            DocumentChunk(
                chunk_id="test-002",
                content="Memory leak detection and remediation",
                metadata={"source": "test.md", "section": "Memory"},
            ),
            DocumentChunk(
                chunk_id="test-003",
                content="Pod crash loop debugging steps",
                metadata={"source": "test.md", "section": "Pods"},
            ),
        ]
        
        store.add_chunks(chunks)
        console.print(f"  ✅ Added {len(chunks)} chunks")
        
        # Test search
        results = store.search("cpu high usage", top_k=2)
        console.print(f"  ✅ Search returned {len(results)} results")
        
        for r in results:
            console.print(f"    - {r['id']}: score={r['score']:.3f}")
        
        # Cleanup
        store.clear()
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_hybrid_retriever() -> bool:
    """Test hybrid retriever (BM25 + dense)."""
    console.print("\n[bold cyan]Testing Hybrid Retriever...[/]")
    
    try:
        from src.rag.document_processor import DocumentChunk
        from src.rag.hybrid_retriever import HybridRetriever
        from src.rag.vector_store import VectorStore
        
        # Setup
        store = VectorStore(collection_name="test_hybrid")
        store.clear()
        
        chunks = [
            DocumentChunk(
                chunk_id="h-001",
                content="kubectl scale deployment to increase replicas for high traffic",
                metadata={"source": "scaling.md"},
            ),
            DocumentChunk(
                chunk_id="h-002",
                content="kubectl rollout undo to rollback deployment after failed release",
                metadata={"source": "rollback.md"},
            ),
            DocumentChunk(
                chunk_id="h-003",
                content="check pod logs for application errors and exceptions",
                metadata={"source": "debugging.md"},
            ),
        ]
        
        store.add_chunks(chunks)
        
        retriever = HybridRetriever(vector_store=store)
        retriever.build_bm25_index(chunks)
        
        # Test hybrid search
        results = retriever.search("scale deployment replicas", top_k=2)
        
        console.print(f"  ✅ Hybrid search returned {len(results)} results")
        for r in results:
            console.print(f"    - {r['id']}: fused_score={r['fused_score']:.3f}")
        
        # Cleanup
        store.clear()
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_reranker() -> bool:
    """Test cross-encoder reranker."""
    console.print("\n[bold cyan]Testing Cross-Encoder Reranker...[/]")
    
    try:
        from src.rag.reranker import get_reranker
        
        reranker = get_reranker()
        
        query = "How to fix high CPU usage"
        documents = [
            {"id": "1", "content": "Increase memory limits in kubernetes deployment"},
            {"id": "2", "content": "Scale up replicas to handle high CPU load"},
            {"id": "3", "content": "Check network connectivity issues"},
        ]
        
        reranked = reranker.rerank(query, documents, top_k=2)
        
        console.print(f"  ✅ Reranked {len(documents)} -> {len(reranked)} results")
        for r in reranked:
            console.print(f"    - {r['id']}: rerank_score={r['rerank_score']:.3f}")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_rag_pipeline() -> bool:
    """Test full RAG pipeline."""
    console.print("\n[bold cyan]Testing Full RAG Pipeline...[/]")
    
    try:
        from src.rag import get_rag_pipeline
        
        # Initialize pipeline (auto-indexes runbooks)
        pipeline = get_rag_pipeline(collection_name="runbooks", auto_index=True)
        
        stats = pipeline.stats()
        console.print(f"  ✅ Pipeline initialized: {stats['document_count']} chunks indexed")
        
        # Test retrieval
        result = pipeline.retrieve(
            query="high CPU usage remediation kubernetes",
            top_k=5,
            rerank=True,
            rerank_top_k=3,
        )
        
        console.print(f"  ✅ Retrieved {len(result.documents)} documents")
        console.print(f"  Sources: {result.metadata['sources']}")
        
        # Show formatted context preview
        context_preview = result.context[:200] + "..." if len(result.context) > 200 else result.context
        console.print(f"  Context preview: {context_preview}")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_client() -> bool:
    """Test Mem0 client."""
    console.print("\n[bold cyan]Testing Memory Client...[/]")
    
    try:
        from src.memory.mem0_client import Mem0Client
        
        client = Mem0Client(collection_name="test_memory")
        
        # Add test memory
        memory_id = client.add_memory(
            content="High CPU incident resolved by scaling replicas",
            metadata={"incident_id": "TEST-001", "severity": "P3"},
        )
        console.print(f"  ✅ Memory added: {memory_id}")
        
        # Search memories
        results = client.search_memories(
            query="CPU scaling",
            top_k=3,
        )
        console.print(f"  ✅ Search returned {len(results)} results")
        
        # Get stats
        stats = client.stats()
        console.print(f"  ✅ Stats: {stats}")
        
        # Cleanup
        client.clear_memories()
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


def test_incident_memory() -> bool:
    """Test incident memory."""
    console.print("\n[bold cyan]Testing Incident Memory...[/]")
    
    try:
        from src.memory import get_incident_memory
        from src.workflows.states import IncidentState
        
        memory = get_incident_memory()
        
        # Create mock incident state
        state = IncidentState(
            incident_id="TEST-INC-001",
            severity="P3",
            cluster="test-cluster",
            namespace="production",
            title="High CPU Alert",
            description="API server showing high CPU usage",
            anomalies=["CPU usage > 90%", "Response latency increased"],
            affected_resources=["deployment/api-server"],
            root_cause="Traffic spike during peak hours",
            diagnosis="Increased load from marketing campaign",
            time_to_resolve_minutes=30.0,
        )
        
        # Store incident
        memory_id = memory.store_incident(state, resolution_success=True)
        console.print(f"  ✅ Incident stored: {memory_id}")
        
        # Recall similar
        similar = memory.recall_similar_incidents(state, top_k=3)
        console.print(f"  ✅ Found {len(similar)} similar incidents")
        
        # Get stats
        stats = memory.stats()
        console.print(f"  ✅ Resolution stats: {stats.get('resolution_stats', {})}")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prometheus_mcp() -> bool:
    """Test Prometheus MCP server."""
    console.print("\n[bold cyan]Testing Prometheus MCP Server...[/]")
    
    try:
        from src.mcp import get_prometheus_mcp
        
        server = get_prometheus_mcp()
        
        # Get tools
        tools = server.get_tools()
        console.print(f"  ✅ Available tools: {[t['name'] for t in tools]}")
        
        # Test query
        result = await server.execute_tool(
            "prometheus_query",
            {"query": "container_cpu_usage_seconds_total"},
        )
        console.print(f"  ✅ Query returned {len(result.get('results', []))} results")
        
        # Test alerts
        alerts = await server.execute_tool("prometheus_alerts", {})
        console.print(f"  ✅ Found {len(alerts.get('alerts', []))} alerts")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


async def test_kubernetes_mcp() -> bool:
    """Test Kubernetes MCP server."""
    console.print("\n[bold cyan]Testing Kubernetes MCP Server...[/]")
    
    try:
        from src.mcp import get_kubernetes_mcp
        
        server = get_kubernetes_mcp()
        
        # Get tools
        tools = server.get_tools()
        console.print(f"  ✅ Available tools: {[t['name'] for t in tools]}")
        
        # Test get pods
        pods = await server.execute_tool(
            "k8s_get_pods",
            {"namespace": "production"},
        )
        console.print(f"  ✅ Found {pods.get('total', 0)} pods")
        
        # Test get events
        events = await server.execute_tool(
            "k8s_get_events",
            {"namespace": "production"},
        )
        console.print(f"  ✅ Found {len(events.get('events', []))} events")
        
        return True
    except Exception as e:
        console.print(f"  ❌ Error: {e}")
        return False


async def main() -> None:
    """Run all Phase 3 tests."""
    console.print(Panel.fit(
        "[bold green]Phase 3 Component Tests[/]\n"
        "RAG Pipeline + Episodic Memory + MCP Servers",
        border_style="green",
    ))
    
    results = {}
    
    # Synchronous tests
    results["Embeddings"] = test_embeddings()
    results["Document Processor"] = test_document_processor()
    results["Vector Store"] = test_vector_store()
    results["Hybrid Retriever"] = test_hybrid_retriever()
    results["Reranker"] = test_reranker()
    results["RAG Pipeline"] = test_rag_pipeline()
    results["Memory Client"] = test_memory_client()
    results["Incident Memory"] = test_incident_memory()
    
    # Async tests
    results["Prometheus MCP"] = await test_prometheus_mcp()
    results["Kubernetes MCP"] = await test_kubernetes_mcp()
    
    # Summary
    console.print("\n")
    table = Table(title="Test Results Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    passed = 0
    failed = 0
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        table.add_row(name, status)
        if success:
            passed += 1
        else:
            failed += 1
    
    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed[/]")
    
    if failed > 0:
        console.print("[yellow]Some tests failed. Check output above for details.[/]")
    else:
        console.print("[green]All Phase 3 tests passed! 🎉[/]")


if __name__ == "__main__":
    asyncio.run(main())
