# 🤖 Multi-Agent SRE Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-orange.svg)](https://github.com/langchain-ai/langgraph)

> **Autonomous incident response system using multi-agent AI with human-in-the-loop governance**

An intelligent SRE platform that detects, diagnoses, and remediates infrastructure incidents autonomously. Built with LangGraph for orchestration, featuring production-grade reliability patterns, RAG-powered runbook retrieval, and comprehensive audit trails.

## ✨ Highlights

- **~$0.002 per incident** with intelligent LLM routing (Groq → Gemini → Ollama fallback)
- **15-20 second resolution** for typical incidents
- **85%+ diagnosis confidence** using hybrid RAG with cross-encoder reranking
- **Full audit trail** for compliance and accountability
- **Human approval gates** for high-risk remediations

## 🎯 Features

### Core Capabilities
- **5 Specialized AI Agents** coordinated via LangGraph state machine
- **Intelligent LLM Gateway** with circuit breakers and automatic failover
- **Hybrid RAG Pipeline** combining BM25 + semantic search + cross-encoder reranking
- **Episodic Memory** for learning from past incidents
- **Policy Engine** with blast radius analysis and immutable safety rules
- **Human-in-the-Loop** approval workflows with Slack notifications
- **Comprehensive Audit Trail** for all actions and decisions

### Production Reliability
- Circuit breakers with configurable thresholds
- Exponential backoff retry policies with jitter
- LLM provider fallback chain (Groq → Gemini → Ollama)
- Workflow checkpointing for failure recovery
- Automatic rollback on remediation failures

## 🏗️ Architecture
<p align="center">
  <img src="assets/images/Project_architecture.png" alt="Architecture Overview" width="80%"/>
</p>

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local LLM fallback)
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/ashrane111/multi-agent-sre.git
cd multi-agent-sre

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# (Optional) Pull Ollama models for local fallback
ollama pull llama3.1:8b
```

### Run Demo

```bash
# Index runbook documentation
sre-platform index-runbooks

# Simulate an incident
python scripts/simulate_incident.py

# View audit trail
sre-platform audit-log

# Start API server (optional)
sre-platform serve
```

## 📁 Project Structure

```
multi-agent-sre/
├── src/
│   ├── agents/          # 5 specialized AI agents
│   │   ├── monitor_agent.py
│   │   ├── diagnose_agent.py
│   │   ├── policy_agent.py
│   │   ├── remediate_agent.py
│   │   └── report_agent.py
│   ├── workflows/       # LangGraph state machine
│   ├── models/          # LLM Gateway with fallback
│   ├── rag/             # Hybrid RAG pipeline
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── rag_pipeline.py
│   ├── memory/          # Episodic memory (Mem0)
│   ├── hitl/            # Human-in-the-loop
│   │   ├── approval_manager.py
│   │   └── slack_notifier.py
│   ├── governance/      # Audit trail & policies
│   ├── reliability/     # Circuit breakers, retries
│   ├── mcp/             # MCP server stubs
│   └── api/             # FastAPI endpoints
├── data/
│   ├── runbooks/        # Incident runbooks (indexed by RAG)
│   ├── policies/        # Immutable safety rules
│   └── audit/           # Audit log storage
├── scripts/
│   ├── simulate_incident.py
│   ├── test_phase3.py
│   └── test_phase4.py
└── tests/
```

## 🤖 Agent Workflow

```
Alert Triggered
      │
      ▼
┌─────────────┐
│   MONITOR   │ ─── Classifies alert, extracts anomalies
│    Agent    │     Escalates severity if needed
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  DIAGNOSE   │ ─── Queries RAG for runbooks
│    Agent    │     Recalls similar past incidents
└──────┬──────┘     Generates root cause + actions
       │
       ▼
┌─────────────┐
│   POLICY    │ ─── Checks against immutable rules
│    Agent    │     Calculates blast radius
└──────┬──────┘     Decides: APPROVED / NEEDS_REVIEW / BLOCKED
       │
       ▼
┌─────────────┐
│    HITL     │ ─── Creates approval request
│   Gateway   │     Sends Slack notification
└──────┬──────┘     Waits for human (or auto-approves P3/P4)
       │
       ▼
┌─────────────┐
│ REMEDIATE   │ ─── Executes approved actions
│    Agent    │     Monitors for failures
└──────┬──────┘     Triggers rollback if needed
       │
       ▼
┌─────────────┐
│   REPORT    │ ─── Generates incident summary
│    Agent    │     Stores in episodic memory
└─────────────┘     Sends final notification
```

## 📊 CLI Commands

```bash
# Incident simulation
python scripts/simulate_incident.py    # Run full workflow demo

# RAG Pipeline
sre-platform index-runbooks            # Index runbook documents
sre-platform rag-search "high cpu"     # Search runbooks
sre-platform rag-stats                 # View RAG statistics

# Memory
sre-platform memory-stats              # View memory statistics
sre-platform memory-clear --confirm    # Clear episodic memory

# HITL & Governance
sre-platform approvals-pending         # List pending approvals
sre-platform approve <request_id>      # Approve a request
sre-platform reject <request_id>       # Reject a request
sre-platform audit-log                 # View audit trail
sre-platform audit-stats               # Audit statistics

# API Server
sre-platform serve                     # Start FastAPI server
sre-platform check-llm                 # Verify LLM connectivity
sre-platform cost-report               # View LLM costs
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness probe |
| `/health/circuits` | GET | Circuit breaker status |
| `/api/v1/approvals/pending` | GET | List pending approvals |
| `/api/v1/approvals/{id}` | GET | Get approval details |
| `/api/v1/approvals/{id}/approve` | POST | Approve request |
| `/api/v1/approvals/{id}/reject` | POST | Reject request |
| `/api/v1/audit/entries` | GET | Query audit trail |
| `/api/v1/audit/stats` | GET | Audit statistics |
| `/api/v1/costs` | GET | LLM cost report |

## 💰 Cost Analysis

| Component | Model | Cost per Call |
|-----------|-------|---------------|
| Classification | Groq llama-3.3-70b | ~$0.0004 |
| Diagnosis | Groq llama-3.3-70b | ~$0.0008 |
| Policy Check | Groq llama-3.3-70b | ~$0.0003 |
| Report Gen | Groq llama-3.3-70b | ~$0.0008 |
| Embeddings | Local (MiniLM) | FREE |
| Reranking | Local (ms-marco) | FREE |

**Total per incident: ~$0.002** (with Groq free tier: 14,400 requests/day)

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Resolution Time | 15-20 seconds |
| Diagnosis Confidence | 85%+ |
| RAG Retrieval Latency | <500ms |
| LLM Fallback Success | 99.9% |
| Cost per Incident | ~$0.002 |

## 🗺️ Roadmap

- [x] **Phase 1-2**: Core Infrastructure + Agent Implementation
- [x] **Phase 3**: Advanced RAG + Episodic Memory
- [x] **Phase 4**: HITL & Governance
- [ ] **Phase 5**: Reliability Hardening (Chaos Testing)
- [ ] **Phase 6**: Security Layer (PII Redaction)
- [ ] **Phase 7**: Enhanced Observability (LangSmith, OpenTelemetry)
- [ ] **Phase 8**: Production Deployment Guide

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Orchestration | LangGraph, Python asyncio |
| LLM Providers | Groq, Google Gemini, Ollama |
| LLM Gateway | LiteLLM |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (MiniLM) |
| Reranking | Cross-encoder (ms-marco) |
| Memory | Mem0 (with local fallback) |
| API | FastAPI, Pydantic |
| CLI | Typer, Rich |

## 🧪 Testing

```bash
# Run Phase 3 tests (RAG + Memory)
python scripts/test_phase3.py

# Run Phase 4 tests (HITL + Governance)
python scripts/test_phase4.py

# Run unit tests
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Ashutosh Rane**
- LinkedIn: [linkedin.com/in/rane-ashutosh](https://linkedin.com/in/rane-ashutosh)
- GitHub: [@ashrane111](https://github.com/ashrane111)

---

Built with LangGraph, Groq, and multi-agent AI
