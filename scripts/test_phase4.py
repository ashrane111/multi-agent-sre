#!/usr/bin/env python3
"""
Phase 4 Component Tests: HITL & Governance

Tests:
- Approval Manager
- Slack Notifier
- Audit Trail
- API Routes
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_approval_manager():
    """Test the approval manager."""
    console.print("\n[bold cyan]Testing Approval Manager...[/]")
    
    from src.hitl.approval_manager import ApprovalManager, ApprovalStatus
    
    manager = ApprovalManager(default_timeout_minutes=5)
    
    # Test creating approval request
    request = await manager.create_request(
        incident_id="TEST-INC-001",
        actions=[
            {"action_id": "act-1", "action_type": "scale", "command": "kubectl scale..."},
            {"action_id": "act-2", "action_type": "restart", "command": "kubectl rollout restart..."},
        ],
        severity="P2",
        policy_violations=[
            {"rule_id": "RULE-001", "rule_name": "Large Scale Change", "severity": "HIGH"},
        ],
        blast_radius_score=0.7,
        root_cause="High CPU usage due to traffic spike",
        diagnosis_confidence=0.85,
    )
    
    assert request.request_id is not None
    assert request.status == ApprovalStatus.PENDING
    console.print(f"  ✅ Created request: {request.request_id}")
    
    # Test getting pending
    pending = manager.get_pending()
    assert len(pending) == 1
    console.print(f"  ✅ Pending requests: {len(pending)}")
    
    # Test approving
    decision = await manager.approve(
        request_id=request.request_id,
        approver="test-user@example.com",
        notes="Approved for testing",
    )
    
    assert decision.approved is True
    console.print(f"  ✅ Approved by: {decision.approver}")
    
    # Verify moved to resolved
    pending = manager.get_pending()
    resolved = manager.get_resolved()
    assert len(pending) == 0
    assert len(resolved) == 1
    console.print(f"  ✅ Resolved requests: {len(resolved)}")
    
    # Test rejection flow
    request2 = await manager.create_request(
        incident_id="TEST-INC-002",
        actions=[{"action_id": "act-3", "action_type": "delete", "command": "kubectl delete..."}],
        severity="P1",
    )
    
    decision2 = await manager.reject(
        request_id=request2.request_id,
        approver="security-team@example.com",
        notes="Too risky without rollback plan",
    )
    
    assert decision2.approved is False
    console.print(f"  ✅ Rejected: {decision2.notes}")
    
    # Test stats
    stats = manager.stats()
    assert stats["resolved_count"] == 2
    console.print(f"  ✅ Stats: {stats}")
    
    return True


async def test_slack_notifier():
    """Test the Slack notifier (stub mode)."""
    console.print("\n[bold cyan]Testing Slack Notifier...[/]")
    
    from src.hitl.slack_notifier import SlackNotifier
    from src.hitl.approval_manager import ApprovalRequest, ApprovalStatus
    
    notifier = SlackNotifier(channel="#test-channel")
    
    # Create a mock request
    request = ApprovalRequest(
        request_id="APR-TEST-001",
        incident_id="INC-TEST-001",
        actions=[
            {"action_type": "scale", "description": "Scale deployment to 10 replicas"},
        ],
        policy_violations=[
            {"rule_id": "R1", "rule_name": "Scale Limit", "severity": "HIGH", "message": "Exceeds limit"},
        ],
        severity="P2",
        root_cause="Traffic spike causing high latency",
        diagnosis_confidence=0.9,
        blast_radius_score=0.5,
        expires_at=datetime.utcnow() + timedelta(minutes=15),
    )
    
    # Test sending notification (stub mode - just logs)
    result = await notifier.send_approval_request(request)
    assert result.get("ok") is True
    console.print(f"  ✅ Notification sent (stub mode)")
    
    # Test building blocks
    blocks = notifier._build_approval_blocks(request)
    assert len(blocks) > 0
    assert blocks[0]["type"] == "header"
    console.print(f"  ✅ Built {len(blocks)} Slack blocks")
    
    # Test result notification
    result = await notifier.send_approval_result(
        request=request,
        approved=True,
        approver="test-user",
        notes="LGTM",
    )
    assert result.get("ok") is True
    console.print(f"  ✅ Result notification sent")
    
    return True


def test_audit_trail():
    """Test the audit trail."""
    console.print("\n[bold cyan]Testing Audit Trail...[/]")
    
    import tempfile
    from src.governance.audit_trail import AuditTrail, AuditAction
    
    # Create audit trail with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        audit = AuditTrail(storage_path=tmpdir)
        
        # Test logging entries
        entry1 = audit.log(
            action=AuditAction.INCIDENT_CREATED,
            description="New incident detected",
            incident_id="INC-001",
            agent="monitor",
            risk_level="medium",
        )
        assert entry1.entry_id is not None
        console.print(f"  ✅ Logged: {entry1.action.value}")
        
        entry2 = audit.log(
            action=AuditAction.APPROVAL_REQUESTED,
            description="Approval requested for remediation",
            incident_id="INC-001",
            request_id="APR-001",
            risk_level="high",
        )
        console.print(f"  ✅ Logged: {entry2.action.value}")
        
        entry3 = audit.log(
            action=AuditAction.APPROVAL_GRANTED,
            description="Approved by SRE team",
            incident_id="INC-001",
            request_id="APR-001",
            actor_type="human",
            actor_id="sre@example.com",
        )
        console.print(f"  ✅ Logged: {entry3.action.value}")
        
        # Test querying by incident
        incident_entries = audit.get_by_incident("INC-001")
        assert len(incident_entries) == 3
        console.print(f"  ✅ Entries for INC-001: {len(incident_entries)}")
        
        # Test querying by action
        approval_entries = audit.get_by_action(AuditAction.APPROVAL_GRANTED)
        assert len(approval_entries) == 1
        console.print(f"  ✅ Approval granted entries: {len(approval_entries)}")
        
        # Test getting recent
        recent = audit.get_recent(limit=10)
        assert len(recent) == 3
        console.print(f"  ✅ Recent entries: {len(recent)}")
        
        # Test stats
        stats = audit.stats()
        assert stats["total_entries"] == 3
        assert stats["incidents_tracked"] == 1
        console.print(f"  ✅ Stats: {stats['total_entries']} entries, {stats['incidents_tracked']} incidents")
        
        # Test persistence (create new instance)
        audit2 = AuditTrail(storage_path=tmpdir)
        assert len(audit2.get_recent()) == 3
        console.print(f"  ✅ Persistence verified: {len(audit2.get_recent())} entries reloaded")
    
    return True


async def test_api_routes():
    """Test API routes."""
    console.print("\n[bold cyan]Testing API Routes...[/]")
    
    from fastapi.testclient import TestClient
    from src.main import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    console.print(f"  ✅ Health check: {response.json()['status']}")
    
    # Test approvals endpoint
    response = client.get("/api/v1/approvals/pending")
    assert response.status_code == 200
    data = response.json()
    console.print(f"  ✅ Pending approvals endpoint: {data['pending_count']} pending")
    
    # Test approval stats
    response = client.get("/api/v1/approvals/stats")
    assert response.status_code == 200
    console.print(f"  ✅ Approval stats endpoint: OK")
    
    # Test audit entries endpoint
    response = client.get("/api/v1/audit/entries")
    assert response.status_code == 200
    data = response.json()
    console.print(f"  ✅ Audit entries endpoint: {data['total']} entries")
    
    # Test audit stats endpoint
    response = client.get("/api/v1/audit/stats")
    assert response.status_code == 200
    console.print(f"  ✅ Audit stats endpoint: OK")
    
    return True


async def test_integration():
    """Test integration between components."""
    console.print("\n[bold cyan]Testing Integration...[/]")
    
    from src.hitl.approval_manager import get_approval_manager
    from src.hitl.slack_notifier import get_slack_notifier
    from src.governance.audit_trail import get_audit_trail, AuditAction
    
    manager = get_approval_manager()
    notifier = get_slack_notifier()
    audit = get_audit_trail()
    
    # Register notification handler
    manager.register_notification_handler(notifier.send_approval_request)
    
    # Create approval which should trigger notification
    request = await manager.create_request(
        incident_id="INT-TEST-001",
        actions=[{"action_type": "test", "description": "Test action"}],
        severity="P3",
    )
    
    # Log to audit
    audit.log(
        action=AuditAction.APPROVAL_REQUESTED,
        description="Integration test approval",
        incident_id="INT-TEST-001",
        request_id=request.request_id,
    )
    
    # Approve
    await manager.approve(request.request_id, "integration-test", "Auto-approved")
    
    audit.log(
        action=AuditAction.APPROVAL_GRANTED,
        description="Integration test approved",
        incident_id="INT-TEST-001",
        request_id=request.request_id,
        actor_id="integration-test",
    )
    
    # Verify
    incident_audit = audit.get_by_incident("INT-TEST-001")
    assert len(incident_audit) >= 2
    console.print(f"  ✅ Integration flow completed: {len(incident_audit)} audit entries")
    
    return True


async def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold]Phase 4 Component Tests[/]\n"
        "HITL & Governance",
        title="🧪 Test Suite",
    ))
    
    results = {}
    
    # Run tests
    tests = [
        ("Approval Manager", test_approval_manager),
        ("Slack Notifier", test_slack_notifier),
        ("Audit Trail", test_audit_trail),
        ("API Routes", test_api_routes),
        ("Integration", test_integration),
    ]
    
    for name, test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                result = await test_fn()
            else:
                result = test_fn()
            results[name] = result
        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/]")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    console.print("\n")
    table = Table(title="Test Results Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        table.add_row(name, status)
        if result:
            passed += 1
        else:
            failed += 1
    
    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed[/]")
    
    if failed > 0:
        console.print("[red]Some tests failed. Check output above for details.[/]")
        sys.exit(1)
    else:
        console.print("[green]All tests passed! ✨[/]")


if __name__ == "__main__":
    # Setup logging
    from src.logging_config import setup_logging
    setup_logging()
    
    asyncio.run(main())
