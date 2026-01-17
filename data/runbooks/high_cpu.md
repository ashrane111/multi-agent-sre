# High CPU Usage Runbook

## Overview
This runbook covers diagnosis and remediation of high CPU usage incidents in Kubernetes clusters.

## Symptoms
- CPU utilization > 80% for extended periods
- Increased response latency
- Pod throttling alerts
- Node pressure conditions

## Diagnosis Steps

### Step 1: Identify the offending pods
```bash
kubectl top pods -n <namespace> --sort-by=cpu
```

### Step 2: Check pod resource limits
```bash
kubectl describe pod <pod-name> -n <namespace> | grep -A5 "Limits:"
```

### Step 3: Review recent deployments
```bash
kubectl rollout history deployment/<deployment-name> -n <namespace>
```

### Step 4: Check for memory leaks causing GC pressure
```bash
kubectl logs <pod-name> -n <namespace> | grep -i "gc\|memory\|heap"
```

## Common Root Causes

1. **Infinite loops in application code** - Check recent code changes
2. **Missing resource limits** - Pods consuming unbounded CPU
3. **Traffic spike** - Sudden increase in requests
4. **Memory pressure causing GC** - High garbage collection activity
5. **Crypto mining malware** - Unauthorized workloads

## Remediation Actions

### For Traffic Spikes
```bash
# Scale up the deployment
kubectl scale deployment/<deployment-name> --replicas=<count> -n <namespace>
```

### For Resource Limit Issues
```bash
# Update resource limits (requires deployment edit)
kubectl set resources deployment/<deployment-name> \
  --limits=cpu=500m,memory=512Mi \
  --requests=cpu=200m,memory=256Mi \
  -n <namespace>
```

### For Application Issues
```bash
# Rollback to previous version
kubectl rollout undo deployment/<deployment-name> -n <namespace>
```

### Emergency Pod Restart
```bash
# Delete the pod (will be recreated by deployment)
kubectl delete pod <pod-name> -n <namespace>
```

## Escalation Criteria

Escalate to on-call engineer if:
- CPU remains > 90% after remediation
- Multiple nodes affected
- Customer-facing impact detected
- Root cause unclear after 15 minutes

## Prevention

1. Set appropriate resource requests and limits
2. Implement horizontal pod autoscaling (HPA)
3. Add CPU-based alerts at 70% threshold
4. Regular load testing
5. Code review for CPU-intensive operations
