# Pod CrashLoopBackOff Runbook

## Overview
This runbook covers diagnosis and remediation of pods stuck in CrashLoopBackOff state.

## Symptoms
- Pod status shows "CrashLoopBackOff"
- Rapid container restarts
- Increasing restart count
- Application unavailability

## Diagnosis Steps

### Step 1: Check pod status and events
```bash
kubectl describe pod <pod-name> -n <namespace>
```

### Step 2: Check container logs
```bash
# Current container logs
kubectl logs <pod-name> -n <namespace>

# Previous container logs (if crashed)
kubectl logs <pod-name> -n <namespace> --previous
```

### Step 3: Check resource constraints
```bash
kubectl describe pod <pod-name> -n <namespace> | grep -A10 "State:"
```

### Step 4: Verify ConfigMaps and Secrets
```bash
kubectl get configmap -n <namespace>
kubectl get secret -n <namespace>
```

## Common Root Causes

1. **Application error** - Code bug causing crash on startup
2. **Missing dependencies** - Required services unavailable
3. **Configuration error** - Invalid environment variables or config
4. **Resource exhaustion** - OOMKilled due to memory limits
5. **Failed health checks** - Liveness probe failing
6. **Permission issues** - Service account or RBAC problems
7. **Image pull errors** - Wrong image tag or registry issues

## Remediation Actions

### For OOMKilled
```bash
# Increase memory limits
kubectl set resources deployment/<deployment-name> \
  --limits=memory=1Gi \
  -n <namespace>
```

### For Configuration Issues
```bash
# Check and update ConfigMap
kubectl edit configmap <configmap-name> -n <namespace>

# Restart pods to pick up changes
kubectl rollout restart deployment/<deployment-name> -n <namespace>
```

### For Image Issues
```bash
# Verify image exists
docker pull <image-name>:<tag>

# Update image if needed
kubectl set image deployment/<deployment-name> \
  <container-name>=<new-image>:<tag> \
  -n <namespace>
```

### For Code Bugs
```bash
# Rollback to last known good version
kubectl rollout undo deployment/<deployment-name> -n <namespace>
```

### For Health Check Issues
```bash
# Temporarily disable liveness probe (emergency only)
kubectl patch deployment <deployment-name> -n <namespace> \
  --type='json' \
  -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/livenessProbe"}]'
```

## Exit Codes Reference

| Exit Code | Meaning | Common Cause |
|-----------|---------|--------------|
| 0 | Success | Container completed normally |
| 1 | Application error | General application failure |
| 137 | SIGKILL (OOMKilled) | Out of memory |
| 139 | SIGSEGV | Segmentation fault |
| 143 | SIGTERM | Graceful termination |

## Escalation Criteria

Escalate if:
- Root cause not identified within 10 minutes
- Multiple pods affected
- Critical service impacted
- Rollback doesn't resolve the issue

## Prevention

1. Implement proper health checks
2. Set appropriate resource limits
3. Use readiness probes before liveness
4. Test configuration changes in staging
5. Implement gradual rollouts
