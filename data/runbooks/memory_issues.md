# Memory Issues Runbook

## Overview
This runbook covers diagnosis and remediation of memory-related incidents in Kubernetes clusters.

## Symptoms
- OOMKilled events in pod events
- Memory utilization approaching limits
- Swap usage on nodes
- Application slowdowns due to garbage collection

## Diagnosis Steps

### Step 1: Check pod memory usage
```bash
kubectl top pods -n <namespace> --sort-by=memory
```

### Step 2: Check for OOMKilled events
```bash
kubectl get events -n <namespace> --field-selector reason=OOMKilled
```

### Step 3: Review container memory limits
```bash
kubectl describe pod <pod-name> -n <namespace> | grep -A10 "Limits:"
```

### Step 4: Check node memory pressure
```bash
kubectl describe node <node-name> | grep -A5 "Conditions:"
```

## Common Root Causes

1. **Memory leak in application** - Gradual memory increase over time
2. **Insufficient memory limits** - Limits set too low for workload
3. **Traffic spike** - Sudden increase in concurrent requests
4. **Large data processing** - Batch jobs consuming excessive memory
5. **JVM heap misconfiguration** - Java applications with wrong heap settings

## Remediation Actions

### Increase Memory Limits
```bash
kubectl set resources deployment/<deployment-name> \
  --limits=memory=2Gi \
  --requests=memory=1Gi \
  -n <namespace>
```

### Restart Pods (Clear Memory)
```bash
kubectl rollout restart deployment/<deployment-name> -n <namespace>
```

### Scale Horizontally
```bash
kubectl scale deployment/<deployment-name> --replicas=<count> -n <namespace>
```

### For Java Applications (Update JVM Settings)
```bash
kubectl set env deployment/<deployment-name> \
  JAVA_OPTS="-Xmx1g -Xms512m" \
  -n <namespace>
```

## Escalation Criteria

Escalate to on-call engineer if:
- Memory usage doesn't decrease after remediation
- OOMKilled events continue after scaling
- Node-level memory pressure persists
- Application behavior indicates memory leak

## Prevention

1. Set appropriate memory requests and limits
2. Implement memory profiling in staging
3. Use vertical pod autoscaler (VPA) for recommendations
4. Monitor memory trends over time
5. Regular load testing with memory profiling
