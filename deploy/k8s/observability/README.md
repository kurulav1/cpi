# Distributed CPI serving -- observability + autoscaling

This adds the **observe → autoscale** half of the serving plane on top of the GPU
inference pod (`../kind-inference-gpu.yaml`): high-scale metrics with
**VictoriaMetrics** and demand-driven replica autoscaling with **KEDA**.

```
                 load
                  │
                  ▼
        ┌───────────────────┐    /metrics     ┌──────────┐   remote_write   ┌────────────┐
        │ CPI pods (N)      │ ◀────scrape──────│ vmagent  │ ───────────────▶ │ vmsingle   │
        │  web :3001        │                  │ (k8s_sd) │                  │ TSDB+vmui  │
        │  /metrics         │                  └──────────┘                  │  :8428     │
        └───────────────────┘                                               └─────┬──────┘
                  ▲  scale 1..N                                                    │ PromQL
                  │                                                                │ sum(rate(...))
                  │            ┌──────────────────────────────────────────────────┘
            ┌─────┴──────┐     ▼
            │ HPA (KEDA) │ ◀── KEDA operator (ScaledObject → external metric)
            └────────────┘
```

## Components

| File | What |
|---|---|
| CPI `web/server/metrics.mjs` + `/metrics` | dependency-free Prometheus endpoint: `cpi_http_requests_total`, `cpi_inflight_requests`, `cpi_generated_tokens_total`, `cpi_generation_decode_ms_sum`, `cpi_model_ready` |
| `victoriametrics.yaml` | `vmsingle` (TSDB + MetricsQL + vmui at :8428) and `vmagent` (k8s pod service-discovery, scrapes `app=cpi-kind` pods every 5s) + RBAC |
| `keda-autoscale.yaml` | KEDA `ScaledObject` scaling the `cpi-kind` Deployment on a VictoriaMetrics query |
| `grafana.yaml` | Grafana with the VictoriaMetrics datasource + a CPI dashboard, both provisioned from ConfigMaps |

## Why VictoriaMetrics

Single binary, very high ingestion/cardinality per core, and a drop-in
Prometheus-compatible scrape + PromQL/MetricsQL surface -- so it scales to a real
fleet without the Prometheus HA/sharding dance. Here it runs as `vmsingle`; on a
real cluster use the VictoriaMetrics Operator (`VMSingle`/`VMAgent`/`VMServiceScrape`
CRDs) instead of the static scrape config.

## The autoscaling signal (important -- and counter-intuitive)

CPI is **single-stream**: it generates one request at a time (no continuous
batching). The first guess is to scale on a 409-rejection rate -- but CPI's
`/v1/completions` does **not** fast-reject concurrent requests; it **waits-for-idle**
and lets them **queue** (measured: 8 concurrent generations → `ok=45, busy409=0`).

That means **in-flight request count == queue depth == the true saturation signal**:
1 generating + N waiting. So the primary trigger is:

```promql
sum(cpi_inflight_requests)            # threshold 1 per replica
```

When more than one request is outstanding, a single-stream replica is already
behind. Two secondary triggers back it up: the **409 rate** (only appears once
waits *time out* under extreme overload -- a hard "add capacity now" signal) and the
**request rate** (catches bursty fast traffic like `/v1/models`). KEDA scales on the
max desired-replicas across all three.

> If/when CPI gains **continuous batching**, a replica serves many concurrent
> requests, so raise the per-replica `inflight` threshold (e.g. to the batch size)
> instead of 1.

## Deploy

```bash
kubectl apply -f deploy/k8s/observability/victoriametrics.yaml
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.16.1/keda-2.16.1.yaml
kubectl apply -f deploy/k8s/observability/keda-autoscale.yaml
kubectl apply -f deploy/k8s/observability/grafana.yaml

# Grafana (anonymous admin) -- CPI Serving dashboard:
kubectl -n observability port-forward svc/grafana 3000:3000
#   http://localhost:3000/d/cpi-serving/cpi-serving
# vmui / raw PromQL:
kubectl -n observability port-forward svc/vmsingle 8428:8428
#   http://localhost:8428/vmui   →  sum(cpi_inflight_requests)
```

The Grafana datasource and the **CPI Serving** dashboard (request rate by status,
409 backpressure, tokens/s, in-flight, avg latency, replica count) are provisioned
from ConfigMaps -- no manual setup.

## Verified end-to-end (single-node kind, RTX 5090)

Driving 8 concurrent generations at one replica, the queue-depth loop fired as designed:

```
idle:            inflight 0,  hpa 0/1,     pods=1
queue builds:    inflight 8,  hpa 8/1   →  pods=3   (KEDA scaled 1→3 on queue depth)
3 replicas:                   hpa 2667m/1            (8 in-flight ÷ 3 = 2.67/replica)
load ends:       inflight 0            →   pods=1   (KEDA scales back after cooldown)
```

The new replicas schedule and share the GPU because these pods don't request a
finite `nvidia.com/gpu` resource (they bind-mount `/dev/dxg`); each loads the model
onto the shared 5090 and becomes Ready.

## Honest caveats (this is kind, not a real cluster)

- **Single node, shared GPU.** Real scale-out wants a GPU node pool with the NVIDIA
  device plugin and `nvidia.com/gpu` requests, so replicas land on distinct GPUs.
- **`vmsingle` + `emptyDir`** -- no HA, no persistence. Use the VM Operator + PVs.
- **Per-replica throughput is the real ceiling** -- without continuous batching you
  scale out instead of batching up, which is less GPU-efficient. That engine work
  (continuous batching, then a network NCCL backend for sharded models) is the
  prerequisite for *efficient* large-scale serving; this stack is the control plane
  around it.
