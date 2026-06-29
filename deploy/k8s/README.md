# CPI on Kubernetes — the serving half of the train + serve platform

This directory holds the **inference plane**. Its sibling, the **training plane**, lives
in the CPT repo under `deploy/k8s/` (the Indexed data-parallel Job). Together they form a
comprehensive setup: **CPT trains, CPI serves, and they meet on a shared model store.**

```
   TRAINING PLANE (CPT repo)         MODEL STORE (the seam)        SERVING PLANE (this repo)
 ┌──────────────────────┐        ┌──────────────────────┐      ┌──────────────────────────┐
 │ Indexed Job          │ write  │ RWX PVC / object store│ read │ Deployment (N replicas)  │
 │  gang of ranks       │ ─────► │   /models/<name>/     │ ───► │  web :3001 → llama_infer │
 │  gradient sync       │  ckpt  │     model.ll2c        │      │  Service + Ingress + HPA │
 └──────────────────────┘        │     tokenizer.json    │      └──────────────────────────┘
          │ on success                     ▲
          ▼                                │ write .ll2c
   ┌──────────────────────┐                │
   │ Export Job           │ ───────────────┘
   │ export_cpt_to_hf.py  │   (CPT) then  convert_hf_to_bins.py  (CPI)
   └──────────────────────┘
```

The two planes are **decoupled**: serving replicas never talk to the training Job and never
talk to each other — they only read the artifact. That is what lets training be a batch Job
and serving be an always-on autoscaled service.

## Why the shapes differ

| | Training (CPT) | Serving (CPI, here) |
|---|---|---|
| k8s object | `Job` (Indexed, run-to-completion) | `Deployment` + `Service` + `HPA` |
| Pods | gang of ranks, gradient-synced | independent replicas, stateless |
| Scaling | fixed rank count, all-or-nothing | autoscale on load |
| Storage | scratch + checkpoint **out** | model store **in** (read-only) |
| GPU | exclusive, long holds | always reserved, latency-sensitive |

## Files

- **`inference-deployment.yaml`** — production variant: namespace, RWX `models` PVC,
  Deployment (1 GPU/replica, rolling updates, liveness/readiness probes), Service,
  Ingress, HPA. Edit the `storageClassName`, Ingress host, and HPA metric for your cluster.
- **`kind-inference-deployment.yaml`** — local single-node (kind) variant: CPU-only, no GPU
  plugin, NodePort, `hostPath` model store. Mirrors the CPT `kind-data-parallel-job.yaml`.

## Local end-to-end loop (kind)

Run on the same kind cluster used for CPT training:

```bash
# 1. build + load the CPI serving image
docker build -t cpi-llama:test .
kind load docker-image cpi-llama:test --name cpt

# 2. provide a model on the node (any existing .ll2c works for a serving smoke test)
#    place model.ll2c + tokenizer.json under the node's /tmp/cpt-models

# 3. deploy + probe
kubectl apply -f deploy/k8s/kind-inference-deployment.yaml
kubectl get pods -l app=cpi-kind -w
kubectl port-forward svc/cpi-kind 3001:80
curl localhost:3001/api/health           # {"ok":true,"ready":...}
curl localhost:3001/v1/models            # OpenAI-compatible model list
```

## Production-grade serving (no cold start, survives restarts)

`kind-inference-gpu.yaml` is hardened for real serving; the gaps it closes:

- **Persistent model** — `model-pvc.yaml` (a `standard`/local-path PVC on the node's
  `/var` ext4) replaces the old `hostPath /tmp/cpt-models`. In kind, `/tmp` is **tmpfs
  (RAM)**, so every node/Docker restart wiped the 6.8 GB model and forced a re-seed.
  On the PVC it persists. Seed it **once** (stream into the PV dir on the node):
  ```bash
  kubectl apply -f deploy/k8s/model-pvc.yaml
  # provision + find the PV dir, then stream the model in via the node (robust for GBs):
  PV=$(kubectl get pv "$(kubectl get pvc cpi-models -o jsonpath='{.spec.volumeName}')" -o jsonpath='{.spec.local.path}')
  docker exec -i cpt-control-plane sh -c "cat > '$PV/model.ll2c'"    < model.ll2c
  docker exec -i cpt-control-plane sh -c "cat > '$PV/tokenizer.json'" < tokenizer.json
  ```
- **No cold start** — `LLAMA_WARM_ON_START=1` loads the model into the GPU **at boot**,
  and readiness is gated on the model actually being warm. The first real request is
  generation-time only (~2 s), never a ~60 s lazy load.
- **Model-aware probes** — the server exposes dedicated endpoints (don't gate liveness
  on the model, so a slow load can't trigger a restart loop):
  - `GET /healthz/live` → 200 once the process is up (**liveness**)
  - `GET /healthz/ready` → 200 only when the model is warm, else 503 (**readiness** +
    **startupProbe**, with a generous `failureThreshold` for the load window)
- **Zero-downtime + graceful** — `maxUnavailable: 0` rollout, a `preStop` drain, and
  `terminationGracePeriodSeconds` so rollouts/scale-down don't drop in-flight requests.
- **Right-sized resources** — requests are small (the model is in GPU VRAM, not host
  RAM), so replicas and the alternative KServe path co-schedule on the node.

Verified: after a full Docker Desktop restart the model stays in the PVC (no re-seed),
the pod auto-warms, and the first request is ~2.5 s — not a cold load.

## Probes (legacy note)

`/api/health` still returns **200 whenever the server is listening** with model state in
the body — fine as a simple liveness check. The `/healthz/*` endpoints above are the
production probes (status-code based, so plain `httpGet` works — no body-grep needed).

## GPU on kind (WSL2 / Docker Desktop)

`kind-inference-gpu.yaml` runs the pod on the GPU **without the NVIDIA device plugin**, which is
unreliable on Docker Desktop. Instead the GPU is injected the WSL-native way and bind-mounted into
the pod. Verified end-to-end on an RTX 5090 (sm_120) with the CUDA 12.4 build (the `90-virtual`
PTX JITs to Blackwell at load time).

1. **Make `nvidia` Docker's default runtime** so kind's node container inherits GPU access. Add to
   `~/.docker/daemon.json` (Docker Desktop → Settings → Docker Engine), then restart Docker:
   ```json
   { "default-runtime": "nvidia",
     "runtimes": { "nvidia": { "path": "nvidia-container-runtime" } } }
   ```
2. **Build a GPU-enabled kind node image** (`gpu-node.Dockerfile`) — it just sets
   `NVIDIA_VISIBLE_DEVICES=all`, which makes the runtime inject `/dev/dxg` + the WSL driver libs
   (`libcuda.so.1`, `libnvidia-ptxjitcompiler.so.1`) into the node, exactly like `docker run --gpus all`:
   ```bash
   docker build -t cpt-gpu-node:v1.32.0 -f deploy/k8s/gpu-node.Dockerfile deploy/k8s
   kind create cluster --name cpt --image cpt-gpu-node:v1.32.0
   docker exec cpt-control-plane sh -c 'ldconfig -p | grep libcuda'   # verify GPU in node
   ```
3. **Deploy the GPU pod.** `kind-inference-gpu.yaml` is `privileged`, bind-mounts the node's
   `/usr/lib/wsl` + `/dev/dxg`, and runs `ldconfig` over the WSL driver dirs before starting the
   server so `libcuda.so.1` resolves:
   ```bash
   kind load docker-image cpi-llama:test --name cpt
   kubectl apply -f deploy/k8s/kind-inference-gpu.yaml
   ```
   A `/v1/completions` call then returns coherent output GPU-accelerated (~12 s vs ~54 s on CPU).

On a real cluster none of this applies — use the NVIDIA device plugin and `nvidia.com/gpu` resource
requests (`inference-deployment.yaml`); the WSL dance is a Docker-Desktop-only workaround.

## The real integration gap

The orchestration here is plumbing. The substance is the **artifact round-trip** — a CPT-trained
checkpoint actually loading in `llama_infer`:

- ✅ Export path exists: CPT `export_cpt_to_hf.py` → CPI `convert_hf_to_bins.py` (HF → `.ll2c`).
- ⚠️ **Round-trip not yet verified** end-to-end. That is the one piece that is engineering, not YAML.

## GPU sharing

Training wants GPUs for long exclusive holds; serving wants them always free for latency. On a
shared cluster, resolve with **two node pools** (batch/preemptible for training, always-on for
serving) and separate namespaces with quotas. On a single GPU, **time-share**: serve by default,
train on a schedule.
