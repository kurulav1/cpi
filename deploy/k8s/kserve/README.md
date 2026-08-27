# CPI under KServe (RawDeployment)

The production-shaped control plane: instead of hand-managing a Deployment +
Service + HPA, declare **one `InferenceService`** and let **KServe** build and own
the serving lifecycle around the CPI engine.

```
 InferenceService (cpi)                     KServe controller
   modelFormat: ll2c        ──────────────▶  matches runtime, builds:
   runtime: cpi-llama                          • Deployment  cpi-predictor
        │                                       • Service     cpi-predictor
        ▼                                       • HPA         cpi-predictor (1→3)
 ServingRuntime (cpi-llama)                      • route       http://cpi-default.example.com
   container: cpi-llama image (GPU)
   serves format "ll2c"
```

## Why this over the raw Deployment

The plain `kind-inference-gpu.yaml` + KEDA stack works, but you own every object.
KServe gives a **model-centric abstraction**: a `ServingRuntime` describes *how to
run an engine*, an `InferenceService` requests *a model*, and KServe builds the
Deployment/Service/HPA/route, canary rollouts, and a model-status lifecycle. CPI
plugs in as a **custom runtime** (its own image + the `ll2c` format) while KServe
owns orchestration. This is the path to model-registry-style multi-model serving.

## Files

| File | What |
|---|---|
| `cpi-servingruntime.yaml` | `ServingRuntime cpi-llama`; the CPI image as a serving container for the `ll2c` format (GPU via the WSL2 `/dev/dxg` + `/usr/lib/wsl` bind-mount, `ldconfig` before start) |
| `cpi-inferenceservice.yaml` | `InferenceService cpi`; requests an `ll2c` model on the `cpi-llama` runtime, RawDeployment, HPA 1→3 |

## Deploy

```bash
# Prereqs: cert-manager + KServe (RawDeployment mode):
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve-cluster-resources.yaml
kubectl patch configmap inferenceservice-config -n kserve --type merge \
  -p '{"data":{"deploy":"{\"defaultDeploymentMode\":\"RawDeployment\"}"}}'
kubectl -n kserve rollout restart deploy/kserve-controller-manager

# CPI:
kubectl apply -f deploy/k8s/kserve/cpi-servingruntime.yaml
kubectl apply -f deploy/k8s/kserve/cpi-inferenceservice.yaml
kubectl get inferenceservice cpi -w        # wait for READY=True
```

## Verified (single-node kind, RTX 5090)

```
InferenceService cpi  READY=True  url=http://cpi-default.example.com
  Active Model State: Loaded
KServe built: deployment/cpi-predictor 1/1, service/cpi-predictor, hpa 1→3
Routed cpi-predictor.default.svc → /v1/completions → coherent GPU output
```

## Notes & caveats

- **RawDeployment** (not Serverless) is used so we don't need Knative/Istio; right
  for kind, and gives a plain Deployment/Service/HPA you can inspect.
- **No `storageUri`**; the `cpi-llama` runtime mounts the model from the node
  (hostPath). On a real cluster set `storageUri: s3://…` (or `pvc://…`) and KServe's
  storage-initializer stages it to `/mnt/models`.
- **Autoscaling here is KServe's default CPU HPA.** That's the wrong signal for a
  single-stream LLM (see `../observability/`; queue depth is correct). To use the
  queue-depth signal under KServe, either annotate the ISVC for an external metric
  or keep KEDA scaling the `cpi-predictor` Deployment KServe creates.
- The GPU bind-mount block is a **WSL2/Docker-Desktop** workaround; on a real GPU
  node drop it and add `resources.limits.nvidia.com/gpu: 1` to the runtime container.
