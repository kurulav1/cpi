# Accessing CPI

CPI runs in the local kind cluster and is published to the host by small proxy
containers (`deploy/k8s/expose-host.sh`), so it's reachable at a stable URL with no
`kubectl port-forward` to keep alive (the proxies use `--restart unless-stopped`, so
they come back after a Docker/host restart — and the model is persistent + warmed, so
the API is ready without a re-seed or cold start).

| What | URL |
|---|---|
| **CPI Chat Studio** (web UI) | http://localhost:3001 |
| **OpenAI-compatible API** | http://localhost:3001/v1 |
| **Health (readiness)** | http://localhost:3001/healthz/ready |
| **Grafana** (serving metrics dashboard) | http://localhost:3300/d/cpi-serving |

> Same machine → use `localhost`. Another machine on the LAN → the proxies bind
> `0.0.0.0`, so use `http://<this-host-ip>:3001` (open the firewall for 3001).
> Note: **no auth** — keep it on trusted networks / behind a proxy if exposed.

## Using the API (it's the OpenAI API; model id is `model`)

```bash
curl http://localhost:3001/v1/models

curl -X POST http://localhost:3001/v1/completions \
  -H "content-type: application/json" \
  -d '{"model":"model","prompt":"def add(a, b):","max_tokens":64,"temperature":0}'

curl -X POST http://localhost:3001/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:3001/v1", api_key="not-checked")
print(client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "Hello"}],
).choices[0].message.content)
```

There's also a streaming endpoint `POST /api/chat/stream` and the web Chat UI at `/`.

## (Re)establishing host access

The proxies are created once and auto-restart. If you ever need to recreate them
(e.g. after recreating the cluster, which changes NodePorts):

```bash
bash deploy/k8s/expose-host.sh     # run from WSL; reads NodePorts, (re)starts proxies
```

To check / stop them:

```bash
docker ps  --filter name=-access
docker rm -f cpi-access grafana-access      # stop publishing
```

## How it works (kind specific)

This kind cluster was created without `extraPortMappings`, so NodePorts aren't on the
host directly. `expose-host.sh` runs a `socat` container per service on the `kind`
docker network, publishing a host port and forwarding to the node's NodePort
(`host:3001 → cpt-control-plane:<cpi-nodeport>`). For a from-scratch native ingress on
`:80`, recreate the cluster with `extraPortMappings` + ingress-nginx instead.
