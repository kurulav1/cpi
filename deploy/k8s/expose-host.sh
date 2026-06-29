#!/usr/bin/env bash
# Publish in-cluster services to the Windows host via tiny socat proxy containers on
# the kind docker network. Survives Docker/session restarts (--restart unless-stopped)
# and reconnects per-connection (fork) — no kubectl port-forward to keep alive.
export PATH="$HOME/.local/bin:$PATH"
export DOCKER_CONFIG=/tmp/dockercfg; mkdir -p "$DOCKER_CONFIG"; echo '{}' > "$DOCKER_CONFIG/config.json"

CPI_NP=$(kubectl get svc cpi-kind -o jsonpath='{.spec.ports[0].nodePort}')
GRAF_NP=$(kubectl -n observability get svc grafana -o jsonpath='{.spec.ports[0].nodePort}')
echo "cpi-kind NodePort=$CPI_NP   grafana NodePort=$GRAF_NP"

run_proxy() {  # name hostport nodeport
  docker rm -f "$1" >/dev/null 2>&1
  docker run -d --restart unless-stopped --name "$1" --network kind -p "$2:$2" \
    alpine/socat "tcp-listen:$2,fork,reuseaddr" "tcp-connect:cpt-control-plane:$3" >/dev/null
  echo "  $1 rc=$?"
}
echo "=== start proxies ==="
run_proxy cpi-access     3001 "$CPI_NP"
run_proxy grafana-access 3300 "$GRAF_NP"   # 3000 is taken by a host Node process
sleep 3
docker ps --filter name=-access --format '{{.Names}}  {{.Status}}  {{.Ports}}'
echo "DONE"
