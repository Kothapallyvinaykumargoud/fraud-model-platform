#!/usr/bin/env bash
# One-time (and re-runnable) setup for the k3s cluster: namespace, RBAC,
# Prometheus, Grafana (provisioned from monitoring/grafana/), and the
# inference API. Run this from the repo root on the EC2 box (or against it
# via `KUBECONFIG=... kubectl` from your laptop).
#
# Usage:
#   IMAGE=ghcr.io/you/fraud-model-platform:latest ./k8s/deploy.sh
#
# After this initial apply, the build-and-deploy CI workflow handles
# redeploys via `kubectl set image` — this script is for first-time setup
# or manual recovery.
set -euo pipefail

IMAGE="${IMAGE:-IMAGE_PLACEHOLDER}"
NS=fraud-platform
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

kubectl apply -f "$ROOT/k8s/namespace.yaml"

kubectl apply -f "$ROOT/k8s/prometheus-rbac.yaml"
kubectl apply -f "$ROOT/k8s/prometheus-configmap.yaml"
kubectl apply -f "$ROOT/k8s/prometheus-deployment.yaml"

kubectl create configmap grafana-datasources \
  --from-file="$ROOT/monitoring/grafana/provisioning/datasources" \
  -n "$NS" --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboard-provisioning \
  --from-file="$ROOT/monitoring/grafana/provisioning/dashboards/dashboards.yml" \
  -n "$NS" --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboards \
  --from-file="$ROOT/monitoring/grafana/dashboards" \
  -n "$NS" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$ROOT/k8s/grafana-deployment.yaml"

sed "s|IMAGE_PLACEHOLDER|$IMAGE|" "$ROOT/k8s/inference-deployment.yaml" | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/inference-service.yaml"

echo "Deployed. Inference API: http://<node-ip>:30080  Prometheus: http://<node-ip>:30090  Grafana: http://<node-ip>:30030 (admin/changeme — change this)"
