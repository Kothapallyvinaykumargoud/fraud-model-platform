---
title: "Product Brief: Fraud Model Platform"
status: final
created: 2026-07-20
updated: 2026-07-20
---

# Product Brief: Fraud Model Platform

## Executive Summary

A personal, hands-on learning project to build the **MLOps/backend platform** that takes a fraud-detection model from "approved" to "serving real predictions" — and keeps it healthy over time. The learner plays both Data Scientist (training a minimal model) and Backend/MLOps Engineer (everything after that), but the project's center of gravity is the second role: validating and packaging a model, versioning it through MLflow, containerizing it, deploying it to a self-managed Kubernetes cluster on real AWS infrastructure, exposing it as an inference API, observing it in production with real metrics, detecting when the incoming data drifts from what it was trained on, and automatically retraining and redeploying when that happens — all on a free or near-free budget. The point is not a novel fraud algorithm; it's building and operating the full lifecycle around one, the way a backend/MLOps engineer would.

## Motivation & Learning Goals

The learner wants to learn cloud and platform engineering **by doing it for real** — not tutorials, not toy diagrams, but an actual AWS account, actual Kubernetes cluster, actual CI pipeline, and an actual closed loop that notices when the model is going stale and does something about it.

## What You're Building

Starting point: a trained, "approved" fraud-detection model, trained by the learner on Kaggle's Credit Card Fraud Detection (ULB) dataset, with a slice of that dataset deliberately held back to later simulate drift.

End-to-end flow:

1. **Validate Model** — a validation gate (metrics threshold check) before a model is allowed to be promoted.
2. **Package Model** — bundle the model artifact with versioned metadata.
3. **Register in MLflow** — track experiments and register model versions (run on-demand, not always-on — see Architecture & Constraints).
4. **Build Docker Image** — containerize a model-serving API around the current registered model.
5. **CI Pipeline (GitHub Actions)** — build and push the Docker image, trigger deployment.
6. **Deploy on Kubernetes (self-managed, AWS EC2)** — a single-node k3s/kubeadm cluster on a free-tier EC2 instance.
7. **Backend Inference API** — a REST endpoint that accepts a transaction payload and returns a fraud verdict.
8. **Monitoring & Logging** — Prometheus + Grafana on the same cluster: infra health (CPU/RAM/pod status), model-serving signals (latency, prediction class distribution), and drift signals (feature/prediction distribution vs. training baseline).
9. **Data Drift Detection & Automated Retraining** — held-back data is fed in over time (optionally with injected synthetic shifts) to simulate real-world drift; a drift check runs on-demand on the EC2 box, and when drift crosses a threshold it triggers retraining, validation, and re-registration in MLflow — which CI then picks up and redeploys.
10. **Rollback** — the ability to roll the deployment back to a specific previous *model version* (not just a previous container image) if a new one — auto-retrained or manual — misbehaves.
11. **Consumer** — a self-built client script that sends realistic transaction payloads to the API and prints the verdict, standing in for "Banking Applications."

## Scope

**In scope:** Everything in the end-to-end flow above (steps 1–11) — model validation, MLflow versioning, containerization, GitHub Actions CI, self-managed k3s on AWS, the inference API, Prometheus/Grafana monitoring, drift detection, automated retraining, version-aware rollback, and the client script.

**Out of scope:**
- Model research / accuracy optimization / feature engineering depth.
- Managed Kubernetes (EKS) — rejected specifically to avoid its ~$0.10/hr control-plane cost.
- Self-hosted Jenkins — rejected; resource budget can't fit it alongside k3s + monitoring on a 1GB box.
- Multi-node / high-availability Kubernetes — single node only.
- A live/real data feed — drift is simulated from held-back data, not sourced from a real bank.
- Real banking integration, compliance, or production security hardening (this is a learning project, not a production banking system).

## Architecture & Constraints

- **Compute**: One AWS free-tier EC2 instance (t2.micro/t3.micro — 1 vCPU, 1GB RAM), running a self-managed single-node Kubernetes cluster (k3s or kubeadm). Chosen specifically to avoid the EKS control-plane fee.
- **CI/CD**: GitHub Actions, not Jenkins — a deliberate tradeoff of Jenkins hands-on experience for a workable free-tier resource budget (Jenkins + k3s + monitoring would not fit in 1GB RAM together).
- **Model registry**: MLflow, run on-demand (started/stopped on the EC2 box only during registration) rather than as an always-on cluster service, to keep it off the steady-state RAM budget. The serving path depends on a model-version pointer, not on MLflow being continuously up.
- **Drift detection & retraining**: On-demand on the same EC2 box, sharing the same "spin up, do work, spin down" pattern as MLflow. This means MLflow and retraining both add load during their windows, but steady-state serving (k3s + inference API + monitoring) is unaffected since neither runs continuously.
- **Monitoring**: Prometheus + Grafana as pods on the same k3s cluster — the single tightest resource constraint in the design (see Risks).
- **Budget**: $0 target. AWS free tier is time-limited (typically 12 months per account for EC2 free-tier hours) — cost applies once that window closes.

## Success Criteria

- A transaction payload sent to the inference API returns a real fraud verdict from a model actually running on the self-managed k8s cluster — not a local dev server.
- A code/model change flows through the full pipeline unattended: GitHub Actions builds and pushes the image, the cluster deploys it, and the new version is what answers subsequent requests.
- Simulated drift is actually detected — the drift check flags a real distribution shift injected via held-back data, not a hardcoded/fake trigger — and that detection kicks off retraining, validation, and re-registration without the learner manually doing each step.
- Rolling back to a prior model version is a real, exercised action (done at least once on purpose, including after an auto-retrain the learner decides was bad), not just a theoretical capability.
- Grafana shows infra health, model-serving signals, and drift signals for at least one real run that includes a deliberate drift event.
- The learner can explain and defend every architectural tradeoff made here (EKS vs. self-managed, Jenkins vs. GitHub Actions, always-on vs. on-demand MLflow/retraining) — the tradeoffs are as much the learning outcome as the working system.

## Risks & Open Questions

- **RAM budget is unverified.** k3s + Prometheus + Grafana + the model-serving API running steady-state, plus MLflow and retraining sharing the box during their on-demand windows, is tight on 1GB. May require aggressive tuning (short metric retention, dropping Grafana in favor of raw Prometheus queries, trimming what's monitored, or ensuring retraining/MLflow never run at the same moment) once actually built. Flagged, not resolved.
- **Retraining on a 1GB box may simply be too slow or may OOM** even run in isolation, depending on dataset size and algorithm choice — may force a smaller held-back sample or a lighter model class than originally planned.
- **Free-tier window is unconfirmed.** Whether the AWS account is still within its 12-month free-tier period is unknown — affects whether "free" actually holds.
- **MLflow backend store** (SQLite/local vs. something else) not yet decided — affects how "on-demand" MLflow persists registry state between runs.
- **Drift detection method and threshold** not yet chosen (e.g. PSI, KS-test, or a simpler mean/variance shift check) — a downstream (architecture/PRD) decision, not resolved here.
- **Auto-retrain safety**: nothing yet prevents an auto-retrained model from being *worse* than the one it replaces beyond the validation gate — worth deciding whether promotion requires the new model to beat the old one on held-out metrics, or just clear a fixed bar.

## Vision

If this goes well, it becomes a portfolio piece that demonstrates real (not simulated) backend/MLOps competence: a working, observable, versioned, rollback-capable, self-healing model-serving platform built on real cloud infrastructure under a real cost constraint — the kind of tradeoff-driven engineering judgment and full-lifecycle thinking that a resume line can't show but a working system can.
