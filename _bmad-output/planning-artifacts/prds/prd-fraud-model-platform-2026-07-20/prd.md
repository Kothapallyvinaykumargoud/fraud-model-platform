---
title: Fraud Model Platform
status: final
created: 2026-07-20
updated: 2026-07-21
---

# PRD: Fraud Model Platform

## 0. Document Purpose

This PRD turns the product brief (`briefs/brief-fraud-model-platform-2026-07-20/brief.md`) into concrete, testable requirements for a solo learner building a backend/MLOps platform around a fraud-detection model. It assumes the brief's architecture decisions as given (self-managed k3s on an AWS free-tier EC2 instance, GitHub Actions over Jenkins, on-demand MLflow, on-demand drift detection/retraining) and does not re-litigate them. Features are grouped; functional requirements (FRs) are nested under each and numbered globally. `[ASSUMPTION]` tags mark places this draft inferred without confirmation — see §9.

## 1. Vision

A single learner operates a small but real MLOps platform: a fraud-detection model is validated, versioned, containerized, deployed to a self-managed Kubernetes cluster, and served over a REST API — then watched. When the data it sees drifts from what it was trained on, the platform notices, retrains, and redeploys on its own, and the learner can roll any of it back to a known-good version on demand. Every piece runs on real cloud infrastructure under a real cost constraint. Success is a working, observable, self-healing system the learner built and can explain end to end — not a novel fraud-detection algorithm.

## 2. Target User

### 2.1 Primary Persona

The learner, acting in two roles: briefly as Data Scientist (trains one simple model to have something real to operate on) and primarily as Backend/MLOps Engineer (owns validation, packaging, deployment, serving, observability, drift response, and rollback). The platform is built for this one operator; there is no separate end-user audience.

### 2.2 Jobs To Be Done

- Practice real backend and MLOps engineering — containers, orchestration, CI/CD, observability, versioning — hands-on, on real (not simulated) cloud infrastructure.
- See a model actually serve predictions on real requests, not a notebook demo.
- Build and exercise a closed loop that detects drift and responds to it automatically.
- End up with a portfolio-worthy system whose tradeoffs the learner can defend.

### 2.3 Key User Journeys

*Single-operator, hobby-scale — lightweight form per scope dial.*

- **UJ-1.** The learner sends a transaction payload to the deployed inference API and gets back a real fraud verdict from the model currently running on the cluster.
- **UJ-2.** The learner feeds in simulated drifted data, watches the drift check fire, watches automated retraining and validation run, and sees a new model version registered and serving — without performing each step by hand.
- **UJ-3.** The learner deliberately rolls the deployment back to a specific earlier model version (e.g., after judging an auto-retrained version worse) and confirms the prior version is what now answers requests.

## 3. Glossary

- **Model Version** — A single trained, packaged model artifact registered in MLflow with metadata (training date, dataset slice, metrics).
- **Production Pointer** — The record of which Model Version is currently deployed; the serving path reads this without requiring MLflow to be running.
- **Validation Gate** — The metric-threshold check a Model Version must pass before it can be registered/promoted.
- **Drift Score** — A statistical measure of how far incoming data has moved from the training data distribution.
- **On-Demand Service** — A component (MLflow, the drift/retraining job) that runs only when triggered, not continuously, to fit the RAM budget.
- **Fraud Verdict** — The inference API's response to a Transaction Payload: a fraud/not-fraud classification plus a confidence score.
- **Transaction Payload** — The input record sent to the inference API, structured like a single transaction.

## 4. Features

### 4.1 Model Validation & Packaging

**Description:** Before a model can be served, it must pass a fixed quality bar and be packaged with enough metadata to trace it later.

#### FR-1: Validation Gate
A candidate model can be validated against a fixed metric threshold before promotion.
**Consequences (testable):**
- A model scoring below the threshold is not registered or deployed.
- Every validation attempt (pass or fail) is logged with its metrics.

#### FR-2: Model Packaging
A validated model is packaged with versioned metadata (training date, dataset slice used, validation metrics).
**Consequences (testable):**
- A packaged artifact carries its metadata alongside it, retrievable without re-running validation.
- Two packaged artifacts from different training runs are distinguishable by their metadata alone.

### 4.2 Model Registry

**Description:** MLflow tracks and registers Model Versions as an On-Demand Service rather than running continuously.

#### FR-3: Model Registration
A validated, packaged model is registered as a new Model Version in MLflow.
**Consequences (testable):**
- After registration, the new Model Version appears in MLflow with a unique, incrementing version identifier.
- Registration succeeds by starting MLflow on-demand rather than requiring it to already be running.

#### FR-4: Production Pointer
The system maintains a Production Pointer identifying the current serving Model Version, readable by the serving path without MLflow running.
**Consequences (testable):**
- The serving path resolves the Production Pointer to a specific Model Version ID with MLflow stopped.
- Updating the Production Pointer does not require starting MLflow.

### 4.3 Build & Deploy

**Description:** A registered Model Version becomes a running service through an unattended CI/CD path.

#### FR-5: Containerization
A Docker image is built that serves the Model Version identified by the Production Pointer.
**Consequences (testable):**
- A running container built from the image serves predictions from exactly the Model Version the Production Pointer named at build time.
- The image builds successfully from a clean checkout with no manual local setup steps.

#### FR-6: CI Pipeline
On a new Model Version being registered or a code change being pushed, GitHub Actions builds, tests, and pushes the Docker image, then triggers deployment.
**Consequences (testable):**
- A push or new registration results in a new image in the registry without manual build steps.

#### FR-7: Deployment
CI triggers deployment of the new image to the self-managed k3s cluster; the running pod set reflects the latest deployed image within a bounded time after trigger.
**Consequences (testable):**
- Within a defined window after CI triggers deployment (`[ASSUMPTION: target not set — propose within 5 minutes]`), the inference API's reported Model Version matches the newly deployed image.
- If the new pod fails to become healthy, the previous pod set continues serving rather than both being down simultaneously.

### 4.4 Inference API

**Description:** The platform's single external surface — realistic transaction traffic in, a Fraud Verdict out.

#### FR-8: Prediction Endpoint
A client can submit a Transaction Payload and receive a Fraud Verdict via REST. `[ASSUMPTION: response includes verdict, confidence score, and the serving Model Version ID — exact payload schema not yet confirmed]`
**Consequences (testable):**
- A well-formed Transaction Payload returns a Fraud Verdict within a bounded response time. `[ASSUMPTION: target not yet set — propose <1s for a single prediction on this hardware]`
- A malformed payload returns a structured error, not a crash.

#### FR-9: Client Script
A self-built script sends realistic (and, for drift testing, deliberately shifted) Transaction Payloads to the endpoint and prints the Fraud Verdict received — standing in for "Banking Applications."
**Consequences (testable):**
- The script can send a batch of Transaction Payloads unattended and print a Fraud Verdict for each.
- The script can run in a "drifted" mode that sends payloads sourced from the held-back, shifted data.

### 4.5 Observability

**Description:** Prometheus + Grafana on the same cluster, covering infrastructure, model-serving, and drift signals.

#### FR-10: Infrastructure Monitoring
Prometheus scrapes CPU, RAM, and pod-health metrics from the cluster; Grafana visualizes them.
**Consequences (testable):**
- A Grafana dashboard shows current CPU/RAM usage and pod health, updated within one scrape interval of a real change (e.g., a pod restart is reflected).

#### FR-11: Model-Serving Monitoring
Prometheus captures prediction latency and predicted-class distribution; visible in Grafana.
**Consequences (testable):**
- After the Client Script (FR-9) sends a batch of payloads, Grafana shows updated latency and predicted-class distribution for that batch.

#### FR-12: Drift Signal Monitoring
The current Drift Score and its threshold are visible in Grafana alongside the model-serving signals. `[ASSUMPTION: drift statistic method — e.g. PSI or KS-test — not yet chosen]`
**Consequences (testable):**
- After a Drift Check (FR-13) runs, its resulting Drift Score and the threshold it was compared against are both visible in Grafana, not just in a log.

### 4.6 Drift Detection & Automated Retraining

**Description:** An On-Demand Service that forms a closed loop: it notices when incoming data no longer resembles the training data and reacts without manual intervention.

#### FR-13: Drift Check
An on-demand job compares incoming data (held-back or synthetically shifted) against the training baseline and computes a Drift Score.
**Consequences (testable):**
- Run against unshifted held-back data, the Drift Check produces a low Drift Score and does not trigger FR-14.
- Run against deliberately shifted data, the Drift Check produces a materially higher Drift Score.

#### FR-14: Drift-Triggered Retraining
When the Drift Score crosses its threshold, retraining runs automatically on-demand, producing a new candidate model.
**Consequences (testable):**
- A Drift Score below threshold does not start a retraining run.
- A Drift Score above threshold starts a retraining run without manual action, producing a candidate that then passes through FR-15.

#### FR-15: Auto-Retrain Promotion Gate
A candidate produced by FR-14 must pass the Validation Gate (FR-1) before registration. `[ASSUMPTION: promotion additionally requires the candidate to match or exceed the current production Model Version's held-out metric, not merely clear the fixed validation bar — resolves the brief's flagged "auto-retrain safety" risk; not yet confirmed by the learner]`
**Consequences (testable):**
- A retrained candidate that fails FR-1 or the regression check is not registered and does not replace the Production Pointer.

### 4.7 Rollback

**Description:** Any deployed Model Version, including auto-retrained ones, can be reverted to a specific earlier version.

#### FR-16: Version-Aware Rollback
The operator can point the Production Pointer at a specific prior Model Version, and the deployment redeploys that version.
**Consequences (testable):**
- After rollback, the inference API's Fraud Verdicts come from the specified prior Model Version, confirmed via the version ID in the API response (FR-8).

### 4.8 Data Sourcing & Feature Store

**Description:** Where training data comes from, and one shared place the transform from raw columns to model-ready features lives — reused by training, retraining, and serving instead of being duplicated in each. Added 2026-07-21, reconciling a request to source real, bank-like transaction data against the platform's free-tier cost model (see decision log).

#### FR-17: Real Feature Dataset
Training uses a small, stratified sample of the IEEE-CIS Fraud Detection dataset (real Vesta e-commerce transactions with named features — card network/type, product code, purchaser email domain, device type, transaction amount) in place of the earlier anonymized-PCA Kaggle ULB dataset.
**Consequences (testable):**
- Absence of the real sample fails loudly with instructions (mlops/download_ieee_cis.py), rather than silently substituting synthetic data without saying so.
- Every model input is traceable to a human-readable column name — no anonymized PCA components.

#### FR-18: Lightweight Feature Store
Feature transformation logic (numeric columns, categorical vocabulary) lives in one versioned, file-based module reused by training and serving.
**Consequences (testable):**
- A feature definition change updates every consumer from one place; nothing recomputes its own transform independently.
- The feature set carries its own schema/version (feature_definitions.json), retrievable and inspectable independent of any single model run, and travels with a model artifact through packaging, registration, shadow promotion, and rollback alike.

### 4.9 Explainability

**Description:** A validation-time report of what the model weighs, added alongside the existing metric-threshold gate. Added 2026-07-21 (see decision log).

#### FR-19: Feature Attribution
A SHAP-based feature-attribution summary is produced once per training run, not per live request.
**Consequences (testable):**
- A candidate model's package includes a ranked list of its top contributing features and their relative weight.
- Computing it adds no latency or CPU cost to the `/predict` path — the 1GB RAM budget has no room for per-request SHAP.

## 5. Non-Goals (Explicit)

- Not building a novel or highly accurate fraud-detection model — model quality is a means, not the goal.
- Not using managed Kubernetes (EKS), multi-AZ high availability, self-hosted Jenkins, Airflow, Datadog/ELK, or IAM/KMS/VPC hardening — ruled out on cost/RAM grounds; these appear in enterprise MLOps patterns referenced for inspiration (2026-07-21) but stay out of scope here.
- Not multi-node or highly-available Kubernetes — single node only.
- Not integrating with an actual bank or live transaction feed — a small, sampled slice of a real public dataset (IEEE-CIS, FR-17) stands in for bank data.
- Not building a production-grade feature platform (e.g. SageMaker Feature Store) — a lightweight, file/local versioned feature store only (FR-18).
- Not implementing production security hardening, compliance controls, or authentication on the API. `[ASSUMPTION: no auth layer in scope — flag if this should change]`
- Not building alerting (paging, on-call) beyond Grafana dashboards. `[ASSUMPTION]`

## 6. MVP Scope

### 6.1 In Scope
FR-1 through FR-19 (see §4).

### 6.2 Out of Scope for MVP
- Multi-model support (serving more than one fraud model concurrently).
- Canary or A/B deployment strategies — new versions replace the current one directly.
- (See also §5 Non-Goals for project-wide exclusions — auth, alerting, HA — that apply beyond MVP.)

## 7. Success Metrics

**Primary**
- **SM-1**: A code or model change reaches serving unattended — GitHub Actions builds, deploys, and the new Model Version answers subsequent requests — with zero manual steps between push/registration and serving. Validates FR-6, FR-7.
- **SM-2**: Simulated drift is detected and triggers retraining, validation, and re-registration without manual intervention, at least once in a deliberate test. Validates FR-13, FR-14, FR-15.

**Secondary**
- **SM-3**: Rollback to a specific prior Model Version is exercised at least once and confirmed via the API's reported version. Validates FR-16.

**Counter-metrics (do not optimize)**
- **SM-C1**: Model predictive accuracy is not a target to chase — time spent tuning the model instead of the platform is time misspent for this project's goal. Counterbalances SM-1/SM-2.

## 8. Open Questions

1. Which drift-detection statistic and threshold (PSI, KS-test, or a simpler mean/variance shift check)? — carried from the brief, unresolved.
2. What backend store does the on-demand MLflow instance use (SQLite/local vs. something else) to persist registry state between runs?
3. Does k3s + Prometheus + Grafana + the inference API actually fit in 1GB RAM steady-state, and does retraining fit when it shares the box on-demand? — unverified until built.
4. Is the AWS account still within its 12-month free-tier window?
5. Exact inference API request/response schema (see FR-8 assumption).
6. IEEE-CIS is a Kaggle *competition* dataset (FR-17) — downloading it needs a Kaggle account that has accepted the competition rules on kaggle.com, not just an API token. Unresolved until the learner has actually done that once; mlops/download_ieee_cis.py fails loudly with instructions until then.

## 9. Assumptions Index

- §4.4 FR-8 — Response schema (verdict + confidence score + serving Model Version ID) and response-time target (<1s) not yet confirmed.
- §4.5 FR-12 — Drift statistic method not yet chosen.
- §4.6 FR-15 — Auto-retrain promotion requires beating the current production model's held-out metric, not just clearing the fixed validation bar — a proposed resolution to the brief's flagged risk, not yet confirmed.
- §4.8 FR-17 — Sample size (50,000 rows, stratified to preserve IEEE-CIS's real ~3.5% fraud rate) is implemented as a default in mlops/download_ieee_cis.py, not separately confirmed with the learner beyond this PRD update.
- §5 — No API authentication layer and no alerting beyond dashboards assumed in scope for MVP.
