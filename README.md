# Fraud Model Platform

A backend/MLOps learning project: validate → package → register (MLflow) →
containerize → CI (GitHub Actions) → deploy (self-managed k3s on an AWS
free-tier EC2 instance) → serve → monitor (Prometheus/Grafana) → detect
drift → auto-retrain → roll back. Planning docs (brief, PRD, decision logs)
live under `_bmad-output/planning-artifacts/`.

The model is deliberately simple — this project is about operating the
platform around it, not about fraud-detection accuracy.

## Repo layout

```
model/        training, validation gate, packaging, MLflow registration (FR-1..4)
serving/      FastAPI inference API + Dockerfile source (FR-5, FR-8)
mlops/        drift check, drift-triggered retraining, rollback (FR-13..16)
client/       stand-in for "Banking Applications" (FR-9)
k8s/          k3s manifests + deploy.sh (FR-7)
monitoring/   Grafana provisioning + dashboard (FR-10..12)
.github/      CI workflows (FR-6)
data/         put the Kaggle dataset here (gitignored)
models/       generated locally — model artifacts, pointer, logs (gitignored)
```

## 1. Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: download the real dataset (see `data/README.md`). Without it,
everything below runs on a synthetic stand-in — fine for exercising the
pipeline, not for a model worth deploying.

## 2. Get your first model registered

```bash
python -m model.pipeline
```

This trains, validates (FR-1), packages (FR-2), and registers a Model
Version in MLflow (FR-3), writing `production_pointer.json` and
`models/production/model.joblib` — the two files the serving API actually
reads (FR-4). Nothing here needs a running MLflow server; it's plain file
I/O, by design (see `model/register.py` docstring).

## 3. Run the API locally

```bash
uvicorn serving.app:app --reload
```

```bash
curl localhost:8000/health
python -m client.client --url http://localhost:8000 --count 10
python -m client.client --url http://localhost:8000 --count 10 --drifted
```

`/metrics` exposes Prometheus-format latency, prediction, and drift metrics.

## 4. Exercise drift detection + auto-retraining locally

```bash
python -m mlops.drift_check                 # against a normal holdout sample — should not trigger
python -m mlops.drift_check --simulate-drift # forces a high Drift Score
python -m mlops.retrain --simulate-drift     # drift check -> retrain -> validate -> promotion gate -> register
```

The promotion gate (FR-15) only lets a retrained candidate replace the
current production version if it passes validation **and** matches or
beats the current version's ROC-AUC — a regression doesn't get promoted
even if it clears the fixed bar.

## 5. Roll back

```bash
python -m mlops.rollback --list
python -m mlops.rollback --version 2
git add production_pointer.json models/production
git commit -m "Roll back to v2"
git push   # CI redeploys the rolled-back version
```

## 6. Build the image locally (sanity check before AWS)

```bash
docker build -t fraud-model-platform:test .
docker run -d --name fraud-test -p 8000:8000 fraud-model-platform:test
curl localhost:8000/health
docker rm -f fraud-test
```

## 7. Deploying to AWS (the part you're driving)

### 7.1 Launch the EC2 box

- Instance type: `t2.micro` or `t3.micro` (free-tier eligible — confirm your
  account is still within its 12-month window).
- Security group: allow inbound SSH (22), and NodePorts 30080 (API), 30090
  (Prometheus), 30030 (Grafana) from your IP.
- **Not EKS** — this project deliberately runs self-managed Kubernetes to
  avoid the ~$0.10/hr EKS control-plane charge.

### 7.2 Add swap (do this before installing k3s)

A 1GB box running k3s + Prometheus + Grafana + the API is genuinely tight —
this was flagged as an open risk in planning and is unverified until you
build it. Swap won't fix CPU pressure but gives k3s headroom to not get
OOM-killed during bursts:

```bash
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

If things still don't fit: drop Grafana first (query Prometheus directly),
then shorten Prometheus retention further, before considering a bigger
instance.

### 7.3 Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes   # confirms it's up
```

### 7.4 First deploy

From your laptop or the EC2 box itself (with this repo checked out and a
model already registered — step 2):

```bash
IMAGE=ghcr.io/<your-github-username-lowercase>/fraud-model-platform:manual ./k8s/deploy.sh
```

This sets up the namespace, RBAC, Prometheus, Grafana (provisioned from
`monitoring/grafana/`), and the inference API. For this very first run
you'll need an image already pushed (see 7.5) — after that, CI handles
rebuilds automatically.

Visit `http://<ec2-public-ip>:30080/health`, `:30090` (Prometheus),
`:30030` (Grafana, `admin` / the password you set in
`k8s/grafana-deployment.yaml` — **change it from `changeme`**).

### 7.5 Wire up CI (GitHub Actions)

The build half of `.github/workflows/build-and-deploy.yml` needs no setup —
it pushes to GHCR using the built-in `GITHUB_TOKEN` on every push to `main`.
Make the resulting package public in GitHub's package settings (Settings →
Packages) so your EC2 box can pull it without an image pull secret — fine
for a learning project with no sensitive code.

For the deploy half, add these repo secrets (Settings → Secrets and
variables → Actions):

| Secret | Value |
|---|---|
| `EC2_HOST` | EC2 public IP or DNS |
| `EC2_USER` | SSH user (`ubuntu`, `ec2-user`, etc.) |
| `EC2_SSH_KEY` | private key with access to the instance |

Without these, `build-and-deploy.yml` still builds and pushes the image —
it just skips the deploy step rather than failing.

### 7.6 Drift + retraining on the EC2 box

This runs **on the box**, on-demand, not in GitHub Actions (deliberate —
see `mlops/retrain.py` docstring: keeps it off the RAM budget and off CI
minutes). Set up a periodic check with cron:

```bash
# crontab -e, on the EC2 box, in the repo directory:
0 */6 * * * cd /path/to/fraud-model-platform && .venv/bin/python -m mlops.retrain >> /var/log/fraud-retrain.log 2>&1 && git add production_pointer.json models/production models/registration_log.jsonl && git commit -m "auto-retrain" && git push || true
```

You'll need the EC2 box to have push access to your repo (a deploy key or
a PAT) for the last step. When a retrain gets promoted and pushed,
`build-and-deploy.yml` picks it up like any other push.

## Open items (see PRD §8 for the full list)

A few things are deliberately left for you to decide/verify once this is
actually running, not resolved in code:

- Whether Prometheus + Grafana + the API actually fit in 1GB steady-state.
- The drift threshold (`0.25` PSI default in `mlops/drift_check.py`) —
  tune once you see real Drift Scores.
- Whether your AWS account is still within its free-tier window.

## Planning docs

- Brief: `_bmad-output/planning-artifacts/briefs/brief-fraud-model-platform-2026-07-20/brief.md`
- PRD: `_bmad-output/planning-artifacts/prds/prd-fraud-model-platform-2026-07-20/prd.md`
