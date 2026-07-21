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
  account is still within its 12-month window). **Confirmed by building
  this**: a 1GB `t2.micro`/`t3.micro` is not enough — k3s's own control
  plane alone uses ~370Mi, and CoreDNS + metrics-server + Traefik (bundled
  into k3s by default) + Prometheus + Grafana + the API push it into a
  crash-restart loop even with swap. Use **`t3.small` (2GB)** instead —
  it's outside the free tier (~$0.0208/hr, ~$15/month if left running
  24/7, ~$0 if you stop the instance between sessions).
- Security group: allow inbound **SSH (22) from Anywhere-IPv4
  (`0.0.0.0/0`)**, not just "My IP" — GitHub Actions' runners connect from
  their own cloud IPs, not yours, so restricting SSH to your IP silently
  breaks CI's deploy step. Key-based auth only, so this is standard
  practice, not a real exposure. Also allow NodePorts 30080 (API), 30090
  (Prometheus), 30030 (Grafana) — from your IP is fine for these three.
- **Not EKS** — this project deliberately runs self-managed Kubernetes to
  avoid the ~$0.10/hr EKS control-plane charge.
- Amazon Linux doesn't ship with `git` — `sudo dnf install -y git` before
  cloning this repo onto the instance.

### 7.2 Add swap

Cheap insurance even on 2GB — doesn't fix CPU pressure, but gives k3s
headroom during bursts (e.g. a rolling deployment briefly running two pod
versions at once):

```bash
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

If you're stuck on a 1GB instance and things still don't fit after this:
drop Grafana first (query Prometheus directly), then shorten Prometheus
retention further — but honestly, resizing to `t3.small` is less fighting
than tuning around a genuinely undersized box.

### 7.3 Install k3s and set up kubectl access

```bash
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(whoami):$(whoami) ~/.kube/config
export KUBECONFIG=~/.kube/config   # add to ~/.bashrc to persist across sessions
kubectl get nodes   # confirms it's up, status should be Ready
```

### 7.4 First deploy

From the EC2 box, with this repo cloned and a model already registered
(step 2 — run `python -m model.pipeline` there too, or `scp` your local
`production_pointer.json` + `models/production/` + `models/packages/`
over):

```bash
IMAGE=ghcr.io/<your-github-username-lowercase>/fraud-model-platform:manual ./k8s/deploy.sh
```

This sets up the namespace, RBAC, Prometheus, Grafana (provisioned from
`monitoring/grafana/`), and the inference API. For this very first run
you'll need an image already pushed (see 7.5) — after that, CI handles
rebuilds automatically. Check `kubectl get pods -n fraud-platform` — all
three should reach `1/1 Running` within a minute or so.

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
minutes). Confirmed working live, end to end, on AWS.

**Setup (what's actually running):**

1. Python 3.11 + venv on the box: `sudo dnf install -y python3.11 python3.11-pip cronie`
2. The real dataset transferred **privately via `scp`**, not through the public repo:
   `scp -i your-key.pem data/creditcard.csv ec2-user@<ip>:~/fraud-model-platform/data/`
   (same redistribution-terms reasoning as §"Ship the trained model" — the model can ship publicly, the raw dataset shouldn't.)
3. `python -m model.train` once, to produce `models/candidate/holdout.csv` (the drift-check baseline) — doesn't touch the deployed production pointer.
4. A dedicated **SSH deploy key** (not your personal key) so the box can push on its own:
   ```bash
   ssh-keygen -t ed25519 -C "fraud-platform-ec2-deploy-key" -f ~/.ssh/github_deploy_key -N ""
   # add ~/.ssh/github_deploy_key.pub as a repo Deploy Key with write access (repo Settings -> Deploy keys, or `gh repo deploy-key add`)
   ```
   Then point git at it (`~/.ssh/config` with an `IdentityFile` entry for `github.com`) and `git remote set-url origin git@github.com:...` (SSH, not HTTPS).
5. Cron:
   ```bash
   # crontab -e
   0 */6 * * * cd /home/ec2-user/fraud-model-platform && .venv/bin/python -m mlops.retrain >> /home/ec2-user/fraud-retrain.log 2>&1 && git add production_pointer.json models/production models/packages models/registration_log.jsonl models/validation_log.jsonl && git commit -m "auto-retrain: drift-triggered promotion" && git push >> /home/ec2-user/fraud-retrain.log 2>&1 || true
   0 3 * * 0 sudo k3s crictl rmi --prune >> /home/ec2-user/image-prune.log 2>&1
   ```
   Note `models/packages` in the `git add` — `mlops/package.py` creates a **new** directory every retrain, and `mlops/rollback.py` needs it there to have something to roll back to later. Easy to miss (we did, the first time).

**Lesson learned running this for real:** every promotion triggers a new CI build + deploy, which pulls a new ~170MB image onto the node. The default 8GB EBS root volume fills up fast under repeated deploys — we hit real `DiskPressure` (not the RAM issue from §7.1) after three deploys in quick succession, which cascaded into mass pod evictions across the whole namespace. Fixed by resizing to 20GB (`EC2 Console -> Elastic Block Store -> Volumes -> Modify volume`, then on the box: `sudo growpart /dev/nvme0n1 1 && sudo xfs_growfs -d /` — no reboot needed) and adding the weekly `crictl rmi --prune` cron line above. If you skip the resize, expect this to recur within a handful of retrain cycles.

**Also learned:** floating-point non-determinism. `RandomForestClassifier(n_jobs=-1)` doesn't produce bit-identical metrics across separate runs even with the same `random_state` — two models trained on identical data differed by ~2e-6 ROC-AUC purely from parallel tree-building order. The promotion gate in `mlops/retrain.py` uses a `0.005` tolerance, not a strict `<`, specifically because of this — a strict inequality would reject nearly every retrain as a "regression" that wasn't real.

## Open items (see PRD §8 for the full list)

Resolved by actually deploying this:

- ~~Whether Prometheus + Grafana + the API fit in 1GB steady-state~~ — they
  don't, reliably. Use `t3.small` (see §7.1).
- ~~Whether the automated drift-triggered retraining loop works live on
  AWS~~ — yes, confirmed end to end (§7.6): real drift detected on real
  data, real retrain, real promotion, real `git push` from the box's own
  deploy key, real CI redeploy.

Still open, deliberately left for you to decide once this is running:

- The drift threshold (`0.25` PSI default in `mlops/drift_check.py`) —
  tune once you see real Drift Scores in Grafana over time.
- Whether your AWS account is still within its free-tier window (moot for
  the EC2 box once you're on `t3.small`, which was never free-tier — but
  still relevant to other AWS usage, including EBS beyond the 30GB/month
  free allowance).
- No Elastic IP set up yet — the public IP changes every time the
  instance stops/starts, which breaks the `EC2_HOST` GitHub secret and
  any bookmarks until you update it. Recommended: `EC2 Console -> Elastic
  IPs -> Allocate -> Associate` with the instance.

## Planning docs

- Brief: `_bmad-output/planning-artifacts/briefs/brief-fraud-model-platform-2026-07-20/brief.md`
- PRD: `_bmad-output/planning-artifacts/prds/prd-fraud-model-platform-2026-07-20/prd.md`
