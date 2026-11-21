# G134 team plan: from this folder to a marked submission

The tool counts commits **per GitHub account** and checks that the parts each
of you claims in the form really exist. So each member commits their own part,
from their own account, in the order below. This order was tested: after the
CI commit, every single commit passes all tests on a clean checkout, so the
Actions tab stays green the whole way.

Pick who takes which slot. Everyone must still be able to explain every file
(`docs/VIVA_GUIDE.md`).

| Slot | Role | Owns |
|---|---|---|
| M1 | Data lead | DVC, `params.yaml`, `config.py`, `profiling.py`, `cleaning.py`, data tests |
| M2 | Model lead | `model.py`, `train.py`, MLflow registry, `pyproject.toml` |
| M3 | Service lead | `app/main.py`, request rules, JSON logging, API tests, Dockerfile |
| M4 | Ops lead | GitHub Actions, Kubernetes, `selfcheck.py` |
| M5 | Monitoring lead | `monitor.py`, `retrain.py`, `pipeline.py`, running it, findings |

## 0. Once, before anything (every member)

Your commits only count for you if git uses an e-mail **linked to your GitHub
account** (GitHub > Settings > Emails).

```bash
git config --global user.name  "Your Name"
git config --global user.email "the-email-on-your-github-account"
```

## 1. M1 creates the repository

1. GitHub > New repository > name `G134_MP290` > **Public** > no README.
2. Settings > Collaborators > add the other four. Each of them accepts the invite.
3. Unzip `G134_MP290_repo.zip`, then in that folder:

```bash
git init -b main
git remote add origin https://github.com/<M1-username>/G134_MP290.git

git add README.md .gitignore .gitattributes .dvcignore .dvc data/raw/train.csv.dvc \
        data/raw/later.csv.dvc data/raw/.gitignore data/raw/holdout_features.csv \
        params.yaml requirements.txt pytest.ini src/mp290/__init__.py src/mp290/config.py \
        src/mp290/profiling.py src/mp290/cleaning.py tests/conftest.py tests/test_data_checks.py
git commit -m "Data: DVC, params, seven data checks, cleaning, tests"
git push -u origin main
```

`train.csv` and `later.csv` stay out of git (`data/raw/.gitignore` from DVC
does that); their `.dvc` files go in. The `.dvc` files already hold the md5 of
your original files. If you prefer to make them yourself, delete `.dvc/`,
`.dvcignore` and the two `.dvc` files first, then run `dvc init` and
`dvc add data/raw/train.csv data/raw/later.csv`; you get the same hashes.

## 2. Everyone else: clone once, then commit your part in turn

```bash
git clone https://github.com/<M1-username>/G134_MP290.git
cd G134_MP290
```

Copy the files for your part from the zip into the clone (same paths). The
CSVs are not in git, so copy `train.csv` and `later.csv` from **the original
problem zip** into `data/raw/` too (never through Excel). Always
`git pull` before you start, then:

**M4** (makes CI exist; first green tick):
```bash
git add .github/workflows/ci.yml
git commit -m "Ops: GitHub Actions CI running pytest on push"
git push
```

**M2**:
```bash
git add src/mp290/model.py src/mp290/train.py pyproject.toml
git commit -m "Model: sklearn pipelines, v1 training with MLflow tracking and registry"
git push
```

**M3**:
```bash
git add app/__init__.py app/main.py tests/test_api.py Dockerfile .dockerignore
git commit -m "Service: FastAPI, client rules in schema, JSON logging, Dockerfile, API tests"
git push
```

**M5** (code first, then the run):
```bash
git add src/mp290/monitor.py src/mp290/retrain.py src/mp290/pipeline.py tests/test_monitoring.py
git commit -m "Monitoring: PSI, per-line MAE, v2 retrain, promotion rule"
git push
```

Then M5 runs the whole pipeline **on their own machine**, so `mlflow.db`
points at model files that exist there. Numbers come out the same; only the
run ids change, and `findings.yaml` is updated automatically:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt && pip install -e .
rm -f mlflow.db                                       # Windows: del mlflow.db
python -m mp290.pipeline
git add mlflow.db models/model.joblib predictions/holdout_predictions.csv \
        reports/findings.yaml reports/drift_report.md
git commit -m "Run pipeline: MLflow runs and registry, champion model, findings, holdout predictions"
git push
```

**M4**:
```bash
git add k8s/deployment.yaml k8s/service.yaml src/mp290/selfcheck.py
git commit -m "Ops: Kubernetes manifests and submission self-check"
git push
```

**M1** (last, ties it together):
```bash
git add docs/
git commit -m "Docs: team plan and viva guide"
git push
```

Small real improvements by anyone afterwards (a clearer comment, a better
README line) are good: they are honest extra commits.

## 3. Check before anyone fills the form

```bash
git pull
python -m mp290.selfcheck        # should end with every check passing
```

Then by hand:

- [ ] Open the repo link in a **private** browser window: it opens, so it is public.
- [ ] The **Actions** tab shows a green tick on the latest commit.
- [ ] The **Insights > Contributors** page shows all five of you.
- [ ] Optional: `docker build -t mp290-api:1.0 .` and `docker run -p 8000:8000 mp290-api:1.0`, then open http://localhost:8000/docs.

## 4. Submit (every member, separately, before Sun 29 Nov 11:59 PM)

Form: https://forms.cloud.microsoft/r/9ZPB1ZzcAK, signed in with your Bennett account.

- Group `G134`, problem `MP290`, your GitHub username.
- **The same repository link** as everyone else.
- The parts you built, your files and your hours: describe what you actually
  did. The tool checks your claims against your commits, and the viva checks
  them against you.
- If your course asks you to declare help from AI tools, declare it.

The tool marks the last commit made before the **last** member's form time.
Filling the form again after 29 Nov turns your Timely mark from 3 into 2, so
push everything first and fill the form once.
