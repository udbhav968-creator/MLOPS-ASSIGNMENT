# G134 --- MP290: Metro Station Crowd (Dehradun)

SCSE3040 MLOps mini project. Client: Dehradun Metro Rail Corporation.
We predict `entries_per_hour` (people entering a station in an hour) so extra
gates and staff can be put in place in time. Regression, measured by **MAE**.

| | |
|---|---|
| Group | **G134** |
| Problem | **MP290** |
| Faculty | Dr. Ravi Pratap, Batch 18 |

| Enrolment no. | Member |
|---|---|
| S24CSEU1284 | Pranjal Bisht |
| S24CSEU1277 | Anshul Kundu |
| S24CSEU0505 | Shivam Singh Anand |
| S24CSEU1283 | Prabhjot Singh |
| S24CSEU0095 | Udbhav |

## Repository layout

```
params.yaml                 every setting (paths, MLflow names, models, retrain rule)
src/mp290/
  config.py                 reads params.yaml, file hashing
  profiling.py              the seven data checks
  cleaning.py               fixes the problems that are present
  model.py                  scikit-learn pipelines
  train.py                  v1: MLflow runs, registry, champion alias
  monitor.py                PSI and MAE per group
  retrain.py                v2 and the 8% promotion rule
  pipeline.py               runs everything, writes all outputs
  selfcheck.py              checks the repo against the handbook before submitting
app/main.py                 FastAPI: POST /predict, GET /health, JSON logs
tests/                      pytest (no CSVs needed; TestClient for the API)
data/raw/*.csv.dvc          data under DVC
mlflow.db                   MLflow tracking + registry (committed)
models/model.joblib         the champion model the service loads
reports/findings.yaml       the filled findings form
reports/drift_report.md     PSI, per-line MAE and the promotion decision, for people
docs/                       team plan and viva guide
pyproject.toml, pytest.ini  package install and test settings
predictions/holdout_predictions.csv
Dockerfile, .dockerignore, .github/workflows/ci.yml, k8s/
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .                     # makes `mp290` importable; no PYTHONPATH needed
pip install dvc                      # DVC is a tool, kept out of the Docker image

python -m mp290.profiling            # the seven data checks only
python -m mp290.pipeline             # everything: clean, v1, drift, v2, champion, predictions, findings
pytest                               # 16 tests, no CSVs needed
python -m mp290.selfcheck            # the handbook's checklist, before submitting

uvicorn app.main:app --port 8000     # the service; docs at http://localhost:8000/docs
docker build -t mp290-api:1.0 . && docker run -p 8000:8000 mp290-api:1.0
kubectl apply -f k8s/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"hour": 18,
 "nearby_offices": 90, "temp_c": 30.5, "rain_mm": 2.0, "feeder_buses": 20,
 "ticket_price_rs": 35, "day_type": "weekday", "line": "Yellow Line"}'
```

## Results

### Data problems in train.csv (1040 rows as given)

| Problem | Present | Details | How we found it |
|---|---|---|---|
| leaking_column | yes | `gate_count_end_of_hour` | missing from the holdout file; counted at the END of the hour; equals the target within a few people |
| unit_mixup | yes | `rain_mm`, `Yellow Line`, factor 0.1 | Yellow Line mean rain is 3.0 vs about 29 on the other lines; recorded in cm instead of mm |
| sentinel_value | no | | no special number (-1, -999, 999, 9999...) and no value repeated abnormally often |
| duplicate_rows | yes | 40 extra copies | `df.duplicated().sum()` (1040 -> 1000 rows) |
| stuck_sensor | no | | sorted by date, no sensor value repeats for more than 2 days |
| target_typos | yes | 13 rows | sorted target jumps from 4,887 to 10,540 (a 2.2x gap); the 13 values above it are each about 10x their gate count |
| spelling_variants | no | | each text column has exactly 3 clean levels, same as later.csv |

### Model, drift and retraining (full tables in `reports/drift_report.md`)

- **v1**: four settings tried on a 20% validation split of cleaned train.csv:
  Ridge 145.1, **Ridge with squared terms 130.4**, gradient boosting 174.7,
  random forest 220.9. The winner is refit on all of train and registered as version 1.
- **Drift**: PSI of `nearby_offices` is 4.39 (above 0.25 is a big shift): it was
  0-80 in Jan-Jun and is 42-122 in Jul-Aug. Every other column is below 0.06.
- **Worst-served group**: v1's MAE on later.csv is 128.8 (Blue), 146.2 (Red) and
  **1360.1 (Yellow Line)**. v1 under-predicts every Yellow Line row, while its
  Red and Blue errors are centred on zero. So the offices drift alone did not
  break v1; Yellow Line ridership itself changed after June (concept drift).
- **v2**: same setting, retrained on cleaned train + the first 70% of later.csv,
  with the newer rows weighted more. The weight (x100) was picked on an inner
  time-ordered split of the first 70% only, so the check rows stay unseen.
- **Promotion rule**: on the last 30% of later.csv, v1 MAE 512.4, v2 MAE 204.0,
  60.2% lower (needs at least 8%) -> **promote**. v2 is version 2 and `@champion`.

## Who built what

See `docs/TEAM_PLAN.md` for the commit order. `docs/VIVA_GUIDE.md` explains every file.

| Member | Owns |
|---|---|
| | data: DVC, `profiling.py`, `cleaning.py`, findings data part |
| | model: `model.py`, `train.py`, MLflow registry, holdout prediction |
| | service: `app/main.py`, request rules, `tests/` |
| | ops: CI, Dockerfile, `k8s/`, logging |
| | monitoring: `monitor.py`, `retrain.py`, findings monitoring part |
