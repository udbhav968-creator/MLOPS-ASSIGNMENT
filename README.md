# Metro Station Crowd Prediction (Dehradun)

SCSE3040 MLOps mini project. **Group G134**, **Problem MP290**.

The Dehradun Metro Rail Corporation wants to know how many people will enter a
station in the next hour, so extra gates and staff can be ready in time. We
predict `entries_per_hour` (a regression problem, scored by MAE) and run the
model through a complete MLOps pipeline: versioned data, tracked experiments, a
model registry, a prediction API, a container, CI, Kubernetes, logging, drift
monitoring and retraining.

## Team

| Enrolment no. | Name |
|---|---|
| S24CSEU1284 | Pranjal Bisht |
| S24CSEU1277 | Anshul Kundu |
| S24CSEU0505 | Shivam Singh Anand |
| S24CSEU1283 | Prabhjot Singh |
| S24CSEU0095 | Udbhav |

Faculty: Dr. Ravi Pratap, Batch 18.

## Results at a glance

| | |
|---|---|
| Data problems found | 4 of 7 (leak, unit mix-up, duplicates, target typos) |
| Best v1 model | Ridge with squared terms, validation MAE **130.38** |
| Most drifted feature | `nearby_offices`, PSI **4.39** |
| Worst-served group | **Yellow Line**, MAE 1360.12 |
| v1 vs v2 on check set | MAE **512.36 → 204.01** (60.2% lower) |
| Decision | **Promote** v2 to `@champion` |

## Data problems in `train.csv`

| Problem | Found | Details |
|---|---|---|
| Leaking column | Yes | `gate_count_end_of_hour` is counted after the hour ends, is missing from the holdout file, and almost equals the target |
| Unit mix-up | Yes | Yellow Line recorded `rain_mm` in cm, factor 0.1 (mean 3.0 vs about 29 on the other lines) |
| Sentinel value | No | No special placeholder values such as -999 or 9999 |
| Duplicate rows | Yes | 40 extra copies (1040 rows become 1000) |
| Stuck sensor | No | No reading repeats for more than 2 days |
| Target typos | Yes | 13 targets typed 10x too large (sorted target jumps from 4,887 to 10,540) |
| Spelling variants | No | Every text column has clean, consistent categories |

Cleaning removes duplicates, converts Yellow Line rain back to mm, divides the
typo targets by 10, and drops the leaking column so it can never be a feature.

## Model, drift and retraining

**v1.** Four models were compared on a 20% validation split:

| Model | Validation MAE |
|---|---|
| Ridge | 145.11 |
| **Ridge + squared terms** | **130.38** |
| Gradient boosting | 174.71 |
| Random forest | 220.86 |

**Drift (July and August data).** PSI per feature, train vs later:

| Feature | PSI |
|---|---|
| hour | 0.023 |
| **nearby_offices** | **4.386** |
| temp_c | 0.027 |
| rain_mm | 0.027 |
| feeder_buses | 0.042 |
| ticket_price_rs | 0.054 |

`nearby_offices` moved from 0-80 to 42-122. v1 still handled Red Line (MAE
146.16) and Blue Line (128.82) well, but under-predicted every Yellow Line row
(MAE 1360.12): Yellow Line ridership itself changed after June.

**v2.** Same model, retrained on the cleaned training data plus the first 70% of
`later.csv` (by date), with newer rows weighted more heavily. The weight was
chosen on the first 70% only, so the last 30% stayed unseen for the final
comparison.

**Promotion rule.** v2 is promoted only if its MAE is at least 8% lower than
v1's on the check set: 204.01 ≤ 0.92 × 512.36 = 471.37, so v2 was promoted. It
is registered as version 2 with the `champion` alias and is used for the
September holdout predictions.

Full tables: [`reports/drift_report.md`](reports/drift_report.md).
All answers for marking: [`reports/findings.yaml`](reports/findings.yaml).

## Project structure

```
├── params.yaml                    all settings (paths, models, MLflow, retraining)
├── src/mp290/
│   ├── config.py                  reads params.yaml
│   ├── profiling.py               the seven data checks
│   ├── cleaning.py                fixes the problems found
│   ├── model.py                   scikit-learn pipelines
│   ├── train.py                   v1 training, MLflow tracking and registry
│   ├── monitor.py                 PSI and per-line MAE
│   ├── retrain.py                 v2 training and the promotion rule
│   ├── pipeline.py                runs every step end to end
│   └── selfcheck.py               checks the repository before submission
├── app/main.py                    FastAPI service
├── tests/                         pytest suite (16 tests)
├── data/raw/                      DVC-tracked data (.dvc files) and the holdout file
├── mlflow.db                      MLflow runs, registry and aliases
├── models/model.joblib            the champion model used by the API
├── predictions/                   holdout predictions
├── reports/                       findings.yaml and the drift report
├── Dockerfile, .dockerignore
├── .github/workflows/ci.yml       runs pytest on every push
└── k8s/                           Deployment and Service
```

## Getting started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The raw CSVs are tracked with DVC and are not stored in git. Place the original
`train.csv` and `later.csv` in `data/raw/` (never open and re-save them in
Excel), then:

```bash
python -m mp290.profiling          # the seven data checks
python -m mp290.pipeline           # clean, train, monitor, retrain, predict
pytest                             # run the tests
python -m mp290.selfcheck          # check the repository
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Prediction API

```bash
uvicorn app.main:app --port 8000
```

Interactive docs: http://localhost:8000/docs

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "hour": 18, "nearby_offices": 90, "temp_c": 30.5, "rain_mm": 2.0,
  "feeder_buses": 20, "ticket_price_rs": 35,
  "day_type": "weekday", "line": "Yellow Line"}'
```

| Endpoint | Returns |
|---|---|
| `POST /predict` | `{"entries_per_hour": ..., "model_version": "champion"}` |
| `GET /health` | `{"status": "ok", "model_loaded": true, ...}` |

The client's two rules are enforced in the request schema, so invalid input
returns **422**: `nearby_offices` above 150, and `ticket_price_rs` below 5.
Each prediction writes one JSON log line to stdout.

## Docker and Kubernetes

```bash
docker build -t mp290-api:1.0 .
docker run -p 8000:8000 mp290-api:1.0

kubectl apply -f k8s/
```

Deployment `mp290-api` runs 3 replicas with readiness and liveness probes on
`/health`. Service `mp290-svc` exposes port 80 and forwards to the containers'
port 8000.

## MLflow

| | |
|---|---|
| Experiment | `G134_MP290_metro_crowd` |
| Registered model | `mp290-entries-per-hour` |
| Version 1 | v1, trained on cleaned `train.csv` |
| Version 2 | v2, retrained with July-August data |
| `@champion` | version 2 |

Every run logs `data_sha256`, the SHA-256 hash of the original `train.csv`.
