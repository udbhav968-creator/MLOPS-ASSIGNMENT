# G134 / MP290 viva guide

Every member is examined separately, and the handbook says you will run the
code. Read all of it, not only your own part. Each section ends with the
questions an examiner is likely to ask.

## 0. The project in one minute

Dehradun Metro wants to know how many people will enter a station in the
next hour (`entries_per_hour`), so it can open extra gates in time. We get six
months of messy history (`train.csv`), two later months that arrived after
going live (`later.csv`), and September without answers (`holdout_features.csv`).

We found and fixed four data problems, trained four models and tracked them in
MLflow, served the best one through a FastAPI service in Docker and
Kubernetes, then watched it on the new data. One feature drifted a lot, and
Yellow Line ridership changed. We retrained, the new model was 60% better on
unseen rows, so it was promoted to champion and used for September.

## 1. Data and DVC

**Why DVC.** Git is bad at data: big files, and every change stores a full
copy. DVC keeps a small text file (`data/raw/train.csv.dvc`) in git that holds
the file's **md5 hash** and size; the CSV itself is listed in
`data/raw/.gitignore`. Anyone can check that their file is exactly the
recorded version.

Our `train.csv.dvc` says `md5: 4eeb036e23b2fdcaf8a5ae64a80ef7e5`, which is the
md5 of the original file in the zip. MLflow additionally logs its SHA-256
(`db4f47de48a7...`) as `data_sha256` in every run, so every model can be
traced back to the exact data it was trained on (data lineage).

**Why not Excel.** Saving in Excel rewrites the file (dates, number formats,
line endings), so the hash changes and the tool treats the data as not ours.
`.gitattributes` protects us from the same thing on Windows: `*.csv -text`
stops git from turning line endings into CRLF.

Questions:
- *What is inside a .dvc file?* The md5, the size, the path. Not the data.
- *Where is the data, then?* On our machines (from the zip). We have no DVC
  remote; `dvc push` to a remote (Google Drive, S3) would be the next step.
- *What does `dvc add` do?* Hashes the file, copies it into `.dvc/cache`,
  writes the `.dvc` file and adds the CSV to `.gitignore`.

## 2. The seven data problems (`src/mp290/profiling.py`)

Run it live: `python -m mp290.profiling`. Counts are on train.csv as given
(1040 rows).

| Problem | Answer | How the code finds it |
|---|---|---|
| leaking_column | `gate_count_end_of_hour` | `check_leaking_column`: a column missing from the holdout file whose correlation with the target is above 0.99 on normal rows |
| unit_mixup | `rain_mm`, `Yellow Line`, factor 0.1 | `check_unit_mixup`: per-line mean of each column; Yellow is 3.0 against 28.2 (Blue) and 29.1 (Red); 3.0 / 28.6 = 0.105, which is closest to the exact conversion 0.1 |
| sentinel_value | absent | `check_sentinel_value`: no -1, -99, -999, 999, 9999 anywhere, and no value repeated abnormally often |
| duplicate_rows | 40 | `df.duplicated().sum()`; 40 record_ids appear twice with identical rows; 1040 to 1000 |
| stuck_sensor | absent | `check_stuck_sensor`: sort by date, find runs of one repeated value; the longest in any sensor column covers 2 days |
| target_typos | 13 | `target_typo_mask`: sort the target; the biggest jump is 4,887 to 10,540 (2.2x); the 13 values above it are each about 10x their own gate count and all end in 0 |
| spelling_variants | absent | `check_spelling_variants`: lower-case, strip spaces and punctuation; still 3 levels in `day_type` and 3 in `line`, the same as later.csv |

**The leak in depth.** `gate_count_end_of_hour` is counted by the gate
machines when the hour is over. We must predict before the hour starts, so in
real life the value does not exist yet. A model using it would look nearly
perfect in training (it is almost the answer) and be useless in production.
Three signs: its name, its absence from `holdout_features.csv` and from the
API's request fields, and its near-perfect agreement with the target.

**The unit factor.** The form defines `wrong value = right value x factor`.
Yellow Line typed rain in centimetres: 30 mm becomes 3.0 cm, so
`3.0 = 30 x 0.1`, factor 0.1. The fix divides by the factor (multiplies by 10).
After the fix, Yellow Line's mean rain is 30.0, in line with the other lines
and with later.csv, where Yellow is 33.1.

**Why `nearby_offices` is not a data problem.** It is 0-80 in train and
42-122 in later.csv. That is a real change in the world (drift), not a typing
mistake, and later.csv is described as clean. It belongs to monitoring, not
cleaning.

Questions:
- *Why is the stuck-sensor threshold 7 days?* The handbook describes a sensor
  stuck "for a few weeks". Our longest run is 2 days, so any sensible
  threshold gives the same answer.
- *Could a sentinel be 0?* Zero is a real value here (no rain, no buses). The
  zeros are spread normally, not one huge spike, so they are genuine readings.
- *How do you know the typos are 10x and not, say, 9x?* Each typo divided by 10
  lands within a few people of its gate count, as every normal row does.

## 3. Cleaning (`src/mp290/cleaning.py`)

Order matters:

1. **Drop duplicates first**, so nothing below is computed twice.
2. **Fix the unit mix-up**: divide Yellow Line `rain_mm` by 0.1.
3. **Fix target typos**: integer-divide the 13 values by 10.
4. **Drop the leaking column last**.

The fixes are driven by the check functions (the column, line and factor come
from `check_unit_mixup`), so nothing is hard-coded. We never drop the typo
rows: dividing by 10 recovers the true value, so we keep the information.
Result: 1000 rows, 11 columns, no `gate_count_end_of_hour`.

Questions:
- *You used the gate count in profiling. Isn't that the leak?* The rule is
  about **model features**. Using a column to audit data quality is fine; it
  never reaches the model.
- *Why not clean later.csv?* PROBLEM.md says it is clean. We only select the
  eight feature columns from it, so its gate count is ignored too.

## 4. The model (`src/mp290/model.py`, `src/mp290/train.py`)

Features (exactly the API fields): `hour, nearby_offices, temp_c, rain_mm,
feeder_buses, ticket_price_rs` (numeric) and `day_type, line` (categorical).

Each model is one scikit-learn `Pipeline`:
- `ColumnTransformer`: numeric columns go through `StandardScaler` (and, for
  the winner, `PolynomialFeatures(degree=2)`); categoricals go through
  `OneHotEncoder(handle_unknown="ignore")`.
- then the regressor.

Keeping preprocessing inside the pipeline means the exact same transformation
is saved with the model and reused in the API: no training/serving skew.

Four settings from `params.yaml`, scored by MAE on a 20% validation split
(`random_state=42`):

| Setting | Validation MAE |
|---|---|
| Ridge | 145.1 |
| **Ridge + squared terms** | **130.4** |
| Gradient boosting | 174.7 |
| Random forest | 220.9 |

Degree-2 polynomial features turn 6 numeric columns into 27 (6 originals, 6
squares, 15 pairwise products). They let a linear model follow the curve we saw
in the residuals against `nearby_offices`. Ridge's penalty (`alpha=1.0`) keeps
the extra coefficients from overfitting.

Why the tree models lose: with only 1000 rows and a mostly smooth
relationship, a linear model with a few curved terms fits better. Trees also
cannot predict outside the range they were trained on, which matters a lot
here because `nearby_offices` moves to 42-122.

Each run fits on 80% for the score, then refits on 100% of cleaned train and
logs that model. The best run (lowest `mae_val`) is v1.

Questions:
- *What is MAE and why MAE?* Mean absolute error: the average size of a
  mistake, in people. The client chose it; it is easy to explain and not
  dominated by a few large errors like RMSE.
- *Why `handle_unknown="ignore"`?* A category never seen in training becomes
  all zeros instead of crashing. Our API only allows known values anyway.

## 5. MLflow (`train.py`, `retrain.py`, `mlflow.db`)

- `mlflow.set_tracking_uri("sqlite:///mlflow.db")`: runs, parameters, metrics,
  the model registry and aliases all live in one SQLite file that we commit.
  Model files go to `mlruns/`, which is git-ignored.
- Experiment `G134_MP290_metro_crowd`. **12 runs**: 4 for v1, 7 for choosing
  v2's weight, 1 for v2. Every run logs `data_sha256`.
- Metrics: `mae_val` (v1 candidates), `mae_later`, `mae_later_<line>` and
  `psi_<column>` (added to the v1 run), `mae_inner_val` (weight search),
  `mae_check_v1` and `mae_check_v2` (the final comparison).
- Registry: model `mp290-entries-per-hour`, version 1 = v1, version 2 = v2.
- Alias `champion` points at the version in use (2). The pipeline loads
  `models:/mp290-entries-per-hour@champion`, saves it as
  `models/model.joblib` and predicts September with it.

Look at it live: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Questions:
- *Why an alias instead of "the latest version"?* The newest version is not
  automatically the best. The alias records a decision (our promotion rule);
  rolling back is just moving the alias back to version 1.
- *Where is `best_run_id` from?* The v1 run with the lowest `mae_val`.
- *Why does the pipeline delete the registered model at the start?* So a full
  re-run always gives v1 = version 1 and v2 = version 2, as the handbook asks.

## 6. The API (`app/main.py`)

- `POST /predict` with exactly the eight fields, `GET /health`.
- The client's two rules are in the pydantic schema:
  `nearby_offices: int = Field(..., ge=0, le=150)` and
  `ticket_price_rs: float = Field(..., ge=5)`. FastAPI validates the body
  before our function runs, so a bad request gets **422** and never reaches
  the model. 150 and 5 themselves are allowed (`le` = less or equal).
- Extra protection: `extra="forbid"` (so sending `gate_count_end_of_hour` is
  rejected), `Literal` types for `day_type` and `line`, `hour` between 0 and 23.
- The model is loaded once at start-up (`lifespan`), not on every request.
  `/health` also reports `model_loaded`.
- **Logging**: the `logging` module with our own `JsonFormatter`. One JSON line
  per prediction (line, day type, hour, prediction, latency, model version),
  one per request (method, path, status, latency) and one per rejected
  request (only the **field names**, not the values sent). Everything goes to
  stdout, never a file: in Docker and Kubernetes, the platform collects
  stdout (`docker logs`, `kubectl logs`). No personal details are logged.

Demo live:
```bash
uvicorn app.main:app --port 8000
# second terminal, or open http://localhost:8000/docs
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"hour":18,"nearby_offices":151,"temp_c":30,"rain_mm":2,"feeder_buses":20,"ticket_price_rs":35,"day_type":"weekday","line":"Red Line"}'
# -> 422, and a request_rejected log line naming nearby_offices
```

Questions:
- *Why put rules in the schema, not in an `if` inside the function?* One
  place, automatic 422 with a clear message, shown in `/docs`, and the model
  code never sees invalid data.
- *Why is 422 the right code?* The JSON is well-formed but its values break
  the rules: "unprocessable entity". 400 would mean the request itself is broken.

## 7. Tests (`tests/`, 16 tests)

- `test_data_checks.py`: clean synthetic data shows only the leak; planted
  duplicates, a typo, a cm/mm mix-up, a stuck sensor and a spelling variant
  are each found; cleaning fixes them.
- `test_monitoring.py`: PSI near 0 for the same distribution and above 0.25
  for a shifted one; the 8% rule at exactly 8%, at 7% and when v2 is worse;
  later.csv is split by date.
- `test_api.py`: uses FastAPI's `TestClient` with a tiny model trained inside
  the test. Health, a valid prediction, 422 for each rule, boundary values
  150 and 5 accepted, missing or extra fields rejected, one JSON log line per
  prediction.

None of them read the CSVs, because the CSVs are not in git, so CI does not have them.

## 8. Docker (`Dockerfile`, `.dockerignore`)

- `python:3.12-slim`: small official base image.
- `COPY requirements.txt` and `pip install` **before** copying the code. Docker
  caches layers, so changing a `.py` file does not reinstall every library.
- Then copy `src/`, `app/`, `params.yaml` and `models/model.joblib`.
- `EXPOSE 8000`; `uvicorn ... --host 0.0.0.0 --port 8000`. `0.0.0.0` means
  "listen on every network interface". The default `127.0.0.1` would only
  accept connections from inside the container itself.
- `.dockerignore` keeps data, `mlflow.db`, `.git`, tests and caches out of the
  build context: a smaller, faster image without our data in it.

```bash
docker build -t mp290-api:1.0 .
docker run -p 8000:8000 mp290-api:1.0
```

## 9. CI (`.github/workflows/ci.yml`)

On every push (and pull request): check out the code, set up Python 3.12 with
a pip cache, `pip install -r requirements.txt`, `pytest -v`. The file starts
with the key `on:`; written through `yaml.safe_dump` it would become `true:`
because YAML 1.1 reads a bare `on` as a boolean.

Every requirement is pinned (`pandas==3.0.6` and so on) so CI, Docker and our
laptops install the same versions. That also matters because a model saved
with joblib should be loaded with the same scikit-learn version.

## 10. Kubernetes (`k8s/`)

- **Deployment `mp290-api`**: keeps **3 replicas** of the Pod running, image
  `mp290-api:1.0`, `containerPort: 8000`, label `app: mp290-api`, CPU and
  memory requests and limits.
- **Service `mp290-svc`**: one stable address in front of the 3 Pods. It finds
  them with `selector: app: mp290-api`, which must match the Pod labels.
  Port 80 outside, forwarded to `targetPort` 8000 in the container.
- **Readiness probe** (`/health`, after 5 s): "may this Pod get traffic yet?"
  Failing removes the Pod from the Service; nothing is restarted.
- **Liveness probe** (`/health`, after 20 s): "is this Pod still alive?"
  Failing restarts the container. It waits longer so a Pod that is still
  starting up (loading the model) is not killed in a restart loop.

```bash
kubectl apply -f k8s/
kubectl get pods -l app=mp290-api
kubectl port-forward svc/mp290-svc 8080:80
```

## 11. Monitoring and retraining (`monitor.py`, `retrain.py`)

**PSI** compares where the data falls between two periods.
- Bin edges are the 10 deciles of the reference (cleaned train), outer edges
  opened to minus and plus infinity.
- For each bin: expected % (train) and actual % (later).
- `PSI = sum over bins of (actual - expected) * ln(actual / expected)`, with a
  tiny epsilon so an empty bin does not give ln(0).
- Below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 big shift.

| Column | PSI (deciles, used) | PSI (equal-width, for comparison) |
|---|---|---|
| hour | 0.023 | 0.043 |
| **nearby_offices** | **4.386** | **4.577** |
| temp_c | 0.027 | 0.035 |
| rain_mm | 0.027 | 0.027 |
| feeder_buses | 0.042 | 0.040 |
| ticket_price_rs | 0.054 | 0.056 |

Whichever binning P13 used, `nearby_offices` is by far the most drifted.

**Per-line MAE of v1 on later.csv**: Blue 128.8, Red 146.2, **Yellow 1360.1**.
v1 under-predicts **every** Yellow Line row, while its Red and Blue errors are
centred on zero. So:
- The `nearby_offices` drift is **data drift**: the inputs moved, but v1's
  squared terms extrapolated well, and Red and Blue are still fine.
- Yellow Line is **concept drift**: the same inputs now mean more riders
  (Yellow's average went from 2,730 to 5,254 per hour), which no amount of
  extrapolation can fix. Only new training data can.

**Retraining.**
1. Sort later.csv by date. First 70% (210 rows, 1 July to 10 August) = new
   training data; last 30% (90 rows, 10 to 31 August) = check set. The cut
   falls inside 10 August; ties are ordered by `record_id`, so the split is
   still deterministic and no row is in both sets.
2. v2 uses v1's setting, trained on cleaned train + the first 70%.
3. Newer rows get a bigger `sample_weight` because the world changed. The
   weight is chosen on an **inner** time-ordered split of the first 70% (its
   first 70% to fit, its last 30% to score). Inner MAE: weight 1: 466,
   10: 302, 50: 185, 100: 174. We take 100. The check set is never used to
   choose anything.
4. Check set: v1 512.36, v2 204.01.

**Promotion rule.** Promote only if `MAE_v2 <= (1 - 0.08) x MAE_v1`:
`0.92 x 512.36 = 471.37`, and `204.01 <= 471.37`, so **promote**. v2 is
registered as version 2 and gets the `champion` alias.

Questions:
- *Why sort by date before splitting?* To mimic real life: train on the past,
  test on the future. A random split would leak future rows into training.
- *Why not pick the weight on the check set?* That would be tuning on the
  test: v2 would look better than it really is and the 8% comparison would be
  unfair to v1.
- *Weight 100 almost ignores train.csv. Why keep it?* The relationships that
  did not change (hours, rain, buses, day types, Red and Blue) are still
  learned from 1000 rows; the weight mostly lets the newer rows set the
  current level, especially for Yellow Line.
- *What would make you "keep"?* v2 less than 8% better: the cost of a
  deployment (risk, testing) must be paid by a clear gain.

## 12. Final prediction

`pipeline.py` loads `models:/mp290-entries-per-hour@champion` (v2), predicts
all 250 rows of `holdout_features.csv` and writes
`predictions/holdout_predictions.csv` with exactly `record_id,prediction`.
Sanity check: the average prediction per line (Blue 3,926, Red 3,899,
Yellow 5,498) sits close to the July-August averages, slightly higher because
September has a few more nearby offices (mean 82.6 vs 79.8).

## 13. Honest limitations (good to say before you are asked)

- No DVC remote: the data is reproducible from the zip, but not `dvc pull`-able.
- The Docker image is large because `requirements.txt` includes MLflow, which
  the API does not need; a separate `requirements-api.txt` would shrink it.
- We could not run a real Kubernetes cluster; the manifests are checked by
  `selfcheck.py`, not by `kubectl apply`.
- Drift is checked once, by hand. In production we would compute PSI on a
  schedule and alert above 0.25.
- The Yellow Line change is unexplained: worth asking the client (a new
  connection, a closed road?).

## 14. Commands to know by heart

```bash
python -m mp290.profiling       # seven checks
python -m mp290.pipeline        # everything
pytest -v                       # tests
python -m mp290.selfcheck       # handbook checklist
uvicorn app.main:app --port 8000
mlflow ui --backend-store-uri sqlite:///mlflow.db
dvc status
```
