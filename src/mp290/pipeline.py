"""
Runs the whole pipeline end to end and writes every output the tool reads:

    python -m mp290.pipeline

  1. seven data checks on train.csv        -> findings.yaml  data_issues
  2. clean train.csv
  3. v1: several MLflow runs, best registered as version 1   -> tracking
  4. PSI cleaned-train vs later, v1 MAE per line             -> monitoring
  5. v2 retrain, promotion rule, champion alias              -> monitoring
  6. champion -> models/model.joblib, holdout predictions    -> predictions/
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from mp290.cleaning import clean_train
from mp290.config import feature_lists, load_params, path_of
from mp290.monitor import mae_by_group, psi_table
from mp290.profiling import run_all_checks
from mp290.retrain import retrain_v2
from mp290.train import setup_mlflow, train_v1

HEADER = ("# MP290 findings --- written by `python -m mp290.pipeline` (src/mp290/pipeline.py).\n"
          "# Every value comes from our code run on our own data files.\n")


def to_plain(x):
    """numpy scalars -> plain python, so yaml.safe_dump can write them."""
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if hasattr(x, "item"):
        return x.item()
    return x


def main() -> dict:
    params = load_params()
    numeric, categorical, target = feature_lists(params)
    features = numeric + categorical
    group = params["features"]["group_column"]

    raw = pd.read_csv(path_of(params["data"]["train"]))
    later = pd.read_csv(path_of(params["data"]["later"]))
    holdout = pd.read_csv(path_of(params["data"]["holdout"]))

    print("[1] data checks")
    issues = run_all_checks(raw, list(holdout.columns))
    for k, v in issues.items():
        print(f"  {k:18s} {v}")

    print("[2] cleaning")
    train = clean_train(raw, list(holdout.columns))
    print(f"  {len(raw)} rows -> {len(train)} rows, columns: {list(train.columns)}")

    print("[3] v1 training with MLflow")
    setup_mlflow(params)
    v1 = train_v1(params, train)

    print("[4] drift and per-group error on later.csv")
    psi = psi_table(train, later, numeric, params["monitoring"]["psi_bins"])
    most_drifted = max(psi, key=psi.get)
    by_group = mae_by_group(v1["model"], later, features, target, group)
    worst = max(by_group, key=by_group.get)
    client = MlflowClient()
    client.log_metric(v1["run_id"], "mae_later", float(
        (later[target] - v1["model"].predict(later[features])).abs().mean()))
    for g, m in by_group.items():
        client.log_metric(v1["run_id"], f"mae_later_{g.replace(' ', '_')}", m)
    for col, value in psi.items():
        client.log_metric(v1["run_id"], f"psi_{col}", value)
    print(f"  PSI: {psi}\n  most drifted: {most_drifted}")
    print(f"  v1 MAE by {group}: {by_group}\n  worst served: {worst}")

    print("[5] v2 retraining and promotion rule")
    v2 = retrain_v2(params, train, later, v1)

    print("[6] champion -> model file and holdout predictions")
    name = params["mlflow"]["registered_model_name"]
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    model_path = path_of(params["outputs"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path)
    preds = pd.DataFrame({"record_id": holdout["record_id"],
                          "prediction": champion.predict(holdout[features]).round(1)})
    pred_path = path_of(params["outputs"]["predictions"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    print(f"  saved {model_path.name} and {len(preds)} predictions")

    findings = {
        "group_id": "G134",
        "problem_id": "MP290",
        "data_issues": issues,
        "tracking": {
            "experiment_name": params["mlflow"]["experiment_name"],
            "best_run_id": v1["run_id"],
            "registered_model_name": name,
            "champion_version": int(v2["champion_version"]),
        },
        "monitoring": {
            "psi": psi,
            "most_drifted_feature": most_drifted,
            "mae_by_group": by_group,
            "worst_served_group": worst,
            "mae_v1_on_check": round(v2["mae_v1"], 2),
            "mae_v2_on_check": round(v2["mae_v2"], 2),
            "decision": "promote" if v2["promote"] else "keep",
        },
    }
    out = path_of(params["outputs"]["findings"])
    with open(out, "w", encoding="utf-8") as f:
        f.write(HEADER)
        yaml.safe_dump(to_plain(findings), f, sort_keys=False, allow_unicode=True)
    print(f"  wrote {out.relative_to(path_of('.'))}")

    report = path_of("reports/drift_report.md")
    report.write_text(drift_report(train, later, psi, by_group, v1, v2, numeric, group),
                      encoding="utf-8")
    print(f"  wrote {report.relative_to(path_of('.'))}")
    return findings


def psi_label(value: float) -> str:
    return "big shift" if value > 0.25 else "moderate" if value > 0.1 else "stable"


def drift_report(train, later, psi, by_group, v1, v2, numeric, group) -> str:
    """A short Markdown report of the monitoring results, for people to read."""
    rows = "\n".join(
        f"| {c} | {train[c].min():g} - {train[c].max():g} (mean {train[c].mean():.1f}) "
        f"| {later[c].min():g} - {later[c].max():g} (mean {later[c].mean():.1f}) "
        f"| {psi[c]:.4f} | {psi_label(psi[c])} |" for c in numeric)
    groups = "\n".join(f"| {g} | {m:.2f} |" for g, m in by_group.items())
    change = 1 - v2["mae_v2"] / v2["mae_v1"]
    return f"""# Drift and retraining report (generated by src/mp290/pipeline.py)

## 1. PSI, cleaned train.csv (Jan-Jun) vs later.csv (Jul-Aug), 10 bins

Bins are the deciles of train; outer bins are open. PSI < 0.1 stable,
0.1 - 0.25 moderate, > 0.25 big shift.

| column | train range | later range | PSI | reading |
|---|---|---|---|---|
{rows}

**Most drifted: {max(psi, key=psi.get)}.**

## 2. v1 ({v1['name']}) MAE on later.csv, per {group}

| {group} | MAE (people) |
|---|---|
{groups}

**Worst served: {max(by_group, key=by_group.get)}.**

## 3. Retraining and the promotion rule

- later.csv sorted by date; first 70% joins the training data, last 30% is the check set.
- v2 = same setting as v1, newer rows weighted x{v2['recent_weight']}
  (weight chosen on an inner split of the first 70% only).
- Check-set MAE: v1 = {v2['mae_v1']:.2f}, v2 = {v2['mae_v2']:.2f}, i.e. {change:.1%} lower.
- Rule: promote only if v2 is at least 8% lower.
- **Decision: {'promote' if v2['promote'] else 'keep'}.** Champion = version {v2['champion_version']}.
"""


if __name__ == "__main__":
    main()
