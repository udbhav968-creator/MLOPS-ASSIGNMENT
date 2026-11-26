"""
Stage 4 retraining (PROBLEM.md section 7, steps 3 and 4).

later.csv is sorted by date. first 70% = new training data, last 30% = check set.

v2 uses v1's model setting, retrained on cleaned train.csv + the first 70%.
The newer rows get a larger sample weight, because later.csv shows the world
has changed (Yellow Line riders jumped). The weight is chosen WITHOUT touching
the check set: on an inner, time-ordered split of the first 70% only.

Promotion rule: promote v2 only if MAE_v2 <= (1 - 0.08) * MAE_v1 on the check set.
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from mp290.config import feature_lists, file_sha256, path_of
from mp290.model import build_pipeline
from mp290.train import log_sklearn_model, register


def split_later(later: pd.DataFrame, first_fraction: float):
    ordered = later.sort_values(["date", "record_id"]).reset_index(drop=True)
    cut = int(len(ordered) * first_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def fit_weighted(cand, numeric, categorical, old, new, weight, target, seed):
    data = pd.concat([old, new], ignore_index=True)
    w = np.r_[np.ones(len(old)), np.full(len(new), float(weight))]
    model = build_pipeline(cand, numeric, categorical, seed)
    model.fit(data[numeric + categorical], data[target], regressor__sample_weight=w)
    return model


def should_promote(mae_v1: float, mae_v2: float, min_improvement: float) -> bool:
    """Client rule: v2 must be at least `min_improvement` (e.g. 8%) better than v1."""
    return mae_v2 <= (1 - min_improvement) * mae_v1


def retrain_v2(params: dict, train: pd.DataFrame, later: pd.DataFrame, v1: dict) -> dict:
    numeric, categorical, target = feature_lists(params)
    features = numeric + categorical
    cfg = params["retrain"]
    seed = params["train"]["random_state"]
    data_hash = file_sha256(path_of(params["data"]["train"]))
    cand = v1["candidate"]

    first, check = split_later(later, cfg["first_fraction"])
    inner_fit, inner_val = split_later(first, cfg["first_fraction"])

    # choose the recency weight on the inner split (check set untouched)
    inner_scores = {}
    for w in cfg["recent_weights"]:
        with mlflow.start_run(run_name=f"v2_select_weight_{w}"):
            mlflow.log_params({"data_sha256": data_hash, "stage": "v2_weight_selection",
                               "model": cand["name"], "recent_weight": w})
            m = fit_weighted(cand, numeric, categorical, train, inner_fit, w, target, seed)
            score = mean_absolute_error(inner_val[target], m.predict(inner_val[features]))
            mlflow.log_metric("mae_inner_val", score)
            inner_scores[w] = score
    best_w = min(inner_scores, key=inner_scores.get)

    v1_model = v1["model"]
    mae_v1 = mean_absolute_error(check[target], v1_model.predict(check[features]))

    with mlflow.start_run(run_name="v2_retrained") as run:
        mlflow.log_params({"data_sha256": data_hash, "stage": "v2", "model": cand["name"],
                           "recent_weight": best_w, "train_rows": len(train) + len(first),
                           "later_first_rows": len(first), "check_rows": len(check)})
        v2_model = fit_weighted(cand, numeric, categorical, train, first, best_w, target, seed)
        mae_v2 = mean_absolute_error(check[target], v2_model.predict(check[features]))
        mlflow.log_metrics({"mae_check_v2": mae_v2, "mae_check_v1": mae_v1})
        uri = log_sklearn_model(v2_model, first[features].head(3))
        v2_run_id = run.info.run_id

    version = register(uri, params)
    promote = should_promote(mae_v1, mae_v2, cfg["promotion_min_improvement"])
    name = params["mlflow"]["registered_model_name"]
    champion = version if promote else v1["version"]
    MlflowClient().set_registered_model_alias(name, "champion", champion)

    print(f"  recency weight chosen on inner split: {best_w}  {inner_scores}")
    print(f"  check set MAE: v1 = {mae_v1:.2f}, v2 = {mae_v2:.2f} "
          f"({(1 - mae_v2 / mae_v1):.1%} lower)  -> {'promote' if promote else 'keep'}")
    print(f"  v2 = version {version} (run {v2_run_id}); champion = version {champion}")
    return {"model": v2_model, "version": version, "run_id": v2_run_id,
            "mae_v1": mae_v1, "mae_v2": mae_v2, "promote": promote,
            "champion_version": champion, "recent_weight": best_w}
