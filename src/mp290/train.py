"""
Stage 2: train v1 with MLflow tracking and register it.

Each candidate in params.yaml gets its own MLflow run:
  * fit on 80% of the cleaned train.csv, score MAE on the other 20% (mae_val)
  * refit on 100% of the cleaned train.csv and log that model
The run with the lowest mae_val is v1: registered as version 1, alias champion.

    python -m mp290.train
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from mp290.cleaning import load_clean_train
from mp290.config import feature_lists, file_sha256, load_params, path_of
from mp290.model import build_pipeline


def setup_mlflow(params: dict) -> None:
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def log_sklearn_model(model, example: pd.DataFrame) -> str:
    """Log a fitted pipeline in the active run and return its model URI."""
    example = example.astype({c: "float64" for c in example.select_dtypes("number").columns})
    try:
        info = mlflow.sklearn.log_model(model, name="model", input_example=example)
    except Exception as err:                       # MLflow 3.15+ "untrusted types"
        if "untrusted" not in str(err).lower():
            raise
        info = mlflow.sklearn.log_model(model, name="model", input_example=example,
                                        serialization_format="cloudpickle")
    return info.model_uri


def register(model_uri: str, params: dict) -> str:
    """Register a logged model; returns the new version number as a string."""
    version = mlflow.register_model(model_uri, params["mlflow"]["registered_model_name"])
    return str(version.version)


def reset_registered_model(params: dict) -> None:
    """Delete the registered model if it exists, so a full re-run gives v1 = version 1."""
    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]
    try:
        client.delete_registered_model(name)
    except Exception:
        pass


def train_v1(params: dict, train: pd.DataFrame) -> dict:
    numeric, categorical, target = feature_lists(params)
    features = numeric + categorical
    cfg = params["train"]
    data_hash = file_sha256(path_of(params["data"]["train"]))

    fit_part, val_part = train_test_split(train, test_size=cfg["test_size"],
                                          random_state=cfg["random_state"])
    results = []
    for cand in cfg["candidates"]:
        with mlflow.start_run(run_name=f"v1_{cand['name']}") as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("stage", "v1")
            mlflow.log_param("train_rows", len(train))
            mlflow.log_params({k: v for k, v in cand.items()})

            model = build_pipeline(cand, numeric, categorical, cfg["random_state"])
            model.fit(fit_part[features], fit_part[target])
            mae_val = mean_absolute_error(val_part[target], model.predict(val_part[features]))
            mlflow.log_metric("mae_val", mae_val)

            model = build_pipeline(cand, numeric, categorical, cfg["random_state"])
            model.fit(train[features], train[target])          # final v1 fit: all of train
            uri = log_sklearn_model(model, train[features].head(3))
            results.append({"name": cand["name"], "candidate": cand, "run_id": run.info.run_id,
                            "mae_val": mae_val, "model_uri": uri, "model": model})
            print(f"  {cand['name']:15s} mae_val = {mae_val:8.2f}   run {run.info.run_id}")

    best = min(results, key=lambda r: r["mae_val"])
    reset_registered_model(params)
    best["version"] = register(best["model_uri"], params)
    MlflowClient().set_registered_model_alias(
        params["mlflow"]["registered_model_name"], "champion", best["version"])
    print(f"  -> v1 = {best['name']} (run {best['run_id']}), "
          f"registered version {best['version']}, alias champion")
    return best


if __name__ == "__main__":
    p = load_params()
    setup_mlflow(p)
    train_v1(p, load_clean_train(p))
