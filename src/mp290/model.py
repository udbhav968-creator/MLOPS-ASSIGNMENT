"""Builds the scikit-learn pipeline for one candidate setting from params.yaml."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def build_pipeline(candidate: dict, numeric: list[str], categorical: list[str],
                   random_state: int = 42) -> Pipeline:
    kind = candidate["model"]
    if kind == "ridge":
        degree = candidate.get("poly_degree", 1)
        numeric_steps = [StandardScaler()]
        if degree > 1:
            # squared terms let the model follow the curve in nearby_offices
            numeric_steps.append(PolynomialFeatures(degree, include_bias=False))
        regressor = Ridge(alpha=candidate.get("alpha", 1.0))
    elif kind == "gbr":
        numeric_steps = [StandardScaler()]
        regressor = GradientBoostingRegressor(
            n_estimators=candidate.get("n_estimators", 200),
            max_depth=candidate.get("max_depth", 3), random_state=random_state)
    elif kind == "rf":
        numeric_steps = [StandardScaler()]
        regressor = RandomForestRegressor(
            n_estimators=candidate.get("n_estimators", 300),
            max_depth=candidate.get("max_depth"), random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"unknown model type: {kind}")

    preprocess = ColumnTransformer([
        ("num", make_pipeline(*numeric_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])
