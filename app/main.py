"""
MP290 prediction service.

  POST /predict  -> {"entries_per_hour": <float>}
  GET  /health   -> {"status": "ok", ...}

The client's two rules live in the request schema, so a bad request is
rejected with 422 by pydantic and never reaches the model:
  1. nearby_offices above 150 is impossible   -> Field(le=150)
  2. ticket_price_rs below 5 is a typing error -> Field(ge=5)
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")


# ---------- JSON logging: one line per event, to stdout, no personal data ----------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                 "level": record.levelname, "logger": record.name,
                 "message": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


class StdoutHandler(logging.StreamHandler):
    """Always writes to the current sys.stdout (also when a test captures it)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


logger = logging.getLogger("mp290-api")
if not logger.handlers:
    handler = StdoutHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# ---------- request / response schema ----------
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")          # exactly these fields, nothing else

    hour: int = Field(..., ge=0, le=23, description="hour of the day")
    nearby_offices: int = Field(..., ge=0, le=150, description="offices within 1 km; >150 impossible")
    temp_c: float = Field(..., description="temperature, deg C")
    rain_mm: float = Field(..., ge=0, description="rain in the last hour, mm")
    feeder_buses: int = Field(..., ge=0, description="feeder buses arriving")
    ticket_price_rs: float = Field(..., ge=5, description="average fare, rupees; <5 is a typo")
    day_type: Literal["weekday", "weekend", "holiday"]
    line: Literal["Red Line", "Blue Line", "Yellow Line"]


class PredictResponse(BaseModel):
    entries_per_hour: float
    model_version: str


def log_event(message: str, level: int = logging.INFO, **fields) -> None:
    """Write one JSON line. Only technical fields: never a person's details."""
    logger.log(level, message, extra={"extra_fields": {"event": message, **fields}})


# ---------- app ----------
model = None


def get_model():
    """Load the champion model once; tests can set `app.main.model` directly."""
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
        log_event("model_loaded", path=str(MODEL_PATH), model_version=MODEL_VERSION)
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # load at start-up, so the first real request is not slow and a missing
    # model file shows up in the logs straight away
    try:
        get_model()
    except Exception as err:                       # keep serving /health, log the problem
        log_event("model_load_failed", logging.ERROR, error=str(err))
    yield


app = FastAPI(title="MP290 Metro Station Crowd", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log_event("request", method=request.method, path=request.url.path,
              status=response.status_code,
              latency_ms=round((time.perf_counter() - start) * 1000, 2))
    return response


@app.exception_handler(RequestValidationError)
async def rejected(request: Request, exc: RequestValidationError):
    # log which fields broke the rules, not the values that were sent
    fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
    log_event("request_rejected", logging.WARNING, path=request.url.path, fields=fields)
    return JSONResponse(status_code=422, content={"detail": jsonable(exc.errors())})


def jsonable(errors):
    return [{k: (v if isinstance(v, (str, int, float, list, type(None))) else str(v))
             for k, v in e.items()} for e in errors]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    current = get_model()                     # loaded once, outside the timing
    start = time.perf_counter()
    features = pd.DataFrame([req.model_dump()])
    value = float(current.predict(features)[0])
    log_event("prediction", line=req.line, day_type=req.day_type, hour=req.hour,
              prediction=round(value, 1), model_version=MODEL_VERSION,
              latency_ms=round((time.perf_counter() - start) * 1000, 2))
    return PredictResponse(entries_per_hour=round(value, 1), model_version=MODEL_VERSION)
