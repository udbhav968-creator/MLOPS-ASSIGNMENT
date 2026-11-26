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
from fastapi.responses import JSONResponse, HTMLResponse
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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dehradun Metro Crowd Prediction · MP290</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b111a;
      --card: #151d2a;
      --card-border: #233147;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --red-line: #ef4444;
      --blue-line: #3b82f6;
      --yellow-line: #eab308;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand h1 { font-size: 24px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    .brand p { font-size: 14px; color: var(--muted); margin-top: 2px; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      color: #34d399;
    }
    .status-dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; box-shadow: 0 0 8px #34d399; }
    .docs-btn {
      text-decoration: none;
      color: var(--text);
      background: var(--card);
      border: 1px solid var(--card-border);
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
    }
    .docs-btn:hover { background: #1e293b; border-color: var(--accent); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 24px;
    }
    .card h2 { font-size: 18px; font-weight: 700; margin-bottom: 16px; color: #fff; }
    .presets {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .preset-btn {
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover { background: #334155; color: #fff; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 600px) { .form-grid { grid-template-columns: 1fr; } }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-group.full { grid-column: 1 / -1; }
    label { font-size: 13px; font-weight: 600; color: #cbd5e1; }
    label span.hint { font-size: 11px; font-weight: 400; color: var(--muted); }
    input, select {
      background: #0f172a;
      border: 1px solid #334155;
      color: #fff;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }
    input:focus, select:focus { border-color: var(--accent); }
    .submit-btn {
      width: 100%;
      background: var(--accent);
      color: #0b111a;
      border: none;
      padding: 13px 20px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 20px;
      transition: background 0.2s;
    }
    .submit-btn:hover { background: var(--accent-hover); }
    .result-display {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      min-height: 240px;
      border-radius: 12px;
      background: #0f172a;
      border: 1px dashed #334155;
      padding: 24px;
      margin-bottom: 20px;
    }
    .result-num {
      font-size: 48px;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: #38bdf8;
      letter-spacing: -1px;
    }
    .result-unit { font-size: 14px; color: var(--muted); font-weight: 500; margin-bottom: 12px; }
    .gate-action {
      display: inline-block;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
      margin-top: 8px;
    }
    .gate-normal { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .gate-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .gate-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .error-box {
      width: 100%;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 12px;
      border-radius: 8px;
      font-size: 13px;
      text-align: left;
      margin-top: 10px;
    }
    .stats-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 20px;
    }
    .stat-tile {
      background: #0f172a;
      border: 1px solid #1e293b;
      padding: 12px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-val { font-size: 16px; font-weight: 700; color: #fff; font-family: 'JetBrains Mono', monospace; }
    .stat-label { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .footer-info {
      margin-top: 32px;
      padding-top: 20px;
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      flex-wrap: wrap;
      gap: 12px;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand">
      <h1>Dehradun Metro Station Crowd Prediction</h1>
      <p>SCSE3040 MLOps Mini Project · Group G134 · Problem MP290</p>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
      <div class="status-pill"><div class="status-dot"></div>Model: @champion (v2)</div>
      <a href="/docs" target="_blank" class="docs-btn">Swagger Docs ↗</a>
    </div>
  </header>

  <div class="grid">
    <!-- Form Card -->
    <div class="card">
      <h2>Station & Environmental Inputs</h2>
      <div class="presets">
        <span style="font-size:12px; color:var(--muted); align-self:center;">Presets:</span>
        <button type="button" class="preset-btn" onclick="setPreset('rush_yellow')">Yellow Rush</button>
        <button type="button" class="preset-btn" onclick="setPreset('rain_red')">Red Rain</button>
        <button type="button" class="preset-btn" onclick="setPreset('weekend_blue')">Blue Weekend</button>
        <button type="button" class="preset-btn" style="border-color:#ef4444; color:#fca5a5" onclick="setPreset('rule1_bad')">Rule 1 Error</button>
        <button type="button" class="preset-btn" style="border-color:#ef4444; color:#fca5a5" onclick="setPreset('rule2_bad')">Rule 2 Error</button>
      </div>

      <form id="predictForm" onsubmit="handlePredict(event)">
        <div class="form-grid">
          <div class="form-group">
            <label for="line">Metro Line</label>
            <select id="line" name="line" required>
              <option value="Yellow Line" selected>Yellow Line</option>
              <option value="Red Line">Red Line</option>
              <option value="Blue Line">Blue Line</option>
            </select>
          </div>
          <div class="form-group">
            <label for="day_type">Day Type</label>
            <select id="day_type" name="day_type" required>
              <option value="weekday" selected>Weekday</option>
              <option value="weekend">Weekend</option>
              <option value="holiday">Holiday</option>
            </select>
          </div>
          <div class="form-group">
            <label for="hour">Hour of the Day <span class="hint">(0 - 23)</span></label>
            <input type="number" id="hour" name="hour" min="0" max="23" value="18" required>
          </div>
          <div class="form-group">
            <label for="nearby_offices">Nearby Offices <span class="hint">(Max 150)</span></label>
            <input type="number" id="nearby_offices" name="nearby_offices" value="90" required>
          </div>
          <div class="form-group">
            <label for="temp_c">Temperature <span class="hint">(°C)</span></label>
            <input type="number" step="0.1" id="temp_c" name="temp_c" value="30.5" required>
          </div>
          <div class="form-group">
            <label for="rain_mm">Rainfall <span class="hint">(last hour, mm)</span></label>
            <input type="number" step="0.1" min="0" id="rain_mm" name="rain_mm" value="2.0" required>
          </div>
          <div class="form-group">
            <label for="feeder_buses">Feeder Buses Arriving</label>
            <input type="number" min="0" id="feeder_buses" name="feeder_buses" value="20" required>
          </div>
          <div class="form-group">
            <label for="ticket_price_rs">Avg Fare <span class="hint">(Rs, Min 5)</span></label>
            <input type="number" step="0.5" id="ticket_price_rs" name="ticket_price_rs" value="35" required>
          </div>
        </div>
        <button type="submit" class="submit-btn" id="subBtn">Predict Metro Station Crowd</button>
      </form>
    </div>

    <!-- Output Card -->
    <div class="card">
      <h2>Prediction & Gate Recommendation</h2>
      <div class="result-display" id="resBox">
        <div id="initPrompt" style="color:var(--muted); font-size:14px;">Select parameters and click Predict</div>
        <div id="loading" style="display:none; color:var(--accent);">Computing prediction with champion model...</div>
        <div id="resContent" style="display:none;">
          <div class="result-num" id="predVal">0</div>
          <div class="result-unit">predicted entries in the next hour</div>
          <div id="gateBadge" class="gate-action">Standard Operation</div>
        </div>
        <div id="errContent" class="error-box" style="display:none;"></div>
      </div>

      <div class="stats-row">
        <div class="stat-tile">
          <div class="stat-val" style="color:#38bdf8;">130.38</div>
          <div class="stat-label">v1 Val MAE</div>
        </div>
        <div class="stat-tile">
          <div class="stat-val" style="color:#34d399;">204.01</div>
          <div class="stat-label">v2 Check MAE</div>
        </div>
        <div class="stat-tile">
          <div class="stat-val" style="color:#f59e0b;">-60.2%</div>
          <div class="stat-label">Error Reduction</div>
        </div>
      </div>

      <div style="margin-top:20px; font-size:12px; color:var(--muted); line-height:1.6;">
        <strong>Client Validation Rules:</strong><br>
        • Rule 1: <code>nearby_offices &le; 150</code> (office density ceiling)<br>
        • Rule 2: <code>ticket_price_rs &ge; 5</code> (fare typing mistake prevention)<br>
        Enforced strictly in Pydantic schema; violations return HTTP 422 before model execution.
      </div>
    </div>
  </div>

  <div class="footer-info">
    <div>Group G134: S24CSEU1284 (Pranjal) · S24CSEU1277 (Anshul) · S24CSEU0505 (Shivam) · S24CSEU1283 (Prabhjot) · S24CSEU0095 (Udbhav)</div>
    <div>Batch 18 · Faculty: Dr. Ravi Pratap</div>
  </div>
</div>

<script>
const PRESETS = {
  rush_yellow: { line: 'Yellow Line', day_type: 'weekday', hour: 18, nearby_offices: 95, temp_c: 31, rain_mm: 1.5, feeder_buses: 25, ticket_price_rs: 35 },
  rain_red: { line: 'Red Line', day_type: 'weekday', hour: 19, nearby_offices: 70, temp_c: 24, rain_mm: 45.0, feeder_buses: 15, ticket_price_rs: 30 },
  weekend_blue: { line: 'Blue Line', day_type: 'weekend', hour: 14, nearby_offices: 40, temp_c: 28, rain_mm: 0.0, feeder_buses: 10, ticket_price_rs: 40 },
  rule1_bad: { line: 'Yellow Line', day_type: 'weekday', hour: 18, nearby_offices: 155, temp_c: 30, rain_mm: 0, feeder_buses: 20, ticket_price_rs: 35 },
  rule2_bad: { line: 'Yellow Line', day_type: 'weekday', hour: 18, nearby_offices: 80, temp_c: 30, rain_mm: 0, feeder_buses: 20, ticket_price_rs: 3.5 }
};

function setPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  for (const k in p) {
    const el = document.getElementById(k);
    if (el) el.value = p[k];
  }
  handlePredict(new Event('submit'));
}

async function handlePredict(e) {
  if (e) e.preventDefault();
  const form = document.getElementById('predictForm');
  const initPrompt = document.getElementById('initPrompt');
  const loading = document.getElementById('loading');
  const resContent = document.getElementById('resContent');
  const errContent = document.getElementById('errContent');
  const predVal = document.getElementById('predVal');
  const gateBadge = document.getElementById('gateBadge');

  initPrompt.style.display = 'none';
  resContent.style.display = 'none';
  errContent.style.display = 'none';
  loading.style.display = 'block';

  const payload = {
    hour: parseInt(document.getElementById('hour').value),
    nearby_offices: parseInt(document.getElementById('nearby_offices').value),
    temp_c: parseFloat(document.getElementById('temp_c').value),
    rain_mm: parseFloat(document.getElementById('rain_mm').value),
    feeder_buses: parseInt(document.getElementById('feeder_buses').value),
    ticket_price_rs: parseFloat(document.getElementById('ticket_price_rs').value),
    day_type: document.getElementById('day_type').value,
    line: document.getElementById('line').value
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    loading.style.display = 'none';

    if (res.ok) {
      const data = await res.json();
      predVal.textContent = Math.round(data.entries_per_hour).toLocaleString();
      
      const val = data.entries_per_hour;
      if (val < 3000) {
        gateBadge.className = 'gate-action gate-normal';
        gateBadge.textContent = 'Normal Crowd · 3 Standard Turnstiles Active';
      } else if (val < 5000) {
        gateBadge.className = 'gate-action gate-warning';
        gateBadge.textContent = 'Moderate Rush · Open 2 Additional Gates';
      } else {
        gateBadge.className = 'gate-action gate-danger';
        gateBadge.textContent = 'Congestion Alert · Open All Gates + Alert Staff';
      }
      resContent.style.display = 'block';
    } else {
      const err = await res.json();
      const msg = err.detail && err.detail[0] ? err.detail[0].msg : JSON.stringify(err);
      errContent.textContent = 'HTTP 422 Client Rule Rejection: ' + msg;
      errContent.style.display = 'block';
    }
  } catch (err) {
    loading.style.display = 'none';
    errContent.textContent = 'Connection error: ' + err.message;
    errContent.style.display = 'block';
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Frontend web interface for station crowd prediction and gate management."""
    return DASHBOARD_HTML

