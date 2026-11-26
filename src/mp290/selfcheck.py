"""
Pre-submission self-check, following Student Handbook sections 5, 11 and 13.

    python -m mp290.selfcheck

It cannot see GitHub (public link, green Actions run, the form). Everything
inside the repository it checks, and prints PASS / FAIL with a reason.
"""

import hashlib
import re
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from mp290.config import ROOT

ENROLMENTS = ["S24CSEU1284", "S24CSEU1277", "S24CSEU0505", "S24CSEU1283", "S24CSEU0095"]
results: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, why: str = "") -> None:
    results.append((section, name, bool(ok), why))


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def section_a():
    readme = read("README.md")
    check("A", "README has group id, problem id, all enrolment numbers",
          all(x in readme for x in ["G134", "MP290", *ENROLMENTS]))
    req = [l.strip() for l in read("requirements.txt").splitlines() if l.strip() and not l.startswith("#")]
    unpinned = [l for l in req if "==" not in l]
    check("A", "every requirement pinned with ==", req and not unpinned, f"unpinned: {unpinned}")
    check("A", "params.yaml exists and code reads it with yaml.safe_load",
          (ROOT / "params.yaml").exists() and "yaml.safe_load" in read("src/mp290/config.py"))
    modules = list((ROOT / "src/mp290").glob("*.py"))
    check("A", "src/<package>/ with __init__.py and 2+ modules",
          (ROOT / "src/mp290/__init__.py").exists() and len(modules) >= 3)


def section_b():
    tests = list((ROOT / "tests").glob("test_*.py"))
    text = "".join(t.read_text(encoding="utf-8") for t in tests)
    n = len(re.findall(r"^def test_", text, re.M))
    check("B", "3+ pytest tests", n >= 3, f"{n} found")
    check("B", "a TestClient test", "TestClient" in text)
    check("B", "tests do not read the CSVs", "read_csv" not in text)


def section_c():
    check("C", ".dvc/config exists (dvc init)", (ROOT / ".dvc/config").exists())
    for name in ["train", "later"]:
        dvc_file, csv = ROOT / f"data/raw/{name}.csv.dvc", ROOT / f"data/raw/{name}.csv"
        if not dvc_file.exists():
            check("C", f"{name}.csv.dvc exists", False)
            continue
        md5 = yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
        ok = not csv.exists() or hashlib.md5(csv.read_bytes()).hexdigest() == md5
        check("C", f"{name}.csv.dvc hash matches the CSV", ok, "CSV changed? (opened in Excel?)")


def findings():
    return yaml.safe_load(read("reports/findings.yaml")) or {}


def section_d():
    f = findings()
    issues = f.get("data_issues", {})
    check("D", "all 7 data problems answered", len(issues) == 7 and
          all(isinstance(v.get("present"), bool) for v in issues.values()))
    bad = [k for k, v in issues.items() if v.get("present") and any(x is None for x in v.values())]
    check("D", "present problems have every detail filled", not bad, f"missing in {bad}")
    bad = [k for k, v in issues.items() if v.get("present") is False
           and any(x is not None for kk, x in v.items() if kk != "present")]
    check("D", "absent problems have only nulls", not bad, f"{bad}")


def section_e():
    db = ROOT / "mlflow.db"
    if not db.exists():
        check("E", "mlflow.db exists", False)
        return
    con = sqlite3.connect(db)
    q = lambda sql: con.execute(sql).fetchall()
    sha = hashlib.sha256((ROOT / "data/raw/train.csv").read_bytes()).hexdigest() \
        if (ROOT / "data/raw/train.csv").exists() else None
    runs = [r[0] for r in q("select run_uuid from runs where lifecycle_stage='active'")]
    check("E", "3+ MLflow runs", len(runs) >= 3, f"{len(runs)} runs")
    hashes = dict(q("select run_uuid, value from params where key='data_sha256'"))
    check("E", "every run logs data_sha256", all(r in hashes for r in runs))
    if sha:
        check("E", "data_sha256 equals hash of your train.csv",
              all(hashes.get(r) == sha for r in runs))
    check("E", "a metric whose name contains 'mae'",
          bool(q("select 1 from metrics where key like '%mae%' limit 1")))
    t = findings().get("tracking", {})
    check("E", "best_run_id in findings is a real run", t.get("best_run_id") in runs)
    name = t.get("registered_model_name")
    versions = [v[0] for v in q(f"select version from model_versions where name='{name}'")]
    check("E", "2+ registered versions", len(versions) >= 2, f"versions {versions}")
    alias = q(f"select version from registered_model_aliases where name='{name}' and alias='champion'")
    check("E", "@champion alias set and matches findings",
          alias and alias[0][0] == t.get("champion_version"), f"alias {alias}")


def section_f():
    app = read("app/main.py")
    check("F", "POST /predict and GET /health", '@app.post("/predict"' in app and '@app.get("/health")' in app)
    fields = ["hour", "nearby_offices", "temp_c", "rain_mm", "feeder_buses",
              "ticket_price_rs", "day_type", "line"]
    check("F", "every request field in the schema", all(re.search(rf"^\s+{f}:", app, re.M) for f in fields))
    check("F", "rule 1: nearby_offices le=150", re.search(r"nearby_offices:.*le=150", app))
    check("F", "rule 2: ticket_price_rs ge=5", re.search(r"ticket_price_rs:.*ge=5", app))


def section_g():
    d = read("Dockerfile")
    check("G", "python:3.12-slim base", "FROM python:3.12-slim" in d)
    i_req, i_inst = d.find("COPY requirements.txt"), d.find("pip install")
    i_code = min([d.find(x) for x in ["COPY src", "COPY app"] if d.find(x) >= 0] or [-1])
    check("G", "requirements installed before code copied", 0 <= i_req < i_inst < i_code)
    check("G", "EXPOSE 8000", "EXPOSE 8000" in d)
    check("G", "uvicorn on 0.0.0.0:8000", "0.0.0.0" in d and "8000" in d.split("CMD")[-1])
    check("G", ".dockerignore exists", (ROOT / ".dockerignore").exists())


def section_h():
    ci = read(".github/workflows/ci.yml")
    first = [l for l in ci.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("name")]
    check("H", "ci.yml uses the key on: (not true:)", first and first[0].startswith("on:"))
    check("H", "runs on push", "push" in ci)
    check("H", "installs requirements.txt and runs pytest",
          "requirements.txt" in ci and "pytest" in ci)


def section_i():
    docs = {}
    for f in ["k8s/deployment.yaml", "k8s/service.yaml"]:
        for doc in yaml.safe_load_all(read(f)):
            if doc:
                docs[doc["kind"]] = doc
    dep, svc = docs.get("Deployment", {}), docs.get("Service", {})
    spec = dep.get("spec", {})
    c = (spec.get("template", {}).get("spec", {}).get("containers") or [{}])[0]
    check("I", "Deployment mp290-api, 3 replicas, image mp290-api:1.0",
          dep.get("metadata", {}).get("name") == "mp290-api" and spec.get("replicas") == 3
          and c.get("image") == "mp290-api:1.0")
    check("I", "containerPort 8000", any(p.get("containerPort") == 8000 for p in c.get("ports", [])))
    labels = spec.get("template", {}).get("metadata", {}).get("labels", {})
    check("I", "Service mp290-svc, port 80 -> 8000, selector matches Pod labels",
          svc.get("metadata", {}).get("name") == "mp290-svc"
          and any(p.get("port") == 80 and p.get("targetPort") == 8000 for p in svc["spec"]["ports"])
          and svc["spec"]["selector"].items() <= labels.items()
          and spec.get("selector", {}).get("matchLabels", {}).items() <= labels.items())
    r, l = c.get("readinessProbe", {}), c.get("livenessProbe", {})
    check("I", "readiness and liveness on /health, liveness waits longer",
          r.get("httpGet", {}).get("path") == "/health" and l.get("httpGet", {}).get("path") == "/health"
          and l.get("initialDelaySeconds", 0) > r.get("initialDelaySeconds", 0))


def section_j():
    app = read("app/main.py")
    check("J", "logging module with a JSON formatter to stdout",
          "logging.Formatter" in app and "json.dumps" in app and "stdout" in app)


def section_k():
    m = findings().get("monitoring", {})
    psi, groups = m.get("psi", {}), m.get("mae_by_group", {})
    check("K", "PSI for every numeric column", psi and all(v is not None for v in psi.values()))
    check("K", "most_drifted_feature = column with highest PSI",
          psi and m.get("most_drifted_feature") == max(psi, key=psi.get))
    check("K", "worst_served_group = group with highest MAE",
          groups and m.get("worst_served_group") == max(groups, key=groups.get))
    v1, v2 = m.get("mae_v1_on_check"), m.get("mae_v2_on_check")
    rule = "promote" if v1 and v2 and v2 <= 0.92 * v1 else "keep"
    check("K", "decision follows the 8% rule", m.get("decision") == rule, f"rule says {rule}")
    champ = findings().get("tracking", {}).get("champion_version")
    check("K", "champion is v2 if promote, v1 if keep",
          champ == (2 if m.get("decision") == "promote" else 1))


def section_l():
    p = ROOT / "predictions/holdout_predictions.csv"
    h = ROOT / "data/raw/holdout_features.csv"
    if not p.exists():
        check("L", "predictions file exists", False)
        return
    pred = pd.read_csv(p)
    check("L", "exactly the columns record_id,prediction", list(pred.columns) == ["record_id", "prediction"])
    if h.exists():
        ids = pd.read_csv(h)["record_id"]
        check("L", "one row per holdout record, no gaps or repeats",
              len(pred) == len(ids) and set(pred.record_id) == set(ids)
              and not pred.record_id.duplicated().any() and pred.prediction.notna().all())


def section_git():
    if not (ROOT / ".git").exists():
        check("Git", "repository is a git repo", False, "run git init / clone")
        return
    out = subprocess.run(["git", "-C", str(ROOT), "shortlog", "-sne", "HEAD"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    check("Git", "commits from 5 different people", len(out) >= 5, f"{len(out)} authors")
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout
    check("Git", "train.csv / later.csv NOT in git, .dvc files are",
          "data/raw/train.csv\n" not in tracked and "data/raw/train.csv.dvc" in tracked)
    check("Git", "mlflow.db, model.joblib, predictions committed",
          all(x in tracked for x in ["mlflow.db", "models/model.joblib",
                                     "predictions/holdout_predictions.csv"]))


def main() -> int:
    for fn in [section_a, section_b, section_c, section_d, section_e, section_f, section_g,
               section_h, section_i, section_j, section_k, section_l, section_git]:
        try:
            fn()
        except Exception as err:                     # one broken section must not hide the rest
            check(fn.__name__.split("_")[-1].upper(), "section could be checked", False, repr(err))
    failed = 0
    for sec, name, ok, why in results:
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {sec:3s} {name}" + ("" if ok or not why else f"   ({why})"))
    print(f"\n{len(results) - failed}/{len(results)} checks pass.")
    print("Still check by hand: repo is PUBLIC, Actions tab is GREEN, every member filled the form.")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
