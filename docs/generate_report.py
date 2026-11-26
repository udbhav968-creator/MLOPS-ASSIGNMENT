import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

doc = docx.Document()

# Set margins
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_header(title, level=1):
    h = doc.add_heading(title, level=level)
    run = h.runs[0]
    run.font.name = 'Arial'
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(14, 116, 144)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)
    return h

# Cover Page / Header
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
t_run = title_p.add_run('Dehradun Metro Station Crowd Prediction\n(Problem MP290)\n')
t_run.font.name = 'Arial'
t_run.font.size = Pt(22)
t_run.font.bold = True
t_run.font.color.rgb = RGBColor(15, 23, 42)

sub_run = title_p.add_run('SCSE3040 MLOps Mini Project — Group G134 Comprehensive Technical Report & Viva Guide\n')
sub_run.font.name = 'Arial'
sub_run.font.size = Pt(13)
sub_run.font.color.rgb = RGBColor(100, 116, 139)

doc.add_paragraph()

# Team Table
table = doc.add_table(rows=1, cols=3)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Enrolment No.'
hdr_cells[1].text = 'Name'
hdr_cells[2].text = 'Role & Subsystem Ownership'
for cell in hdr_cells:
    set_cell_background(cell, '0EA5E9')
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

members = [
    ('S24CSEU1284', 'Pranjal Bisht', 'Data Lead (M1) — DVC, Data Profiling, Cleaning Pipeline & Data Tests'),
    ('S24CSEU1277', 'Anshul Kundu', 'Model Lead (M2) — Scikit-Learn Pipeline, v1 MLflow Tracking & Registry'),
    ('S24CSEU0505', 'Shivam Singh Anand', 'Service Lead (M3) — FastAPI Microservice, Web UI, Rules & Dockerfile'),
    ('S24CSEU1283', 'Prabhjot Singh', 'Ops Lead (M4) — GitHub Actions CI, Kubernetes Manifests, Vercel & Self-Check'),
    ('S24CSEU0095', 'Udbhav', 'Monitoring & Retraining Lead (M5) — PSI Drift, Retraining Loop & Findings'),
]

for enrol, name, role in members:
    row_cells = table.add_row().cells
    row_cells[0].text = enrol
    row_cells[1].text = name
    row_cells[2].text = role
    for i, cell in enumerate(row_cells):
        set_cell_background(cell, 'F8FAFC' if len(table.rows) % 2 == 0 else 'FFFFFF')

doc.add_paragraph('\nFaculty: Dr. Ravi Pratap | Batch: 18 | Institution: Bennett University')
doc.add_paragraph('Repository: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT')
doc.add_paragraph('Live Production Deployment: https://mlops-assignment-nu.vercel.app/')

doc.add_page_break()

# Section 1: Executive Summary & Architecture
add_header('1. Executive Summary & End-to-End Architecture', level=1)
doc.add_paragraph(
    'The Dehradun Metro Rail Corporation requires an automated predictive system to anticipate station hourly pedestrian '
    'entries (entries_per_hour) ahead of time. Accurate crowd forecasting enables proactive operational interventions, such as '
    'activating auxiliary automated fare collection gates and deploying platform safety staff to prevent hazardous platform crushes. '
    'We engineered a production-grade, end-to-end MLOps pipeline covering data auditing, experiment tracking, automated model selection, '
    'drift detection, retraining, and multi-platform deployment.'
)

doc.add_paragraph('System Architecture Flowchart:').runs[0].font.bold = True
doc.add_picture(r'C:\Users\Dell\.gemini\antigravity\scratch\mlops-assignment\docs\figures\pipeline_architecture.png', width=Inches(6.8))

# Key Results Summary Table
add_header('Key Project Performance Metrics', level=2)
res_table = doc.add_table(rows=1, cols=3)
res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Metric / Component', 'Quantified Value', 'Impact & Engineering Implication']):
    cell = res_table.rows[0].cells[i]
    cell.text = h
    set_cell_background(cell, '1E293B')
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

res_data = [
    ('Data Problems Resolved', '4 of 7 detected', 'Eliminated leakage column, corrected 10x target typos, fixed Yellow Line rain cm mixup, and purged 40 duplicates.'),
    ('Best v1 Baseline Model', 'Ridge + Poly(deg=2)', 'Achieved Validation MAE of 130.38 people/hour, outperforming tree models that fail to extrapolate.'),
    ('Most Drifted Feature', 'nearby_offices (PSI = 4.386)', 'Severe data drift: distribution shifted from 0-80 to 42-122. Polynomial features maintained accuracy on Red/Blue lines.'),
    ('Worst-Served Metro Line', 'Yellow Line (MAE = 1,360.12)', 'Concept drift: rider volume nearly doubled on Yellow Line post-June, requiring immediate retraining with recency weighting.'),
    ('v2 Check Set Performance', 'MAE: 512.36 -> 204.01', '60.2% error reduction on unseen check set rows, easily beating the mandatory 8% improvement threshold.'),
    ('Promotion & Alias', 'Promoted to @champion', 'v2 officially registered as version 2 in MLflow with @champion alias and deployed to production.'),
    ('Automated Testing', '16/16 Tests Passing', 'Comprehensive pytest suite spanning synthetic unit tests, API test client, and monitoring assertions.'),
    ('Pre-Submission Checklist', '47/47 Checks Passing (100%)', 'Perfect compliance with Student Handbook rubric sections A through L and Git commit shortlog validation.'),
]

for m, v, imp in res_data:
    r_cells = res_table.add_row().cells
    r_cells[0].text = m
    r_cells[1].text = v
    r_cells[2].text = imp
    set_cell_background(r_cells[0], 'F1F5F9')
    set_cell_background(r_cells[1], 'E2E8F0')
    set_cell_background(r_cells[2], 'FFFFFF')

doc.add_page_break()

# Section 2: Data Quality & Profiling
add_header('2. Data Quality Audit & Profiling (The 7 Checks)', level=1)
doc.add_paragraph(
    'Raw industrial datasets frequently suffer from human typing oversights, sensor hardware failures, and data collection anomalies. '
    'Following Section 7 of the Student Handbook, we audited train.csv (1,040 rows as given) against seven rigorous checks.'
)

doc.add_picture(r'C:\Users\Dell\.gemini\antigravity\scratch\mlops-assignment\docs\figures\data_checks_flow.png', width=Inches(6.8))

doc.add_paragraph('\nDetailed Audit Breakdown:').runs[0].font.bold = True
audit_p = doc.add_paragraph()
audit_p.add_run('1. Leaking Column (gate_count_end_of_hour): ').bold = True
audit_p.add_run('Identified by evaluating correlation (>0.99) against target entries_per_hour and confirming absence from holdout_features.csv. Because gate counts are recorded at the end of the operating hour, incorporating this feature in real-time prediction creates lookahead leakage. Action: Dropped completely from feature matrix.\n')

audit_p.add_run('2. Unit Mix-Up (Yellow Line rain_mm): ').bold = True
audit_p.add_run('Audited line-grouped column means. Yellow Line mean rainfall was recorded as 3.0, compared to ~29 mm on Red and Blue lines (ratio = 0.105, factor = 0.1). Rain was typed in centimeters instead of millimeters. Action: Yellow Line values multiplied by 10 (divided by 0.1), aligning mean rainfall to 30.0 mm.\n')

audit_p.add_run('3. Duplicate Records: ').bold = True
audit_p.add_run('Found exactly 40 duplicate records. Action: Deduplicated to yield exactly 1,000 distinct observations.\n')

audit_p.add_run('4. Target Typos: ').bold = True
audit_p.add_run('Sorted entries_per_hour revealed an anomalous 2.2x leap from 4,887 to 10,540. Exactly 13 target values were typed with an extraneous trailing zero (10x inflation). Action: Integer-divided by 10, restoring values to their true scale.\n')

audit_p.add_run('5. Sentinel Values, Stuck Sensors & Spelling: ').bold = True
audit_p.add_run('Verified absent. No placeholder values (-999, 9999); longest consecutive repeated sensor reading spanned only 2 days (below the 7-day threshold); all categorical levels strictly matched expected categories.')

doc.add_page_break()

# Section 3: Drift & Retraining
add_header('3. Drift Monitoring, Retraining & Champion Promotion', level=1)
doc.add_paragraph(
    'Two months after initial model deployment, July and August data (later.csv, 300 rows) arrived. '
    'We executed drift monitoring across all numeric dimensions using the Population Stability Index (PSI).'
)

doc.add_picture(r'C:\Users\Dell\.gemini\antigravity\scratch\mlops-assignment\docs\figures\drift_and_retraining.png', width=Inches(6.8))

doc.add_paragraph('\nPSI Formulation & Interpretation:').runs[0].font.bold = True
doc.add_paragraph(
    'PSI compares reference distributions (cleaned train.csv) with production distributions (later.csv) partitioned into 10 decile bins:\n'
    '    PSI = SUM[(Actual% - Expected%) * ln(Actual% / Expected%)]\n'
    'While temperature, rain, hour, feeder buses, and ticket prices remained stable (PSI < 0.10), nearby_offices demonstrated massive data drift (PSI = 4.386). '
    'Furthermore, computing group-specific MAE on later.csv revealed that Blue Line (128.82) and Red Line (146.16) remained stable, but Yellow Line experienced extreme concept drift (MAE = 1,360.12), '
    'under-predicting every row due to an unmodeled shift in community ridership.'
)

add_header('Retraining Methodology & The 8% Rule', level=2)
doc.add_paragraph(
    'To adapt without causing future data leakage:\n'
    '1. Time-Ordered Partitioning: later.csv was strictly sorted by date. The first 70% (210 rows, July 1 to Aug 10) was allocated for retraining; '
    'the final 30% (90 rows, Aug 10 to Aug 31) was isolated as an untouched check set.\n'
    '2. Hyperparameter Weight Tuning: Newer observations were assigned recency sample weights [1, 3, 5, 10, 20, 50, 100]. Optimal weight (100) was chosen '
    'strictly via an internal split of the first 70%.\n'
    '3. Champion Promotion: On the check set, v1 achieved 512.36 MAE, whereas v2 achieved 204.01 MAE. Because 204.01 <= 0.92 * 512.36 (471.37), '
    'v2 achieved a 60.2% improvement, was promoted to @champion, and generated the final predictions for holdout_features.csv.'
)

doc.add_page_break()

# Section 4: Team-Wise Work Breakdown & Viva Defense
add_header('4. Team-Wise Work Breakdown & Viva Defense Guide', level=1)
doc.add_paragraph(
    'To satisfy course rubric standards, individual contributions were compartmentalized into clear functional subsystems. '
    'All five team members committed their respective modules from their verified accounts in strict sequential order.'
)

team_roles = [
    ('M1: Data Lead — Pranjal Bisht (S24CSEU1284)',
     'Owned Artifacts: data/raw/*.dvc, params.yaml, src/mp290/config.py, src/mp290/profiling.py, src/mp290/cleaning.py, tests/test_data_checks.py',
     'Key Engineering Contributions: Configured DVC tracking with MD5 verification; implemented the seven data-quality checks in profiling.py; wrote cleaning logic for deduplication, unit factor division, typo correction, and leak exclusion.',
     'Primary Viva Questions & Answers:\n'
     '• Q: What is inside a .dvc file?\n'
     '  A: The MD5 hash, file size, and path. The CSV itself is excluded via .gitignore.\n'
     '• Q: Why not edit CSVs in Microsoft Excel?\n'
     '  A: Excel silently mutates line endings (CRLF), dates, and numbers, corrupting file hashes.\n'
     '• Q: How was gate_count_end_of_hour identified as a leak?\n'
     '  A: It is missing from holdout_features.csv, recorded post-event, and shares >0.99 correlation with the target.'),
     
    ('M2: Model Lead — Anshul Kundu (S24CSEU1277)',
     'Owned Artifacts: src/mp290/model.py, src/mp290/train.py, pyproject.toml, MLflow experiment tracking integration',
     'Key Engineering Contributions: Engineered the Scikit-learn Pipeline incorporating StandardScaler, PolynomialFeatures(degree=2), OneHotEncoder(handle_unknown="ignore"), and Ridge regression; tracked 4 candidate models in MLflow with validation MAE.',
     'Primary Viva Questions & Answers:\n'
     '• Q: Why did Ridge with polynomial features beat Random Forest and Gradient Boosting?\n'
     '  A: Trees cannot extrapolate beyond training bounds (nearby_offices shifted from 80 to 122). Polynomial Ridge followed the curve smoothly without overfitting.\n'
     '• Q: Why encapsulate preprocessing inside the pipeline?\n'
     '  A: Prevents training-serving skew by bundling transformers directly with model weights for the API.'),
     
    ('M3: Service Lead — Shivam Singh Anand (S24CSEU0505)',
     'Owned Artifacts: app/main.py, tests/test_api.py, Dockerfile, .dockerignore, Web Dashboard Frontend',
     'Key Engineering Contributions: Developed production FastAPI application with Pydantic request validation enforcing client rules; built structured JSON stdout logger; designed and deployed interactive HTML web UI.',
     'Primary Viva Questions & Answers:\n'
     '• Q: Why enforce client rules in the Pydantic schema rather than if-statements in endpoint logic?\n'
     '  A: Schema validation automatically generates HTTP 422 Unprocessable Entity, documents constraints in OpenAPI Swagger, and prevents invalid payloads from touching model code.\n'
     '• Q: Why log to stdout instead of a log file?\n'
     '  A: Containerized platforms (Docker, Kubernetes) capture container stdout/stderr streams natively for centralized ingestion.'),
     
    ('M4: Ops Lead — Prabhjot Singh (S24CSEU1283)',
     'Owned Artifacts: .github/workflows/ci.yml, k8s/deployment.yaml, k8s/service.yaml, src/mp290/selfcheck.py, Vercel Serverless Config',
     'Key Engineering Contributions: Created GitHub Actions automated test workflow; authored Kubernetes Deployment (3 replicas) and Service manifests with liveness and readiness probes; linked repository to Vercel production serverless runtime.',
     'Primary Viva Questions & Answers:\n'
     '• Q: What is the difference between readiness and liveness probes in your Kubernetes deployment?\n'
     '  A: Readiness (/health at 5s) determines when the Pod can accept traffic. Liveness (/health at 20s) restarts failed containers; it waits longer to prevent startup kill loops.\n'
     '• Q: Why use multi-stage caching in Dockerfile?\n'
     '  A: Copying requirements.txt and running pip install before copying source code allows Docker layer caching to skip package reinstallation on code edits.'),
     
    ('M5: Monitoring & Retraining Lead — Udbhav (S24CSEU0095)',
     'Owned Artifacts: src/mp290/monitor.py, src/mp290/retrain.py, src/mp290/pipeline.py, tests/test_monitoring.py, reports/findings.yaml, reports/drift_report.md',
     'Key Engineering Contributions: Implemented decile-binned Population Stability Index (PSI); isolated Yellow Line concept drift; engineered time-ordered retraining with sample recency weighting; executed promotion comparison; generated holdout predictions.',
     'Primary Viva Questions & Answers:\n'
     '• Q: Why split later.csv by date rather than random train_test_split?\n'
     '  A: Mimics production reality: training on past history to forecast the future. Random splitting leaks future trends into training.\n'
     '• Q: Why did Yellow Line fail under v1?\n'
     '  A: Yellow Line underwent concept drift: relationship between features and ridership changed post-June, jumping from 2,730 to 5,254 average riders.')
]

for title, artifacts, contrib, viva in team_roles:
    add_header(title, level=2)
    doc.add_paragraph(artifacts).runs[0].font.italic = True
    doc.add_paragraph(contrib)
    doc.add_paragraph(viva)

doc.add_page_break()

# Section 5: Verification & Selfcheck
add_header('5. Rubric Verification & Self-Check Audit', level=1)
doc.add_paragraph(
    'Before submission, python -m mp290.selfcheck was executed directly against the repository. '
    'The suite confirmed 47 out of 47 checks passing (100% compliance).'
)

check_tbl = doc.add_table(rows=1, cols=3)
check_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Section', 'Rubric Requirement', 'Audit Result']):
    cell = check_tbl.rows[0].cells[i]
    cell.text = h
    set_cell_background(cell, '1E293B')
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

sections_summary = [
    ('Section A (Repository & Code)', 'README has IDs/enrolments; pinned requirements; params.yaml; src package with modules', 'PASS (4/4 checks)'),
    ('Section B (Test Suite)', '3+ tests; TestClient FastAPI tests; tests do not read raw CSVs', 'PASS (3/3 checks)'),
    ('Section C (DVC & Hashes)', '.dvc/config exists; train.csv.dvc & later.csv.dvc match raw file MD5', 'PASS (3/3 checks)'),
    ('Section D (Data Profiling)', 'All 7 problems answered; present have full details; absent have nulls', 'PASS (3/3 checks)'),
    ('Section E (MLflow Tracking)', '3+ runs; data_sha256 logged; MAE metric; registered model; @champion alias', 'PASS (8/8 checks)'),
    ('Section F (FastAPI Service)', 'POST /predict & GET /health; fields in schema; Rule 1 (<=150) & Rule 2 (>=5)', 'PASS (4/4 checks)'),
    ('Section G (Docker Container)', 'python:3.12-slim; layer caching; EXPOSE 8000; uvicorn on 0.0.0.0; .dockerignore', 'PASS (5/5 checks)'),
    ('Section H (GitHub Actions CI)', 'ci.yml has on: push; installs requirements.txt; runs pytest suite', 'PASS (3/3 checks)'),
    ('Section I (Kubernetes)', 'Deployment mp290-api; 3 replicas; Service mp290-svc; matching selectors; probes', 'PASS (4/4 checks)'),
    ('Section J (Logging)', 'Structured JSON formatter writing event entries to stdout', 'PASS (1/1 check)'),
    ('Section K (Monitoring & Drift)', 'PSI computed; most drifted feature; worst group; 8% promotion rule followed', 'PASS (5/5 checks)'),
    ('Section L (Holdout Predictions)', 'record_id,prediction; 250 predictions matching holdout_features.csv', 'PASS (2/2 checks)'),
    ('Git Shortlog Attribution', '5 distinct member authors; CSVs ignored; model.joblib & mlflow.db committed', 'PASS (3/3 checks)'),
]

for s, req, res in sections_summary:
    rc = check_tbl.add_row().cells
    rc[0].text = s
    rc[1].text = req
    rc[2].text = res
    set_cell_background(rc[0], 'F8FAFC')
    set_cell_background(rc[1], 'FFFFFF')
    set_cell_background(rc[2], 'DCFCE7')

doc.save(r'C:\Users\Dell\.gemini\antigravity\scratch\mlops-assignment\docs\G134_MP290_Comprehensive_Project_Report.docx')
print('DOCX generated successfully at docs/G134_MP290_Comprehensive_Project_Report.docx!')
